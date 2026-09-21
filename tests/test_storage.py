"""SQLite storage: the database schema and the functions that read it."""
import array
import json
import sqlite3
import uuid

import numpy as np
import pytest

from harness import image_frames, line_acquisitions
from sensor_core.memory.strg_manager import (SCHEMA_VERSION, StorageManager, add_session, connect, insert_frames,
                                             list_sessions, load_channel, load_frame_times, load_frames,
                                             load_images)

KEYS = ["red", "infrared", "violet"]
UNIX_ANCHOR = 1_700_000_000_000_000_000  # wall clock at the start of a test session, in ns
MONO_ANCHOR = 5_000_000  # the host's monotonic clock at the same instant


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "db.sqlite3")


def store(db, frames, times, data_mode="line", channels=KEYS, session=None):
    """Store frames the way the ingester does; return the session."""
    session = session or {"uuid": uuid.uuid4().hex, "started_unix_ns": UNIX_ANCHOR,
                          "started_monotonic_ns": MONO_ANCHOR}
    conn = connect(db)
    with conn:
        session_id = add_session(conn, session, data_mode, frames.shape[1:], frames.dtype, channels)
        insert_frames(conn, session_id, [(i, t, frame.tobytes()) for i, (t, frame) in enumerate(zip(times, frames))])
    conn.close()
    return session


def test_a_new_database_has_the_tables_and_no_sessions(db):
    StorageManager(KEYS, db).create_serial_database()
    conn = sqlite3.connect(db)
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    version = conn.execute("SELECT value FROM sensor_core WHERE key = 'schema_version'").fetchone()[0]
    conn.close()
    assert {"sensor_core", "sessions", "frames"} <= tables
    assert int(version) == SCHEMA_VERSION
    assert list_sessions(db) == []


def test_line_channels_load_in_acquisition_order(db):
    acquisitions = line_acquisitions(5, window=10, channels=3)
    store(db, acquisitions, range(5))
    for channel, key in enumerate(KEYS):
        np.testing.assert_array_equal(load_channel(db, key), acquisitions[:, :, channel].ravel())
        np.testing.assert_array_equal(StorageManager.load_serial_channel(key, filepath=db),
                                      acquisitions[:, :, channel].ravel())


def test_frames_load_with_their_index_and_time(db):
    acquisitions = line_acquisitions(4, window=10, channels=3)
    store(db, acquisitions, [MONO_ANCHOR + 10, MONO_ANCHOR + 20, MONO_ANCHOR + 30, MONO_ANCHOR + 40])
    index, t_ns, frames = load_frames(db)
    np.testing.assert_array_equal(index, [0, 1, 2, 3])
    np.testing.assert_array_equal(t_ns, [MONO_ANCHOR + 10, MONO_ANCHOR + 20, MONO_ANCHOR + 30, MONO_ANCHOR + 40])
    np.testing.assert_array_equal(frames, acquisitions)


def test_frame_times_convert_to_the_wall_clock(db):
    store(db, line_acquisitions(3, window=10, channels=3), [MONO_ANCHOR, MONO_ANCHOR + 1_000, MONO_ANCHOR + 2_500])
    np.testing.assert_array_equal(load_frame_times(db, clock="monotonic"),
                                  [MONO_ANCHOR, MONO_ANCHOR + 1_000, MONO_ANCHOR + 2_500])
    np.testing.assert_array_equal(load_frame_times(db), [UNIX_ANCHOR, UNIX_ANCHOR + 1_000, UNIX_ANCHOR + 2_500])


def test_sessions_can_be_loaded_together_or_one_at_a_time(db):
    first = line_acquisitions(2, window=10, channels=3)
    second = line_acquisitions(3, window=10, channels=3) + 1000
    first_session = store(db, first, range(2))
    store(db, second, range(3))
    sessions = list_sessions(db)
    assert [s["frames"] for s in sessions] == [2, 3]
    assert sessions[0]["channels"] == KEYS and sessions[0]["frame_shape"] == (10, 3)
    red = lambda frames: frames[:, :, 0].ravel()  # noqa: E731
    np.testing.assert_array_equal(load_channel(db, "red"), np.concatenate([red(first), red(second)]))
    np.testing.assert_array_equal(load_channel(db, "red", session=-1), red(second))
    np.testing.assert_array_equal(load_channel(db, "red", session=sessions[0]["id"]), red(first))
    np.testing.assert_array_equal(load_channel(db, "red", session=first_session["uuid"]), red(first))
    with pytest.raises(ValueError, match="no session"):
        load_channel(db, "red", session=99)


def test_storing_a_frame_twice_keeps_one_copy(db):
    acquisitions = line_acquisitions(3, window=10, channels=3)
    session = store(db, acquisitions, range(3))
    store(db, acquisitions, range(3), session=session)  # e.g., a segment ingested again after a crash
    assert list_sessions(db)[0]["frames"] == 3


def test_images_load_with_their_shape_and_dtype(db):
    frames = image_frames(4, (6, 4, 1), np.uint8)
    store(db, frames, range(4), data_mode="image", channels=["camera"])
    images = load_images(db)
    assert images.dtype == np.uint8
    np.testing.assert_array_equal(images, frames)


def test_the_file_is_readable_without_sensor_core_or_numpy(db):
    acquisitions = line_acquisitions(2, window=10, channels=3)
    store(db, acquisitions, range(2))
    conn = sqlite3.connect(db)
    frame_shape, dtype, channels = conn.execute("SELECT frame_shape, dtype, channels FROM sessions").fetchone()
    blob = conn.execute("SELECT data FROM frames WHERE frame_index = 0").fetchone()[0]
    conn.close()
    assert (json.loads(frame_shape), dtype, json.loads(channels)) == ([10, 3], "<f4", KEYS)
    values = array.array("f", blob)  # float32, row-major (window_size, channels)
    assert list(values[0::3]) == list(acquisitions[0, :, 0])  # the red channel of the first acquisition


def test_loading_an_unknown_key_raises(db):
    store(db, line_acquisitions(1, window=10, channels=3), [0])
    with pytest.raises(ValueError, match="not in sqlite3 file"):
        load_channel(db, "green")


def test_loading_a_missing_database_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_channel(str(tmp_path / "missing.sqlite3"), "red")


@pytest.mark.parametrize("name", ["db.h5", "db.sql", "db"])
def test_only_sqlite3_files_are_supported(tmp_path, name):
    with pytest.raises(ValueError, match="unsupported"):
        StorageManager(KEYS, str(tmp_path / name))


def test_databases_written_by_sensor_core_1_x_are_flagged(db):
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE unnamed (key TEXT PRIMARY KEY, value BLOB)")  # the old sqlitedict layout
    conn.close()
    with pytest.warns(UserWarning, match="sensor_core 1.x"):
        connect(db).close()


def test_databases_from_a_newer_schema_are_refused(db):
    connect(db).close()
    conn = sqlite3.connect(db)
    with conn:
        conn.execute("UPDATE sensor_core SET value = ? WHERE key = 'schema_version'", (str(SCHEMA_VERSION + 1),))
    conn.close()
    with pytest.raises(ValueError, match="schema"):
        connect(db)


def test_offline_plotting_loads_the_requested_channels(db):
    from sensor_core.plot import PlotManager  # loads data only; drawing needs a GPU

    acquisitions = line_acquisitions(5, window=10, channels=3)
    store(db, acquisitions, range(5))
    _, ys = PlotManager.offline_initialize_data(db, [["violet"], ["red"]])
    np.testing.assert_array_equal(ys[0], acquisitions[:, :, 2].ravel())
    np.testing.assert_array_equal(ys[1], acquisitions[:, :, 0].ravel())
