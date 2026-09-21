"""SensorManager configuration and lifecycle: starting, flushing, stopping, and closing."""
import itertools
import time

import numpy as np
import pytest

from sensor_core import SensorManager
from sensor_core.memory.ring_adapter import RingBuffer
from sensor_core.memory.strg_manager import StorageManager

KEYS = ["red", "infrared", "violet"]
WINDOW = 10  # samples per acquisition


def test_plot_keys_default_to_one_row_with_every_channel():
    ser_keys, plot_keys = SensorManager.setup_channel_keys(KEYS)
    assert ser_keys == KEYS
    assert plot_keys == [KEYS]


def test_plot_keys_can_arrange_channels_in_a_grid():
    grid = [["violet"], ["red"]]
    assert SensorManager.setup_channel_keys(KEYS, plot_channel_key=grid)[1] == grid


def test_plot_keys_must_be_serial_keys():
    with pytest.raises(KeyError):
        SensorManager.setup_channel_keys(KEYS, plot_channel_key=[["red", "green"]])


def test_serial_keys_must_be_one_dimensional():
    with pytest.raises(ValueError, match="one-dimensional"):
        SensorManager.setup_channel_keys([["red", "infrared"]])


def counting_acquisition():
    """A custom acquisition function whose values encode their position in the stream."""
    counter = itertools.count()

    def acquire(ser, frame_shape):
        k = next(counter)
        time.sleep(0.002)
        samples = np.arange(WINDOW)[:, None]
        channels = np.arange(len(KEYS))[None, :]
        return ((k * WINDOW + samples) * len(KEYS) + channels).astype(np.float32)

    return acquire


def stored(sqlite_path, key):
    return np.asarray(StorageManager.load_serial_channel(key, filepath=sqlite_path))


def assert_stored_from_start(sqlite_path, acquisitions, session=None):
    """Every channel holds exactly the first `acquisitions` acquisitions, in order."""
    for channel, key in enumerate(KEYS):
        expected = np.arange(acquisitions * WINDOW) * len(KEYS) + channel
        np.testing.assert_array_equal(
            StorageManager.load_serial_channel(key, filepath=sqlite_path, session=session), expected)


def wait_until(condition, timeout=30):
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.02)


@pytest.fixture
def make_manager(tmp_path):
    managers = []

    def make(name="db", **options):
        options.setdefault("frame_shape", (100, WINDOW, len(KEYS)))
        manager = SensorManager(ser_channel_key=KEYS, commport=None,
                                sqlite_path=str(tmp_path / f"{name}.sqlite3"), **options)
        managers.append(manager)
        return manager

    yield make
    for manager in managers:
        manager.close()


def start_acquiring(manager):
    worker = manager.update_data_process(virtual_ser_port=True, func=counting_acquisition())
    manager.start_process(worker)
    return worker


@pytest.mark.processes
def test_flush_stores_everything_published_so_far(make_manager):
    manager = make_manager(start_stream_ingest=True)
    start_acquiring(manager)
    wait_until(lambda: manager.ring.write_idx >= 20)
    published = manager.ring.write_idx
    manager.flush()  # acquisition keeps running
    red = stored(manager.sqlite_path, "red")
    assert len(red) >= published * WINDOW
    np.testing.assert_array_equal(red, np.arange(len(red)) * len(KEYS))  # in order, no gaps


@pytest.mark.processes
def test_stop_stores_every_acquisition(make_manager):
    manager = make_manager(start_stream_ingest=True)
    start_acquiring(manager)
    wait_until(lambda: manager.ring.write_idx >= 20)
    manager.stop()
    assert_stored_from_start(manager.sqlite_path, manager.ring.write_idx)
    with pytest.raises(RuntimeError, match="create a new one"):
        manager.update_data_process(virtual_ser_port=True, func=counting_acquisition())


@pytest.mark.processes
def test_close_stops_the_workers_and_releases_shared_memory(make_manager):
    manager = make_manager(start_stream_ingest=True)
    workers = [start_acquiring(manager), manager._stream_proc, manager._ingest_proc]
    wait_until(lambda: manager.ring.write_idx >= 5)
    manager.close()
    assert manager.closed
    assert not any(worker.is_alive() for worker in workers)
    with pytest.raises(RuntimeError):  # no process can open the ring any more
        RingBuffer(manager.shm_name, manager.ring_capacity, manager.logical_shape, "line", np.float32)
    manager.close()  # closing again is harmless


@pytest.mark.processes
def test_two_sensor_managers_can_store_data_side_by_side(make_manager):
    first = make_manager("first", start_stream_ingest=True)
    second = make_manager("second", start_stream_ingest=True)
    assert first.shm_name != second.shm_name
    for manager in (first, second):
        start_acquiring(manager)
    wait_until(lambda: first.ring.write_idx >= 5 and second.ring.write_idx >= 5)
    for manager in (first, second):
        manager.stop()
        assert_stored_from_start(manager.sqlite_path, manager.ring.write_idx)


@pytest.mark.processes
def test_rerunning_with_the_same_database_closes_the_previous_manager(make_manager):
    first = make_manager(start_stream_ingest=True)
    with pytest.warns(RuntimeWarning, match="closing the previous SensorManager"):
        second = make_manager(start_stream_ingest=True)  # like re-running a notebook cell
    assert first.closed
    assert not second.closed


@pytest.mark.processes
def test_context_manager_closes_on_exit(tmp_path):
    sqlite_path = str(tmp_path / "db.sqlite3")
    with SensorManager(ser_channel_key=KEYS, commport=None, frame_shape=(100, WINDOW, len(KEYS)),
                       start_stream_ingest=True, sqlite_path=sqlite_path) as manager:
        start_acquiring(manager)
        wait_until(lambda: manager.ring.write_idx >= 5)
    assert manager.closed
    acquisitions = len(stored(sqlite_path, "red")) // WINDOW
    assert acquisitions >= 5
    assert_stored_from_start(sqlite_path, acquisitions)


@pytest.mark.processes
def test_without_storage_no_writer_or_ingester_runs(make_manager):
    manager = make_manager()
    assert manager._stream_proc is None and manager._ingest_proc is None
    with pytest.raises(RuntimeError, match="storage is off"):
        manager.flush()


@pytest.mark.processes
def test_the_old_stream_path_arguments_are_deprecated(make_manager):
    with pytest.warns(DeprecationWarning, match="no longer used"):
        make_manager(fast_stream_path_a="./a.bin", fast_stream_path_b="./b.bin")


@pytest.mark.processes
def test_stored_frames_carry_their_acquisition_times(make_manager):
    before = time.time_ns()
    manager = make_manager(start_stream_ingest=True)
    start_acquiring(manager)
    wait_until(lambda: manager.ring.write_idx >= 20)
    manager.stop()
    after = time.time_ns()
    session = manager.session["uuid"]
    unix = StorageManager.load_frame_times(manager.sqlite_path, session=session)
    monotonic = StorageManager.load_frame_times(manager.sqlite_path, session=session, clock="monotonic")
    assert len(unix) == manager.ring.write_idx
    assert np.all(np.diff(monotonic) > 0)
    assert before <= unix[0] and unix[-1] <= after
    # each acquisition sleeps 2 ms, so frames are at least that far apart; the writer's
    # batched polling would instead give many frames the same time
    assert np.median(np.diff(monotonic)) >= 1_900_000


@pytest.mark.processes
def test_each_run_is_stored_as_its_own_session(make_manager):
    first = make_manager(start_stream_ingest=True)
    start_acquiring(first)
    wait_until(lambda: first.ring.write_idx >= 5)
    first.close()
    second = make_manager(start_stream_ingest=True)
    start_acquiring(second)
    wait_until(lambda: second.ring.write_idx >= 5)
    second.stop()
    sessions = StorageManager.list_sessions(second.sqlite_path)
    assert [s["uuid"] for s in sessions] == [first.session["uuid"], second.session["uuid"]]
    assert sessions[-1]["channels"] == KEYS and sessions[-1]["frame_shape"] == (WINDOW, len(KEYS))
    assert_stored_from_start(second.sqlite_path, second.ring.write_idx, session=-1)


@pytest.mark.processes
def test_the_old_storage_arguments_of_update_data_process_are_deprecated(make_manager, tmp_path):
    manager = make_manager()
    with pytest.warns(DeprecationWarning, match="store nothing"):
        manager.update_data_process(save_data=True, filepath=str(tmp_path / "unused.sqlite3"),
                                    virtual_ser_port=True, func=counting_acquisition())
