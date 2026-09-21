"""SQLite storage: creating databases, appending, and loading channels."""
import numpy as np
import pytest

from sensor_core.memory.strg_manager import StorageManager

KEYS = ["red", "infrared", "violet"]


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "db.sqlite3")


def test_new_databases_contain_every_channel_and_a_time_key(db_path):
    StorageManager(KEYS, db_path).create_serial_database()
    for key in KEYS + ["time"]:
        assert StorageManager.load_serial_channel(key, filepath=db_path).size == 0


def test_appended_samples_are_stored_in_order(db_path):
    storage = StorageManager(KEYS, db_path)
    storage.create_serial_database()
    storage.append_serial_channel("red", np.array([1, 2, 3], np.float32))
    storage.append_serial_channel("red", np.array([4, 5], np.float32))
    np.testing.assert_array_equal(StorageManager.load_serial_channel("red", filepath=db_path), [1, 2, 3, 4, 5])


def test_appending_creates_missing_keys_and_keeps_their_dtype(db_path):
    StorageManager(KEYS, db_path).append_serial_channel("counts", np.array([7, 8], np.int16))
    stored = StorageManager.load_serial_channel("counts", filepath=db_path)
    np.testing.assert_array_equal(stored, [7, 8])
    assert stored.dtype == np.int16


def test_loading_an_unknown_key_raises(db_path):
    StorageManager(KEYS, db_path).create_serial_database()
    with pytest.raises(ValueError, match="not in sqlite3 file"):
        StorageManager.load_serial_channel("green", filepath=db_path)


def test_only_sqlite_files_are_supported(tmp_path):
    with pytest.raises(ValueError, match="unsupported"):
        StorageManager(KEYS, str(tmp_path / "db.h5"))
