from sqlitedict import SqliteDict
import numpy as np
import pathlib
from typing import *


def create_sqlite3_file(filepath: str, key: str, dtype=np.float32):
    """Create (or ensure) a sqlite3 key."""
    with SqliteDict(filepath) as db:
        if key not in db:
            db[key] = np.array([], dtype=dtype)
        db.commit()

def load_sqlite3_file(filepath: str):
    """Return a live SqliteDict handle (caller manages context)."""
    return SqliteDict(filepath)


class StorageManager:
    def __init__(self, channel_key: Union[np.ndarray, list, tuple, str],
                 filepath: str = './serial_db.sqlite3', overwrite: bool = False):
        """
        :param channel_key: list/array of channel key names (or single key)
        :param filepath: storage file path (.hdf5 or .sqlite3)
        :param overwrite: if True, allow recreating files (hdf5 only here)
        """
        self.channel_key = channel_key if isinstance(channel_key, (list, tuple, np.ndarray)) else [channel_key]
        self.filepath = filepath
        self.filetype = pathlib.Path(self.filepath).suffix
        self.overwrite = overwrite
        if self.filetype not in ".sqlite3":
            raise ValueError(f"defined filetype {self.filetype} is unsupported. Must use .sqlite3")

    def create_serial_database(self, dtype_map: Dict[str, np.dtype] = None):
        """Create a database with the provided channel keys."""
        dtype_map = dtype_map or {}
        with SqliteDict(self.filepath) as db:
            for key in self.channel_key:
                if key not in db:
                    dt = dtype_map.get(key, np.float32)
                    db[key] = np.array([], dtype=dt)
            if 'time' not in db:
                db['time'] = np.array([], dtype=np.float64)
            db.commit()

    @staticmethod
    def load_serial_database(filepath: str = './serial_db.sqlite3', filetype: str = None):
        """Open a DB handle (caller is responsible for closing it if needed)."""
        if filetype is None:
            filetype = pathlib.Path(filepath).suffix

        if filetype == ".sqlite3":
            return load_sqlite3_file(filepath=filepath)
        else:
            raise ValueError(f"filetype {filetype} must be .sqlite3")

    @classmethod
    def load_serial_channel(cls, key: str, filepath: str = './serial_db.sqlite3',
                            filetype: str = None, return_db: bool = False):
        """
        Load a channel w/in sqlite database
        """
        db = cls.load_serial_database(filepath=filepath, filetype=filetype)
        if filetype is None:
            filetype = pathlib.Path(filepath).suffix

        if filetype == ".sqlite3":
            try:
                with db:
                    channel = db[key]
            except Exception:
                raise ValueError(f'Given key {key} is not in sqlite3 file at {filepath}')
            return (db, channel) if return_db else channel
        else:
            raise ValueError(f"filetype {filetype} must be .sqlite3")

    def append_serial_channel(self, key: str, data: np.ndarray):
        """
        Append data to a channel. For SQLite, this **creates the key if missing**.
        """
        if self.filetype == ".sqlite3":
            db = self.load_serial_database(filepath=self.filepath, filetype=self.filetype)
            with db:
                if key not in db:
                    db[key] = np.array([], dtype=data.dtype)
                existing = np.asarray(db[key], dtype=data.dtype)
                db[key] = np.append(existing, data)
                db.commit()
            return
        else:
            raise ValueError(f"Unsupported filetype {self.filetype}")
