import h5py
from sqlitedict import SqliteDict
import numpy as np
import pathlib
from typing import *


def create_h5_file(filepath: str):
    f = h5py.File(filepath, 'w')
    return f


def create_sqlite3_file(filepath: str, key: str):
    with SqliteDict(filepath) as mydict:
        mydict[key] = []
        mydict.commit()


def load_sqlite3_file(filepath: str):
    with SqliteDict(filepath) as mydict:
        return mydict


def load_h5_file(filepath: str):
    f = h5py.File(filepath, 'a')
    return f


class StorageManager:
    def __init__(self, channel_key: Union[np.ndarray, str], filepath: str = './serial_db',
                 overwrite: bool = False, filetype: str = '.hdf5'):
        self.channel_key = channel_key
        self.full_filepath = filepath + filetype
        self.filetype = filetype
        self.overwrite = overwrite

        if not self.filetype == ".hdf5" and not self.filetype == ".sqlite3":
            raise ValueError(f"defined filetype {filetype} is unsupported. \n"
                             f"Must use .hdf5 or .sqlite3")

        if pathlib.Path.is_file(pathlib.Path(self.full_filepath)) and not self.overwrite:
            raise ValueError(f'proposed filepath {self.full_filepath} already exists. Set overwrite '
                             f'kwarg to true if you wish to replace file')

    def create_serial_database(self):
        if self.filetype == ".hdf5":
            f = create_h5_file(filepath=self.full_filepath)
            for i, key in enumerate(self.channel_key):
                f.create_dataset(name=key,
                                 shape=(1, 1),
                                 chunks=True,
                                 maxshape=(1, None))
            f.close()
        elif self.filetype == ".sqlite3":
            for i, key in enumerate(self.channel_key):
                create_sqlite3_file(filepath=self.full_filepath,
                                    key=key, )

    def load_serial_database(self):
        if self.filetype == ".hdf5":
            db = load_h5_file(filepath=self.full_filepath)
        elif self.filetype == ".sqlite3":
            db = load_sqlite3_file(filepath=self.full_filepath)
        return db

    def load_serial_channel(self, key: str, keep_db_open: bool = False):
        db = self.load_serial_database()
        if self.filetype == ".hdf5":
            try:
                channel = db[key]
            except:
                raise ValueError(f'Given key {key} is not in hdf5 file at {self.full_filepath}')

            channel_data = np.zeros((1, channel.shape[1]), dtype=channel.dtype)
            channel.read_direct(channel_data, np.s_[0, :], np.s_[0, :])

            if not keep_db_open:
                db.close()

        elif self.filetype == ".sqlite3":
            try:
                with db:
                    channel = db[key]
            except:
                raise ValueError(f'Given key {key} is not in sqlite3 file at {self.full_filepath}')
        return db, channel

    def append_serial_channel(self, key: str, data: np.ndarray):
        db, channel = self.load_serial_channel(key=key, keep_db_open=True)
        if self.filetype == ".hdf5":
            channel.resize((channel.shape[1] + data.shape[0]), axis=1)
            channel.write_direct(source=data, dest_sel=np.s_[0, -data.shape[0]:])
            db.close()
        elif self.filetype == ".sqlite3":
            channel = np.append(channel[:], data)
            with db:
                db[key] = channel
                db.commit()
