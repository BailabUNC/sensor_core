import h5py
from sqlitedict import SqliteDict
import numpy as np
import pathlib
from typing import *


def create_h5_file(filepath: str):
    """ Create hdf5 file
    :param filepath: string defining target filepath for storage file.
    """
    f = h5py.File(filepath, 'w')
    return f


def create_sqlite3_file(filepath: str, key: str):
    """ Create sqlite3 file
    :param filepath: string defining target filepath for storage file.
    :param key: string defining key to add to the file
    """
    with SqliteDict(filepath) as mydict:
        mydict[key] = []
        mydict.commit()


def load_h5_file(filepath: str):
    """ Load hdf5 file
     :param filepath: string defining target filepath for storage file.
    """
    f = h5py.File(filepath, 'a')
    return f


def load_sqlite3_file(filepath: str):
    """ Load sqlite3 file
     :param filepath: string defining target filepath for storage file.
    """
    with SqliteDict(filepath) as mydict:
        return mydict


class StorageManager:
    def __init__(self, channel_key: Union[np.ndarray, str],
                 filepath: str = './serial_db.sqlite3', overwrite: bool = False):
        """ Initialize Storage Manager class
        :param channel_key: array containing channel key names
        :param filepath: string defining target filepath for storage file. filetype extracted from suffix
        :param overwrite: boolean, determines whether target file should be overwritten if it already exists
        """
        self.channel_key = channel_key
        self.filepath = filepath
        self.filetype = pathlib.Path(self.filepath).suffix
        self.overwrite = overwrite

        if self.filetype != ".hdf5" and self.filetype != ".sqlite3":
            raise ValueError(f"defined filetype {self.filetype} is unsupported. \n"
                             f"Must use .hdf5 or .sqlite3")

        if pathlib.Path.is_file(pathlib.Path(self.filepath)) and not self.overwrite:
            raise ValueError(f'proposed filepath {self.filepath} already exists. Set overwrite '
                             f'kwarg to true if you wish to replace file')

    def create_serial_database(self):
        """ Create database to store serial data
        """
        if self.filetype == ".hdf5":
            f = create_h5_file(filepath=self.filepath)
            for i, key in enumerate(self.channel_key):
                f.create_dataset(name=key,
                                 shape=(1, 1),
                                 chunks=True,
                                 maxshape=(1, None))
            f.close()
        elif self.filetype == ".sqlite3":
            for i, key in enumerate(self.channel_key):
                create_sqlite3_file(filepath=self.filepath,
                                    key=key)

    @staticmethod
    def load_serial_database(filepath: str = './serial_db.sqlite3', filetype: str = None):
        """ Static method to load serial database
        :param filepath: string defining target storage file to open
        :param filetype: string defining target filetype. If None, extracted from filepath
        return: target database
        """
        if filetype is None:
            filetype = pathlib.Path(filepath).suffix

        if filetype == ".hdf5":
            db = load_h5_file(filepath=filepath)
        elif filetype == ".sqlite3":
            db = load_sqlite3_file(filepath=filepath)
        else:
            raise ValueError(f"filetype {filetype} must be .hdf5 or .sqlite3")
        return db

    @classmethod
    def load_serial_channel(cls, key: str, filepath: str = './serial_db.sqlite3',
                            filetype: str = None, keep_db_open: bool = False):
        """ Class method to load serial channel from serial database
        :param key: string defining target key to unpack
        :param filepath: string defining filepath to target database
        :param filetype: string defining target filetype. If None, extracted from filepath
        :param keep_db_open: boolean, determines whether to close the database or keep it open (hdf5)
        return: db (database) and channel (channel_data for target key)
        """
        db = cls.load_serial_database(filepath=filepath, filetype=filetype)

        if filetype is None:
            filetype = pathlib.Path(filepath).suffix

        if filetype == ".hdf5":
            try:
                channel = db[key]
            except:
                raise ValueError(f'Given key {key} is not in hdf5 file at {filepath}')

            channel_data = np.zeros((1, channel.shape[1]), dtype=channel.dtype)
            channel.read_direct(channel_data, np.s_[0, :], np.s_[0, :])

            if not keep_db_open:
                db.close()

        elif filetype == ".sqlite3":
            try:
                with db:
                    channel = db[key]
            except:
                raise ValueError(f'Given key {key} is not in sqlite3 file at {filepath}')
        return db, channel

    def append_serial_channel(self, key: str, data: np.ndarray):
        """ function to append input data to existing channel in serial database
        :param key: string defining target key to append data to
        :param data: input data to append to existing channel key
        """
        db, channel = self.load_serial_channel(filepath=self.filepath, filetype=self.filetype,
                                               key=key, keep_db_open=True)
        if self.filetype == ".hdf5":
            channel.resize((channel.shape[1] + data.shape[0]), axis=1)
            channel.write_direct(source=data, dest_sel=np.s_[0, -data.shape[0]:])
            db.close()
        elif self.filetype == ".sqlite3":
            channel = np.append(channel[:], data)
            with db:
                db[key] = channel
                db.commit()
