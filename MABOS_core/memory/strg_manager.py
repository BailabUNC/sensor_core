import h5py
import numpy as np
import pathlib
from typing import *


def _create_h5_file(filepath: str, overwrite: bool = False):
    full_path = pathlib.Path(filepath)
    if overwrite:
        f = h5py.File(full_path, 'w')
    else:
        if pathlib.Path.is_file(full_path):
            raise ValueError(f'proposed filepath {full_path} already exists. Set overwrite '
                             f'kwarg to true if you wish to replace file')
        else:
            f = h5py.File(full_path, 'x')
    return f


def create_serial_database(channel_key: Union[np.ndarray, str], filepath: str = './serial_db.hdf5',
                           num_points: int = 1000, overwrite: bool = False):
    f = _create_h5_file(filepath, overwrite)
    for i in range(len(channel_key)):
        f.create_dataset(channel_key[i], shape=(1, num_points), chunks=True, maxshape=(1, None))
    f.close()


def load_h5_file(filepath: str):
    full_path = pathlib.Path(filepath)
    if pathlib.Path.is_file(full_path):
        f = h5py.File(filepath, 'a')
    else:
        raise ValueError(f"given filepath {filepath} does not exist, or is not a valid file")
    return f


def load_dataset(filepath: str, key: str):
    f = load_h5_file(filepath)
    dset = f[key]
    if dset:
        return f, dset
    else:
        raise ValueError(f'Given key {key} is not in hdf5 file at {filepath}')


def load_serial_channel(key: str, filepath: str = './serial_db.hdf5'):
    f, dset = load_dataset(filepath, key)
    channel_data = np.zeros((1, dset.shape[1]), dtype=dset.dtype)
    dset.read_direct(channel_data, np.s_[0, :], np.s_[0, :])
    f.close()
    return channel_data[0]


def append_serial_channel(key: str, data: np.ndarray or dict, filepath: str = './serial_db.hdf5',
                          num_points: int = 1000):
    f, dset = load_dataset(filepath, key)
    dset.resize((dset.shape[1] + data.shape[0]), axis=1)
    dset.write_direct(source=data, dest_sel=np.s_[0, -num_points:])
    f.close()

