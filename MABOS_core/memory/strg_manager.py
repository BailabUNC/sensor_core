from sqlitedict import SqliteDict
import numpy as np


def _save_channel(key: str, value: np.ndarray, cache_file: str = "cache.sqlite3"):
    """ Save array to SQL Dictionary w/ key

    :param key: dictionary key to save array to
    :param value: array being saved
    :param cache_file: path to .sqlite3 file
    """
    try:
        with SqliteDict(cache_file) as mydict:
            mydict[key] = value  # Using dict[key] to store
            mydict.commit()  # Need to commit() to actually flush the data
    except Exception as ex:
        print("Error during storing data (Possibly unsupported):", ex)


def _load_channel(key: str, cache_file: str = "cache.sqlite3"):
    """ Load array under a specific key from SQL Dictionary

    :param key: dictionary key to load array from
    :param cache_file: path to .sqlite3 file
    :return: value, array of values stored at dictionary key
    """
    try:
        with SqliteDict(cache_file) as mydict:
            value = mydict[key]
        return value
    except Exception as ex:
        print("Error during loading data:", ex)


def append_channel(key: str, value: np.ndarray, cache_file: str = "cache.sqlite3"):
    """ Append new data to existing dictionary key

    :param key: dictionary key to load and append to existing array
    :param value: new array being saved
    :param cache_file: path to .sqlite3 file
    """
    try:
        data = _load_channel(key, cache_file)
        data.append(value)
        _save_channel(key, data, cache_file)
    except Exception as ex:
        print("Error during saving data", ex)

