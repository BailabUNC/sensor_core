import multiprocessing
from multiprocessing.shared_memory import SharedMemory
import plot_manager as pm
import numpy as np
from sqlitedict import SqliteDict


def create_event_flags():
    generate_ef = multiprocessing.Event()
    acquire_ef = multiprocessing.Event()
    return generate_ef, acquire_ef


def create_mutex():
    mutex = multiprocessing.Lock()
    return mutex


def acquire_mutex(mutex):
    mutex.acquire()


def release_mutex(mutex):
    mutex.release()


def create_shared_block(channel_key, grid_plot_flag=False, dtype=np.int64):
    if grid_plot_flag:
        plot, data = pm.initialize_grid_plot(num_channel=len(channel_key))
    else:
        plot, data = pm.initialize_plot()

    shm = SharedMemory(create=True, size=data.nbytes)
    data_shared = np.ndarray(shape=data.shape,
                             dtype=dtype, buffer=shm.buf)
    data_shared[:] = data[:]
    for i in range(len(channel_key)):
        _save_channel(key=channel_key[i], value=[0])

    return shm, data_shared, plot


def get_shm_data(shape, dtype, shm_name):
    shm = SharedMemory(shm_name)
    data_shared = np.ndarray(shape=shape, dtype=dtype,
                             buffer=shm.buf)
    return data_shared


def _save_channel(key, value, cache_file="cache.sqlite3"):
    try:
        with SqliteDict(cache_file) as mydict:
            mydict[key] = value  # Using dict[key] to store
            mydict.commit()  # Need to commit() to actually flush the data
    except Exception as ex:
        print("Error during storing data (Possibly unsupported):", ex)


def _load_channel(key, cache_file="cache.sqlite3"):
    try:
        with SqliteDict(cache_file) as mydict:
            value = mydict[key]  # No need to use commit(), since we are only loading data!
        return value
    except Exception as ex:
        print("Error during loading data:", ex)


def save_data(key, value, cache_file="cache.sqlite3"):
    try:
        data = _load_channel(key, cache_file)
        data.append(value)
        _save_channel(key, data, cache_file)
    except Exception as ex:
        print("Error during saving data", ex)
