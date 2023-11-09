import multiprocessing
from typing import *
from multiprocessing.shared_memory import SharedMemory
from sensor_core.plot.plot_utils import *
import numpy as np


def create_mutex():
    """ Create Mutual Exclusion Lock

    :return: mutex object
    """
    mutex = multiprocessing.Lock()
    return mutex


def acquire_mutex(mutex):
    """ Acquire Mutual Exclusion Lock

    :param mutex: mutex object
    """
    mutex.acquire()


def release_mutex(mutex):
    """ Release Mutual Exclusion Lock

    :param mutex: mutex object
    """
    mutex.release()


def create_shared_block(ser_channel_key: Union[np.ndarray, str],
                        num_points: int = 1000, dtype=np.float64):
    """ Create Shared Memory Block for global access to streamed data

    :param ser_channel_key: list of channel names
    :param num_points: number of 'time' points [num_points = time(s) * Hz]
    :param dtype: data type, default 64-bit integer
    :return: shm (shared memory object), data_shared (initial data), plot (Plot/GridPlot object)
    """
    if np.shape(ser_channel_key)[1] > 1:
        xs, ys = initialize_grid_plot_data(num_channel=len(ser_channel_key),
                                           num_points=num_points)
        data = np.vstack((xs, ys))
    else:
        xs, ys = initialize_plot_data(num_points=num_points)
        data = np.dstack([xs, ys])[0]

    shm = SharedMemory(create=True, size=data.nbytes)
    data_shared = np.ndarray(shape=data.shape,
                             dtype=dtype, buffer=shm.buf)
    data_shared[:] = data[:]

    del data, xs, ys

    return shm
