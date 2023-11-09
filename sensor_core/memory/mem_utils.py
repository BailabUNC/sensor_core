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


def create_shared_block(channel_key: Union[np.ndarray, str], num_points: int = 1000,
                        grid_plot_flag: bool = True, dtype=np.int64):
    """ Create Shared Memory Block for global access to streamed data

    :param channel_key: list of channel names
    :param num_points: number of 'time' points [num_points = time(s) * Hz]
    :param grid_plot_flag: boolean, determines usage of Plot or GridPlot
    :param dtype: data type, default 64-bit integer
    :return: shm (shared memory object), data_shared (initial data), plot (Plot/GridPlot object)
    """
    if grid_plot_flag:
        if np.shape(channel_key)[1] > 1:
            xs, ys = initialize_grid_plot_data(num_channel=np.shape(channel_key)[1],
                                               num_points=num_points)
            data = np.vstack((xs, ys))
        else:
            raise ValueError(f"the length of channel key {channel_key} must be greater than one\n"
                             f"if grid_plot_flag {grid_plot_flag} is True")
    else:
        if np.shape(channel_key)[1] == 1:
            xs, ys = initialize_plot_data(num_points=num_points)
            data = np.dstack([xs, ys])[0]
        else:
            raise ValueError(f"the length of channel key {channel_key} must be equal to one\n"
                             f"if grid_plot_flag {grid_plot_flag} is False")

    shm = SharedMemory(create=True, size=data.nbytes)
    data_shared = np.ndarray(shape=data.shape,
                             dtype=dtype, buffer=shm.buf)
    data_shared[:] = data[:]

    del data, xs, ys

    return shm, data_shared
