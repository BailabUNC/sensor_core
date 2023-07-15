import multiprocessing
from typing import *
from multiprocessing.shared_memory import SharedMemory
import MABOS_core.plot.plot_manager as pm
from .strg_manager import _save_channel
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


def create_shared_block(channel_key: Union[np.ndarray, str], num_points: int = 1000, save_data: bool = True,
                        grid_plot_flag: bool = True, dtype=np.int64):
    """ Create Shared Memory Block for global access to streamed data

    :param channel_key: list of channel names
    :param num_points: number of 'time' points [num_points = time(s) * Hz]
    :param save_data: boolean, determines whether streamed data is saved to file
    :param grid_plot_flag: boolean, determines usage of Plot or GridPlot
    :param dtype: data type, default 64-bit integer
    :return: shm (shared memory object), data_shared (initial data), plot (Plot/GridPlot object)
    """

    plot, data = pm.initialize_plot(channel_key=channel_key, num_points=num_points, grid_plot_flag=grid_plot_flag)

    shm = SharedMemory(create=True, size=data.nbytes)
    data_shared = np.ndarray(shape=data.shape,
                             dtype=dtype, buffer=shm.buf)
    data_shared[:] = data[:]
    if save_data:
        for i in range(len(channel_key)):
            _save_channel(key=channel_key[i], value=[0])

    return shm, data_shared, plot
