import numpy as np
from multiprocessing.shared_memory import SharedMemory
from .mem_manager import *
from .ser_manager import *


def initialize_plot_data():
    xs = [np.linspace(0, 999, 1000)]
    ys = np.ones(1000) * np.linspace(0, 1, 1000)
    return xs, ys


def initialize_grid_plot_data(num_channel):
    xs = [np.linspace(0, 999, 1000)]
    ys = np.ones((num_channel, 1000)) * np.linspace(0, 1, 1000)
    # ys = np.array(np.random.randint(0, 1000, size=(3, 1000)))
    return xs, ys


def update_data(args_dict):
    idx = 0
    ser = args_dict["ser"]
    shm_name = args_dict["shm_name"]
    mutex = args_dict["mutex"]
    window_length = args_dict["window_length"]
    shape = args_dict["shape"]
    dtype = args_dict["dtype"]
    channel_key = args_dict["channel_key"]
    while True:
        ys = acquire_data(ser, num_channel=shape[0]-1)
        if ys is None:
            pass
        else:
            shm = SharedMemory(shm_name)
            acquire_mutex(mutex)
            data_shared = np.ndarray(shape=shape, dtype=dtype,
                                     buffer=shm.buf)
            xs = data_shared[0][-window_length:]
            data_shared[0][:-window_length] = data_shared[0][window_length:] - [window_length]
            data_shared[0][-window_length:] = xs
            if idx < 1000:
                for i in range(shape[0] - 1):
                    data_shared[i + 1][:-window_length] = data_shared[i + 1][window_length:]
                    data_shared[i + 1][-window_length:] = ys[i]
                idx += 1
            else:
                for i in range(shape[0] - 1):
                    data_shared[i + 1][:-window_length] = data_shared[i + 1][window_length:]
                    data_shared[i + 1][-window_length:] = ys[i]
                    save_data(key=channel_key[i], value=data_shared[i+1])
                idx = 0

            release_mutex(mutex)
