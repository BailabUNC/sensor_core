import numpy as np
import MABOS_core.memory.mem_manager as mm
import MABOS_core.serial.ser_manager as sm


def initialize_plot_data():
    xs = [np.linspace(0, 999, 1000)]
    ys = np.ones(1000) * np.linspace(0, 1, 1000)
    return xs, ys


def initialize_grid_plot_data(num_channel):
    xs = [np.linspace(0, 999, 1000)]
    ys = np.ones((num_channel, 1000)) * np.linspace(0, 1, 1000)
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
        ys = sm.acquire_data(ser, num_channel=shape[0]-1)
        if ys is None or any(ys == 0):
            pass
        else:
            shm = mm.SharedMemory(shm_name)
            mm.acquire_mutex(mutex)
            data_shared = np.ndarray(shape=shape, dtype=dtype,
                                     buffer=shm.buf)
            xs = data_shared[0][-window_length:]
            data_shared[0][:-window_length] = data_shared[0][window_length:] - [window_length]
            data_shared[0][-window_length:] = xs
            if idx < 1000:
                for i in range(shape[0] - 1):
                    data_shared[i + 1][:-window_length] = data_shared[i + 1][window_length:]
                    data_shared[i + 1][-window_length:] = ys[i]
                    if ys[i] == 0:
                        print("false 0")
                idx += 1
            else:
                for i in range(shape[0] - 1):
                    data_shared[i + 1][:-window_length] = data_shared[i + 1][window_length:]
                    data_shared[i + 1][-window_length:] = ys[i]
                    mm.save_data(key=channel_key[i], value=data_shared[i+1][:])
                idx = 0

            mm.release_mutex(mutex)
