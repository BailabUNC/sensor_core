import numpy as np
import MABOS_core.memory as mm
import MABOS_core.serial.ser_manager as sm


def update_data(args_dict):
    idx = 0
    ser = args_dict["ser"]
    shm_name = args_dict["shm_name"]
    mutex = args_dict["mutex"]
    shape = args_dict["shape"]
    dtype = args_dict["dtype"]
    channel_key = args_dict["channel_key"]
    num_points = args_dict["num_points"]
    while True:
        ys = sm.acquire_data(ser, num_channel=shape[0]-1)
        if ys is not None:
            window_length = np.shape(ys)[0]
            shm = mm.SharedMemory(shm_name)
            mm.acquire_mutex(mutex)
            data_shared = np.ndarray(shape=shape, dtype=dtype,
                                     buffer=shm.buf)
            xs = data_shared[0][-window_length:]
            data_shared[0][:-window_length] = data_shared[0][window_length:] - [window_length]
            data_shared[0][-window_length:] = xs

            if idx < num_points:
                for i in range(shape[0] - 1):
                    data_shared[i + 1][:-window_length] = data_shared[i + 1][window_length:]
                    data_shared[i + 1][-window_length:] = ys[:, i]
                idx += 1
            else:
                for i in range(shape[0] - 1):
                    data_shared[i + 1][:-window_length] = data_shared[i + 1][window_length:]
                    data_shared[i + 1][-window_length:] = ys[:, i]
                    mm.save_channels(key=channel_key[i], value=data_shared[i+1][:])
                idx = 0

            mm.release_mutex(mutex)
