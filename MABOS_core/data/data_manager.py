import numpy as np
import MABOS_core.memory as mm
import MABOS_core.serial.ser_manager as sm


def update_save_data(args_dict: dict, queue):
    """ Update data in shared memory object, and intermittently save data to .sqlite3 file

    :param args_dict: dictionary containing kwargs for memory and plot management. Contains the following:
        channel_key, commport, baudrate, num_points, window_size, mutex, ser, shm.name, plot, shape, dtype
    :param queue: Queue object used to pass dynamic input parameters like {num_points, window_size}
    :return: no explicit return. Continuously runs to update memory object with streamed data and save data to file
    """
    idx = 0
    try:
        ser = args_dict["ser"]
        shm_name = args_dict["shm_name"]
        mutex = args_dict["mutex"]
        shape = args_dict["shape"]
        dtype = args_dict["dtype"]
        channel_key = args_dict["channel_key"]
    except:
        raise ValueError(f"args_dict {args_dict} should contain the following keys:\n"
                         f"ser, shm_name, mutex, shape, dtype, channel_key, num_points")

    try:
        dynamic_args_dict = queue.get()
    except:
        raise ValueError(f"Queue {queue} is empty upon initialization, please restart process")

    while True:
        if queue.empty():
            pass
        else:
            dynamic_args_dict = queue.get()
        window_size = dynamic_args_dict["window_size"]
        num_points = dynamic_args_dict["num_points"]
        ys = sm.acquire_data(ser=ser, num_channel=shape[0] - 1, window_size=window_size)
        if ys is not None:
            window_length = np.shape(ys)[0]
            shm = mm.SharedMemory(shm_name)
            mm.acquire_mutex(mutex)
            data_shared = np.ndarray(shape=shape, dtype=dtype,
                                     buffer=shm.buf)

            if idx < num_points:
                for i in range(shape[0] - 1):
                    data_shared[i + 1][:-window_length] = data_shared[i + 1][window_length:]
                    data_shared[i + 1][-window_length:] = ys[:, i]
                idx += 1
            else:
                for i in range(shape[0] - 1):
                    data_shared[i + 1][:-window_length] = data_shared[i + 1][window_length:]
                    data_shared[i + 1][-window_length:] = ys[:, i]
                    save_data = data_shared[i + 1][:]
                    mm.append_channel(key=channel_key[i], value=save_data)
                idx = 0
            mm.release_mutex(mutex)


def update_data(args_dict: dict, queue):
    """ Update data in shared memory object

    :param args_dict: dictionary containing kwargs for memory and plot management. Contains the following:
        channel_key, commport, baudrate, mutex, ser, shm.name, plot, shape, dtype
    :param queue: Queue object used to pass dynamic input parameters like {num_points, window_size}
    :return: no explicit return. Continuously runs to update memory object with streamed data
    """

    try:
        ser = args_dict["ser"]
        shm_name = args_dict["shm_name"]
        mutex = args_dict["mutex"]
        shape = args_dict["shape"]
        dtype = args_dict["dtype"]
    except:
        raise ValueError(f"args_dict {args_dict} should contain the following keys:\n"
                         f"ser, shm_name, mutex, shape, dtype")

    try:
        dynamic_args_dict = queue.get()
    except:
        raise ValueError(f"Queue {queue} is empty upon initialization, please restart process")

    while True:
        if queue.empty():
            pass
        else:
            dynamic_args_dict = queue.get()
        window_size = dynamic_args_dict["window_size"]
        ys = sm.acquire_data(ser=ser, num_channel=shape[0] - 1, window_size=window_size)
        if ys is not None:
            window_length = np.shape(ys)[0]
            shm = mm.SharedMemory(shm_name)
            mm.acquire_mutex(mutex)
            data_shared = np.ndarray(shape=shape, dtype=dtype,
                                     buffer=shm.buf)

            for i in range(shape[0] - 1):
                data_shared[i + 1][:-window_length] = data_shared[i + 1][window_length:]
                data_shared[i + 1][-window_length:] = ys[:, i]

            mm.release_mutex(mutex)
