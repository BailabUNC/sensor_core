from MABOS_core.data.data_manager import *
from MABOS_core.memory.mem_manager import *
from MABOS_core.serial.ser_manager import *
import multiprocessing
import threading
import os


class SensorManager:
    def __init__(self, channel_key: Union[np.ndarray, str], commport: str, num_points: int = 1000,
                 window_size: int = 1, baudrate: int = 115200):
        """ Initialize SensorManager Class
        Initializes serial port, shared memory object, and kwarg dictionary (args_dict)
        :param channel_key: list of channel names
        :param commport: target serial port
        :param num_points: number of 'time' points [num_points = time(s) * Hz]
        :param window_size: for 1D data, number of timepoints to acquire before passing
        :param baudrate: target baudrate
        """
        mutex = create_mutex()
        self.ser = setup_serial(commport, baudrate)

        self.shm, data_shared, self.plot = create_shared_block(grid_plot_flag=True,
                                                               channel_key=channel_key,
                                                               num_points=num_points)

        self.args_dict = {
            "channel_key": channel_key,
            "commport": commport,
            "baudrate": baudrate,
            "num_points": num_points,
            "window_size": window_size,
            "mutex": mutex,
            "ser": self.ser,
            "shm_name": self.shm.name,
            "plot": self.plot,
            "shape": data_shared.shape,
            "dtype": data_shared.dtype
        }

    def update_process(self, save_data: bool = True):
        """ Initialize dedicated process to update data

        :param save_data: boolean flag. If true, save acquired data to file and RAM, if not just update it in RAM
        :return: pointer to process
        """
        if save_data:
            p = multiprocessing.Process(name='update',
                                        target=update_save_data,
                                        args=(self.args_dict,))
        else:
            p = multiprocessing.Process(name='update',
                                        target=update_data,
                                        args=(self.args_dict,))
        return p
