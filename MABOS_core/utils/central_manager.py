from MABOS_core.data.data_manager import *
from MABOS_core.memory.mem_manager import *
from MABOS_core.serial.ser_manager import *
import multiprocessing
import threading
import os


class CentralManager:
    def __init__(self, channel_key, commport, window_length=1, baudrate=115200):
        mutex = create_mutex()
        ser = setup_serial(commport, baudrate)
        self.shm, data_shared, plot = create_shared_block(grid_plot_flag=True, channel_key=channel_key)

        self.args_dict = {
            "channel_key": channel_key,
            "commport": commport,
            "baudrate": baudrate,
            "window_length": window_length,
            "mutex": mutex,
            "ser": ser,
            "shm_name": self.shm.name,
            "plot": plot,
            "shape": data_shared.shape,
            "dtype": data_shared.dtype
        }

    def update_process(self):
        if os.name == 'nt':
            p = threading.Thread(name='update',
                                 target=update_data,
                                 args=(self.args_dict,))
        else:
            p = multiprocessing.Process(name='update',
                                        target=update_data,
                                        args=(self.args_dict,))
        return p
