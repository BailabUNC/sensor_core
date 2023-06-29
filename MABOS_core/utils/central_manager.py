import data_manager as dm
import mem_manager as mm
import ser_manager as sm
import plot_manager as pm
from functools import partial
import multiprocessing


class CentralManger:
    def __init__(self, channel_key, commport, window_length=1, baudrate=115200):
        mutex = mm.create_mutex()
        ser = sm.setup_serial(commport, baudrate)
        self.shm, data_shared, plot = mm.create_shared_block(grid_plot_flag=True, channel_key=channel_key)

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
        p1 = multiprocessing.Process(name='update',
                                     target=dm.update_data,
                                     args=(self.args_dict,))
        return p1
