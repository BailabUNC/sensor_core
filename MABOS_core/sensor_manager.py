from MABOS_core.data import OnlineDataManager
from MABOS_core.plot import PlotManager
from MABOS_core.memory.mem_manager import *
from MABOS_core.utils.utils import *
from multiprocessing import Process, freeze_support
from threading import Thread
from warnings import warn


class SensorManager(OnlineDataManager, PlotManager):
    def __init__(self, channel_key: Union[np.ndarray, str], commport: str, num_points: int = 1000,
                 window_size: int = 1, baudrate: int = 115200, save_type: str = ".sqlite3"):
        """ Initialize SensorManager Class
        Initializes serial port, shared memory object, and kwarg dictionary (args_dict)
        :param channel_key: list of channel names
        :param commport: target serial port
        :param num_points: number of 'time' points [num_points = time(s) * Hz]
        :param window_size: for 1D data, number of time points to acquire before passing
        :param baudrate: target baudrate
        """

        self.save_type = save_type
        # Defines start method for multiprocessing. Necessary for windows and macOS
        self.os_flag = setup_process_start_method()

        mutex = create_mutex()
        self.window_size = window_size
        self.shm, data_shared = create_shared_block(grid_plot_flag=True,
                                                    channel_key=channel_key,
                                                    num_points=num_points)

        self.static_args_dict = {
            "channel_key": channel_key,
            "commport": commport,
            "baudrate": baudrate,
            "mutex": mutex,
            "shm_name": self.shm.name,
            "shape": data_shared.shape,
            "dtype": data_shared.dtype,
            "EOL": None,
            "num_points": num_points
        }

        self.dynamic_args_dict = {
            "num_points": num_points,
            "window_size": self.window_size
        }
        self.dynamic_args_queue = self.setup_queue(q_type="dynamic")

    def update_data_process(self, save_data: bool = True):
        """ Initialize dedicated process to update data

        :param save_data: boolean flag. If true, save acquired data to file and RAM, if not just update it in RAM
        :return: pointer to process
        """

        odm = OnlineDataManager(static_args_dict=self.static_args_dict,
                                dynamic_args_queue=self.dynamic_args_queue,
                                save_data=save_data,
                                multiproc=True,
                                save_type=self.save_type)

        if self.os_flag == 'win':
            p = Thread(name='update',
                       target=odm.online_update_data)
        else:
            p = Process(name='update',
                        target=odm.online_update_data)

        return p

    def setup_plot(self):
        PlotManager.__init__(self,
                             static_args_dict=self.static_args_dict,
                             online=True,
                             multiproc=True)
        self.initialize_plot()
        return self.plot

    def update_params(self, params: dict):
        """ Check validity and update parameters

        :param params: dictionary of parameters to update. By default, set to dynamic args dict.
        :return: updates queue with new dictionary
        """
        master_keys = self.dynamic_args_dict.keys()
        for param_key in params.keys():
            if param_key in master_keys:
                self.dynamic_args_dict[f"{param_key}"] = params[f"{param_key}"]
                self.update_queue(self.dynamic_args_queue)
            else:
                warn(f"Parameter key {param_key} does not exist in dynamic parameter dictionary\n"
                     f"{self.dynamic_args_dict}")

    def setup_queue(self, q_type: str = "dynamic"):
        """ Setup queue to hold dynamic parameter dictionaries

        :param q_type: string, determines if queue contains static or dynamic dictionary
        :return: queue object
        """
        q = multiprocessing.Queue()

        if q_type == "static":
            q.put(self.static_args_dict)
        elif q_type == "dynamic":
            q.put(self.dynamic_args_dict)
        else:
            warn(f"Unable to setup queue; type argument {q_type} must be 'static' or 'dynamic'")
        return q

    def update_queue(self, q):
        """ Adding item to existing queue object

        :param q: Queue object
        """
        q.put(self.dynamic_args_dict)

    def start_process(self, process):
        """ Function to start given process, and ensure safe operability with windows

        :param process: process object to start
        """
        if self.os_flag == 'win':
            freeze_support()
            process.start()
        else:
            process.start()
