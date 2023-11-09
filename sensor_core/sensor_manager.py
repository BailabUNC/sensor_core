from sensor_core.data import DataManager
from sensor_core.plot import PlotManager
from sensor_core.memory.mem_utils import *
from sensor_core.utils.utils import *
from multiprocessing import Process, freeze_support
from threading import Thread
from warnings import warn
import pathlib


class SensorManager(DataManager, PlotManager):
    def __init__(self, ser_channel_key: Union[np.ndarray, str], commport: str,
                 baudrate: int = 115200, num_points: int = 1000, window_size: int = 1,
                 dtype=np.float64, **kwargs):
        """ Initialize SensorManager Class
        Initializes serial port, shared memory object, and kwarg dictionary (args_dict)
        :param serial_channel_key: list of serial channel names
        :param commport: target serial port
        :param baudrate: target data transfer rate (in bits/sec)
        :param num_points: number of 'time' points [num_points = time(s) * Hz]
        :param window_size: for 1D data, number of time points to acquire before passing
        :param dtype: data type to store in shared memory object
        """
        # Defines start method for multiprocessing. Necessary for windows and macOS
        self.os_flag = setup_process_start_method()
        # Create mutex to manage access to shared memory object
        mutex = create_mutex()
        # Setup serial and plot channels
        self.ser_channel_key, self.plot_channel_key = self.setup_channel_keys(
                                                         ser_channel_key=ser_channel_key,
                                                         **kwargs)

        self.shm, data_shared = create_shared_block(ser_channel_key=ser_channel_key,
                                                    num_points=num_points,
                                                    dtype=dtype)

        self.static_args_dict = self.create_static_dict(ser_channel_key=self.ser_channel_key,
                                                        plot_channel_key=self.plot_channel_key,
                                                        commport=commport,
                                                        baudrate=baudrate,
                                                        mutex=mutex,
                                                        shm_name=self.shm.name,
                                                        shape=data_shared.shape,
                                                        dtype=dtype,
                                                        num_points=num_points)

        self.dynamic_args_dict = self.create_dynamic_dict(num_points=num_points,
                                                          window_size=window_size)

        self.dynamic_args_queue = self.setup_queue()

    @staticmethod
    def setup_channel_keys(ser_channel_key, **kwargs):
        """ Set up serial and plot channel keys
        :param ser_channel_key: serial channel key (list of names for serial channels)
        :param kwargs: can contain plot_channel_key kwarg, which we can use to set plot settings
        :return: ser_channel_key and plot_channel_key, lists of keys for serial and plot functions
        """
        if len(np.shape(ser_channel_key)) > 1:
            raise ValueError(f"serial channel key {ser_channel_key} \n"
                             f"must be a one-dimensional list")
        if "plot_channel_key" in kwargs:
            plot_channel_key = kwargs["plot_channel_key"]
        else:
            plot_channel_key = [ser_channel_key]

        for key in plot_channel_key:
            if key not in ser_channel_key:
                raise KeyError(f'plot_channel_key must include only keys within serial_channel_key')

        return ser_channel_key, plot_channel_key

    def update_data_process(self, save_data: bool = False, filepath: str = None, func=None):
        """ Initialize dedicated process to update data

        :param save_data: boolean flag. If true, save acquired data to file and RAM, if not just update it in RAM
        :param filepath: string denoting target filepath to create database
        :param func: optional custom function to handle serial data acquisition
        :return: pointer to process
        """
        if save_data and (filepath is not None):
            filetype = pathlib.Path(filepath).suffix
            if filetype != ".hdf5" and filetype != ".sqlite3":
                raise ValueError(f"filepath {filepath} is an invalid filetype {filetype}\n"
                                 f"filepaths should create .hdf5 or .sqlite3 files only")

        odm = DataManager(static_args_dict=self.static_args_dict,
                          dynamic_args_queue=self.dynamic_args_queue,
                          save_data=save_data,
                          filepath=filepath)

        if self.os_flag == 'win':
            p = Thread(name='update',
                       target=odm.online_update_data,
                       args=(func,))
        else:
            p = Process(name='update',
                        target=odm.online_update_data,
                        args=(func,))

        return p

    def setup_plot(self):
        PlotManager.__init__(self,
                             static_args_dict=self.static_args_dict)
        self.initialize_plot()
        return self.plot

    def update_params(self, **kwargs):
        """ Check validity and update parameters

        :param kwargs: series of parameters to update
        :return: updates queue with new dictionary
        """
        self.dynamic_args_dict = self.update_dynamic_dict(dynamic_args_dict=self.dynamic_args_dict,
                                                          **kwargs)
        self.update_queue(self.dynamic_args_queue)

    def setup_queue(self):
        """ Setup queue to hold dynamic parameter dictionaries
        :return: queue object
        """
        q = multiprocessing.Queue()
        q.put(self.dynamic_args_dict)
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
