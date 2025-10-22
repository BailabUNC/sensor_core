from sensor_core.data import DataManager
from sensor_core.plot import PlotManager
from sensor_core.memory.strg_manager import StorageManager
from sensor_core.memory.mem_utils import *
from sensor_core.utils.utils import *
from sensor_core.utils.utils import _coerce
from multiprocessing import Process, freeze_support, Manager
from threading import Thread
import pathlib



class SensorManager(DataManager, PlotManager, StorageManager):
    def __init__(self, ser_channel_key: Union[np.ndarray, str], commport: str,
                 baudrate: int = 115200, num_points: int = 1000, window_size: int = 1,
                 dtype=np.float32, **kwargs):
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

        # Setup serial and plot channels
        self.ser_channel_key, self.plot_channel_key = self.setup_channel_keys(
                                                      ser_channel_key=ser_channel_key,
                                                      **kwargs)
        # Setup ring buffer
        self.ring, frame_shape = initialize_ring(ser_channel_key=ser_channel_key,
                                            window_size=window_size,
                                            dtype=np.float32)
        # Setup target consumer params and enforce
        plot_target_fps = kwargs.get("plot_target_fps", 60.0)  # aim for ~60 Hz visualization
        plot_catchup_base_max = kwargs.get("plot_catchup_base_max", 2048)  # baseline cap per tick
        plot_catchup_boost = kwargs.get("plot_catchup_boost", 2.5)  # multiplier on dynamic cap

        plot_target_fps = _coerce(plot_target_fps, 60.0)
        plot_catchup_base_max = int(_coerce(plot_catchup_base_max, 2048))
        plot_catchup_boost = _coerce(plot_catchup_boost, 2.5)
        # Setup static args dict
        self.static_args_dict = create_static_dict(ser_channel_key=self.ser_channel_key,
                                                   plot_channel_key=self.plot_channel_key,
                                                   commport=commport,
                                                   baudrate=baudrate,
                                                   shm_name='/sensor_ring',
                                                   shape=frame_shape,
                                                   dtype=dtype,
                                                   ring_capacity=4096,
                                                   num_points=num_points,
                                                   plot_target_fps=plot_target_fps,
                                                   plot_catchup_base_max=plot_catchup_base_max,
                                                   plot_catchup_boost=plot_catchup_boost
                                                   )
        # Setup dynamic args dict + queue
        self.dynamic_args_dict = create_dynamic_dict(num_points=num_points,
                                                     window_size=window_size)

        self.dynamic_args_queue = self.setup_queue()

        # Make shared proxies for metrics
        self._mp_manager = Manager()
        self.writer_metrics_proxy = self._mp_manager.dict()
        self.plot_metrics_proxy = self._mp_manager.dict()
        # Initialize proxies
        self.writer_metrics_proxy.update({"init": True})
        self.plot_metrics_proxy.update({"init": True})

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

        plot_shape = np.shape(plot_channel_key)

        for key in np.reshape(plot_channel_key, newshape=(1, plot_shape[0] * plot_shape[1]))[0]:
            if key not in ser_channel_key:
                raise KeyError(f'plot_channel_key must include only keys within serial_channel_key')

        return ser_channel_key, plot_channel_key

    def update_data_process(self, save_data: bool = False, filepath: str = None,
                            virtual_ser_port: bool = False, func=None):
        """ Initialize dedicated process to update data

        :param save_data: boolean flag. If true, save acquired data to file and RAM, if not just update it in RAM
        :param filepath: string denoting target filepath to create database
        :param virtual_ser_port: boolean, if true data manager will not instantiate serial port. Will rely on
        user-defined custom function to generate simulated data
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
                          filepath=filepath,
                          virtual_ser_port=virtual_ser_port,
                          metrics_proxy=self.writer_metrics_proxy)

        if self.os_flag == 'win':
            p = Thread(name='update',
                       target=odm.online_update_data,
                       args=(func,))
        else:
            p = Process(name='update',
                        target=odm.online_update_data,
                        args=(func,))

        return p

    def setup_plotting_process(self):
        """ Initialize dedicated process to update plot
        :return: pointer to process and plot object
        """
        pm = PlotManager(static_args_dict=self.static_args_dict,
                         metrics_proxy=self.plot_metrics_proxy,)
        if self.os_flag == 'win':
            p = Thread(name='plot',
                       target=pm.online_plot_data)
        else:
            p = Process(name='plot',
                        target=pm.online_plot_data)

        return p, pm.fig


    def update_params(self, **kwargs):
        """ Check validity and update parameters

        :param kwargs: series of parameters to update
        :return: updates queue with new dictionary
        """
        self.dynamic_args_dict = update_dynamic_dict(dynamic_args_dict=self.dynamic_args_dict,
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

    def get_metrics(self) -> dict:
        """Return a combined metrics snapshot."""
        return {
            "writer": dict(self.writer_metrics_proxy),
            "plot": dict(self.plot_metrics_proxy),
        }
