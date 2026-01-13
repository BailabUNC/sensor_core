import time
import os
from sensor_core.data import DataManager
from sensor_core.memory.strg_manager import StorageManager
from sensor_core.dsp.dsp_manager import DSPManager
from sensor_core.memory.mem_utils import *
from sensor_core.utils.utils import *
from sensor_core.utils.utils import _coerce
from multiprocessing import Process, freeze_support, Manager
from threading import Thread
import pathlib


class SensorManager(DataManager, PlotManager, StorageManager):
    def __init__(self, 
                 ser_channel_key: Union[np.ndarray, str],
                 commport: str,
                 baudrate: int = 115200,
                 dtype=np.float32,
                 data_mode: str = "line",
                 frame_shape: tuple = (1000, 100, 3),
                 fast_stream_path_a: str="./serial_stream_a.bin",
                 fast_stream_path_b: str = "./serial_stream_b.bin",
                 start_stream_ingest: bool = False,
                 sqlite_path: str = "./serial_db.sqlite3",
                 rotate_frames: int = 8192,
                 rotate_seconds: float = 5.0,
                 **kwargs):
        """ Initialize SensorManager Class
        Initializes serial port, shared memory object, and kwarg dictionary (args_dict)
        :param serial_channel_key: list of serial channel names
        :param commport: target serial port
        :param baudrate: target data transfer rate (in bits/sec)
        :param frame_shape: for line data, tuple of (num_points, window_size, num_channels); for image data, tuple of (height, width, num_channels)
        :param dtype: data type to store in shared memory object
        """
        self.dtype = dtype
        self.data_mode = data_mode

        # Defines start method for multiprocessing. Necessary for windows and macOS
        self.os_flag = setup_process_start_method()

        # Setup serial and plot channels
        self.ser_channel_key, self.plot_channel_key = self.setup_channel_keys(
                                                      ser_channel_key=ser_channel_key,
                                                      **kwargs)
        # Setup ring buffer
        self.ring, self.logical_shape = initialize_ring(ser_channel_key=ser_channel_key,
                                                        dtype=dtype,
                                                        data_mode=data_mode,
                                                        frame_shape=frame_shape)
                        
        # Setup target consumer params and enforce
        plot_target_fps = kwargs.get("plot_target_fps", 60.0)
        plot_catch_up_max = kwargs.get("plot_catchup_base_max", 2048)
        plot_catchup_boost = kwargs.get("plot_catchup_boost", 2.5)

        plot_target_fps = _coerce(plot_target_fps, 60.0)
        plot_catch_up_max = int(_coerce(plot_catch_up_max, 2048))
        plot_catchup_boost = _coerce(plot_catchup_boost, 2.5)

        # Setup static args dict
        self.static_args_dict = create_static_dict(ser_channel_key=self.ser_channel_key,
                                                   plot_channel_key=self.plot_channel_key,
                                                   commport=commport,
                                                   baudrate=baudrate,
                                                   shm_name='/sensor_ring',
                                                   shape=self.logical_shape,
                                                   dtype=dtype,
                                                   ring_capacity=4096,
                                                   data_mode=data_mode,
                                                   frame_shape=self.logical_shape
                                                   )
        
        self.static_args_dict = update_static_dict(static_args_dict=self.static_args_dict,
                                                   plot_target_fps=plot_target_fps,
                                                   plot_catch_up_max=plot_catch_up_max,
                                                   plot_catchup_boost=plot_catchup_boost
                                                   )
       

        # Make shared proxies for metrics
        self._mp_manager = Manager()
        self.writer_metrics_proxy = self._mp_manager.dict()
        self.plot_metrics_proxy = self._mp_manager.dict()
        self.ingest_metrics_proxy = self._mp_manager.dict()
        self.stream_ctrl_proxy = self._mp_manager.dict()
        # Initialize proxies
        self.writer_metrics_proxy.update({"init": True})
        self.plot_metrics_proxy.update({"init": True})
        self.ingest_metrics_proxy.update({"init": True})
        self.stream_ctrl_proxy.update({"force_rotate": False})

        # Make shared proxy for DSP
        self._plot_dsp = DSPManager()
        self._plot_dsp_version = 0

        self.plot_dsp_proxy = self._mp_manager.dict()
        self.plot_dsp_proxy.update({
            "version": 0,
            "queue": [],
            "modules": {},
        })

        # Derive default bin paths
        if start_stream_ingest and (fast_stream_path_a is None or fast_stream_path_b is None):
            base = pathlib.Path(sqlite_path).with_suffix('').as_posix()
            fast_stream_path_a = fast_stream_path_a or f"{base}_stream_a.bin"
            fast_stream_path_b = fast_stream_path_b or f"{base}_stream_b.bin"

        # Normalize ALL paths to absolute
        if sqlite_path is not None:
            sqlite_path = os.path.abspath(sqlite_path)
        if fast_stream_path_a is not None:
            fast_stream_path_a = os.path.abspath(fast_stream_path_a)
        if fast_stream_path_b is not None:
            fast_stream_path_b = os.path.abspath(fast_stream_path_b)

        self.fast_stream_path_a = fast_stream_path_a
        self.fast_stream_path_b = fast_stream_path_b
        self.sqlite_path = sqlite_path

        auto_enable_for_image = (self.data_mode.lower() == "image")
        ingest_enabled = bool(start_stream_ingest or auto_enable_for_image)

        # Setup on-disk dual bytestream
        try:
            from sensor_core.memory.stream_logger import dump_loop as _dump_loop
            ring_args = self.static_args_dict
            shm_name = ring_args.get('shm_name', '/sensor_ring')
            capacity = int(ring_args.get('ring_capacity', 4096))
            frame_shape = ring_args.get('logical_shape', self.logical_shape)
            _dtype = ring_args.get("dtype", dtype)
            self._stream_proc = Process(target=_dump_loop,
                                        args=(fast_stream_path_a, fast_stream_path_b, shm_name, capacity, frame_shape, _dtype),
                                        kwargs={'overwrite': False,
                                                'rotate_frames': int(rotate_frames),
                                                'rotate_seconds': float(rotate_seconds) if rotate_seconds else None,
                                                'metrics_proxy': self.writer_metrics_proxy,
                                                'control_proxy': self.stream_ctrl_proxy,
                                                'data_mode': self.data_mode,
                                                })
            self.start_process(self._stream_proc)
        except Exception as e:
            print(f'[SensorManager] failed to start stream logger: {e}')
            self._stream_proc = None

        if start_stream_ingest:
            if ingest_enabled:
                try:
                    from sensor_core.memory.db_ingester import ingest_loop as _ingest_loop
                    ch_keys = list(self.ser_channel_key) if isinstance(self.ser_channel_key, (list, tuple, np.ndarray)) else [self.ser_channel_key]
                    frame_shape = ring_args.get('logical_shape', self.logical_shape)
                    _dtype = ring_args.get("dtype", dtype)
                    # TO DO: change away from 'hint' and just call them the actual kwargs (i.e. frame)shape, data_mode)
                    self._ingest_proc = Process(target=_ingest_loop,
                                                args=(fast_stream_path_a, fast_stream_path_b, sqlite_path, ch_keys),
                                                kwargs={'metrics_proxy': self.ingest_metrics_proxy,
                                                        'data_mode_hint': data_mode,
                                                        'frame_shape_hint': frame_shape,
                                                        'dtype_hint': _dtype,
                                                        'precreate_sqlite': True
                                                        })
                    self.ingest_metrics_proxy.update({
                        "ingest_config_enabled": True,
                        "ingest_config_reason": ("explicit_flag" if start_stream_ingest else "auto_image_mode")
                    })
                    self.start_process(self._ingest_proc)
                except Exception as e:
                    print(f'[SensorManager] failed to start stream ingester: {e}')
                    self.ingest_metrics_proxy.update({
                        "ingest_config_enabled": True,
                        "ingest_last_error": f"spawn_failed: {e.__class__.__name__}: {e}",
                    })
        else:
            self._ingest_proc = None
            self.ingest_metrics_proxy.update({
                "ingest_config_enabled": False,
                "ingest_config_reason": "disabled_by_config"
            })

        if self._ingest_proc is not None:
            t = Thread(target=self._watch_ingester, daemon=True)
            t.start()

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
        from sensor_core.plot import PlotManager

        pm = PlotManager(static_args_dict=self.static_args_dict,
                         metrics_proxy=self.plot_metrics_proxy,
                         plot_dsp_proxy=self.plot_dsp_proxy,)
        if self.os_flag == 'win':
            p = Thread(name='plot',
                       target=pm.online_plot_data)
        else:
            p = Process(name='plot',
                        target=pm.online_plot_data)

        return p, pm.fig



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
            "ingest": dict(self.ingest_metrics_proxy),
        }

    def force_seal_now(self):
        """Request the writer to seal current bin and switch immediately."""
        self.stream_ctrl_proxy.update({"force_rotate": True})

    def debug_status(self) -> dict:
        return {
            "writer_proc": {
                "pid": getattr(self._stream_proc, "pid", None),
                "alive": (self._stream_proc.is_alive() if self._stream_proc else False),
            },
            "ingest_proc": {
                "pid": getattr(self._ingest_proc, "pid", None),
                "alive": (self._ingest_proc.is_alive() if self._ingest_proc else False),
            },
            "paths": {
                "bin_a": self.fast_stream_path_a,
                "bin_b": self.fast_stream_path_b,
                "sqlite": self.sqlite_path,
            },
        }

    def _watch_ingester(self):
        while True:
            time.sleep(1.0)
            if not (self._ingest_proc and self._ingest_proc.is_alive()):
                self.ingest_metrics_proxy.update({
                    "ingest_alive": False,
                    "ingest_last_error": self.ingest_metrics_proxy.get("ingest_last_error", "ingester process not alive"),
                    "ingest_updated_unix": time.time(),
                })
                break

    def _publish_plot_dsp_cfg(self):
        self._plot_dsp_version += 1
        self.plot_dsp_proxy["version"] = int(self._plot_dsp_version)
        self.plot_dsp_proxy["queue"] = list(self._plot_dsp.dsp_modules_queue)
        self.plot_dsp_proxy["modules"] = dict(self._plot_dsp.dsp_modules)

    # UI entrypoints
    def add_plot_dsp_module(self, name: str, target_algo: str, **kwargs):
        """
        Add a visualization-only DSP module. Does not affect stored data.
        Example:
          add_plot_dsp_module("ma1", "moving_average_filter", window_size=16, pad="percentile")
          add_plot_dsp_module("bp1", "butterworth_filter", order=2, min_frq=1, max_frq=20, fs=200)
        """
        self._plot_dsp.add_dsp_module(name, target_algo, **kwargs)
        self._publish_plot_dsp_cfg()

    def remove_plot_dsp_module(self, name: str):
        self._plot_dsp.remove_dsp_module(name)
        self._publish_plot_dsp_cfg()

    def clear_plot_dsp_modules(self):
        self._plot_dsp.dsp_modules.clear()
        self._plot_dsp.dsp_modules_queue.clear()
        self._publish_plot_dsp_cfg()

    def set_plot_dsp_order(self, new_queue):
        new_queue = list(new_queue)
        if set(new_queue) != set(self._plot_dsp.dsp_modules_queue):
            raise ValueError("new_queue must contain exactly the existing DSP module names.")
        self._plot_dsp.dsp_modules_queue = new_queue
        self._publish_plot_dsp_cfg()

    def get_plot_dsp_cfg(self) -> dict:
        return {
            "version": int(self._plot_dsp_version),
            "queue": list(self._plot_dsp.dsp_modules_queue),
            "modules": dict(self._plot_dsp.dsp_modules),
        }
