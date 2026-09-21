from sensor_core.utils.metrics import RingMetrics, timer
from sensor_core.memory.ring_adapter import RingBuffer
import numpy as np
from sensor_core.memory.mem_utils import _assert_ring_layout
from sensor_core.serial import SerialManager
from sensor_core.utils import DictManager
from sensor_core.memory.strg_manager import StorageManager
from time import perf_counter
import multiprocessing
import time, traceback


class DataManager(SerialManager, DictManager, StorageManager):
    def __init__(self,
                 static_args_dict: dict, 
                 virtual_ser_port: bool = False,
                 save_data: bool = False, 
                 filepath: str = None, 
                 overwrite_data: bool = True, 
                 metrics_proxy=None):
        """ Online Data Manager
        - handles serial port initialization and management of data for the online (real-time) use case ONL
        :param static_args_dict: dictionary containing key parameters for initialization
        :param virtual_ser_port: boolean, if True will not initialize serial port, instead will rely on user-defined
        custom function to generate simulated data
        :param save_data: boolean, determines whether to save data
        :param filepath: filepath to save data to
        :param overwrite_data: boolean, decides whether to overwrite existing saved data
        """
        self.metrics = RingMetrics()
        self.static_args_dict = static_args_dict
        self.save_data = save_data
        self._metrics_proxy = metrics_proxy

        # Initialize DictManager Subclass
        DictManager.__init__(self)

        # Unpack static_args_dict
        self.select_dictionary(args_dict=self.static_args_dict,
                               dict_type="static")

        self.unpack_selected_dict()

        # Initialize Buffer
        self.ring = RingBuffer(self.shm_name,
                               int(self.ring_capacity), 
                               tuple(self.shape), 
                               self.data_mode,
                               self.dtype,create=False)
        _assert_ring_layout(self.ring, tuple(self.shape), self.dtype)

        # Start serial port
        self.start_serial(virtual_ser_port=virtual_ser_port)

        # Create serial database
        if save_data:
            StorageManager.__init__(self, channel_key=self.ser_channel_key,
                                    filepath=filepath, overwrite=overwrite_data)
            self.create_serial_database()

    def start_serial(self, virtual_ser_port):
        """ Initialize SerialManager subclass, and setup serial port

        """
        SerialManager.__init__(self, commport=self.commport,
                               baudrate=self.baudrate,
                               frame_shape=self.shape,
                               EOL=self.EOL,
                               virtual_ser_port=virtual_ser_port)
        self.setup_serial()

    def online_update_data(self, func=None, stop_event=None):
        """ Acquire frames and publish them to the ring buffer until stop_event is set
        :param func: optional custom acquisition function, called as func(ser=..., frame_shape=...)
        :param stop_event: multiprocessing.Event that ends the loop when set
        """
        last_push = perf_counter()
        last_log = time.time()
        parent = multiprocessing.parent_process()
        next_parent_check = 0.0
        while stop_event is None or not stop_event.is_set():
            if parent is not None and perf_counter() >= next_parent_check:
                if not parent.is_alive():
                    break  # the process that owns the ring is gone
                next_parent_check = perf_counter() + 0.5
            try:
                with timer(lambda ms: self.metrics.add_acquire_ms(ms)):
                    ys = self.acquire_data(func=func,
                                           data_mode=self.data_mode)
                acquired_ns = time.perf_counter_ns()  # when this frame reached the host
                if ys is None:
                    if time.time() - last_log > 1.0:
                        print("[writer] acquire_data -> None")
                        last_log = time.time()
                    continue

                if self.data_mode=='line':
                    # one acquisition, shaped (window_size, channels), fills one ring slot
                    with timer(lambda ms: self.metrics.note_publish(ms, write_idx=int(self.ring.write_idx))):
                        self.ring.publish(ys, acquired_ns)
                else:
                    with timer(lambda ms: self.metrics.note_publish(ms)):
                        self.ring.publish(np.asarray(ys, dtype=self.dtype), acquired_ns)

                wi = int(self.ring.write_idx)
                self.metrics.last_write_idx = wi

                now = perf_counter()
                if self._metrics_proxy is not None and (now - last_push) > 0.5:
                    self._metrics_proxy.update(self.metrics.snapshot())
                    last_push = now

            except Exception as e:
                print("[writer] EXCEPTION:", repr(e))
                traceback.print_exc()
                time.sleep(0.02)


