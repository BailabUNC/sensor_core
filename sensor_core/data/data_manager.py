from sensor_core.utils.metrics import RingMetrics, timer
from sensor_core.memory.ring_adapter import RingBuffer
import numpy as np
from sensor_core.memory.mem_utils import _assert_ring_layout
from sensor_core.serial import SerialManager
from sensor_core.utils import DictManager
from sensor_core.memory.strg_manager import StorageManager
from time import perf_counter
import time, traceback


class DataManager(SerialManager, DictManager, StorageManager):
    def __init__(self, static_args_dict: dict, dynamic_args_queue, virtual_ser_port: bool = False,
                 save_data: bool = False, filepath: str = None, overwrite_data: bool = True, metrics_proxy=None):
        """ Online Data Manager
        - handles serial port initialization and management of data for the online (real-time) use case ONL
        :param static_args_dict: dictionary containing key parameters for initialization
        :param dynamic_args_queue: queue used to send dictionary with dynamic parameters
        :param virtual_ser_port: boolean, if True will not initialize serial port, instead will rely on user-defined
        custom function to generate simulated data
        :param save_data: boolean, determines whether to save data
        :param filepath: filepath to save data to
        :param overwrite_data: boolean, decides whether to overwrite existing saved data
        """
        self.metrics = RingMetrics()
        self.static_args_dict = static_args_dict
        self.dynamic_args_queue = dynamic_args_queue
        self.save_data = save_data
        self._metrics_proxy = metrics_proxy

        # Initialize DictManager Subclass
        DictManager.__init__(self)

        # Unpack static_args_dict
        self.select_dictionary(args_dict=self.static_args_dict,
                               dict_type="static")

        self.unpack_selected_dict()

        self.ring = RingBuffer(self.shm_name, int(self.ring_capacity), tuple(self.shape), self.dtype, create=False)
        _assert_ring_layout(self.ring, tuple(self.shape), self.dtype)
        # Unpack dynamic_args_dict if it exists
        try:
            self.dynamic_args_dict = self.dynamic_args_queue.get()
        except:
            raise ValueError(f"Queue {self.dynamic_args_queue} initialization failed. "
                             f"Please restart process")

        self.select_dictionary(args_dict=self.dynamic_args_dict,
                               dict_type="dynamic")
        self.unpack_selected_dict()

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
                               num_channel=self.num_channel,
                               window_size=self.window_size,
                               EOL=self.EOL,
                               virtual_ser_port=virtual_ser_port)
        self.setup_serial()

    def online_update_data(self, func=None):
        last_push = perf_counter()
        while True:
            try:
                with timer(lambda ms: self.metrics.add_acquire_ms(ms)):  # reusing bucket if you like; or add another
                    ys = self.acquire_data(func=func)
                if ys is None:
                    print("[writer] acquire_data -> None")
                    continue

                arr = np.asarray(ys)
                if arr.ndim != 2:
                    raise ValueError(f"[writer] ys ndim={arr.ndim}, expected 2 (L,C), got {arr.shape}")

                L_in, C_in = arr.shape
                C_ring, L_ring = self.shape  # ring frame is (C, L)

                if C_in != C_ring:
                    raise ValueError(f"[writer] channels mismatch: ys (L,{C_in}), ring expects C={C_ring}")

                # Clamp to ring L
                L = min(L_in, L_ring)
                if L != L_ring:
                    arr = arr[:L, :]

                # Confirm frame is contiguous and transpose, then publish to ring
                frame = np.ascontiguousarray(arr.T, dtype=np.float32)
                with timer(lambda ms: self.metrics.note_publish(ms, write_idx=int(self.ring.write_idx))):
                    self.ring.publish(frame)

                wi = int(self.ring.write_idx)
                self.metrics.last_write_idx = wi

                now = perf_counter()
                if self._metrics_proxy is not None and (now - last_push) > 0.5:
                    self._metrics_proxy.update(self.metrics.snapshot())
                    last_push = now

            except Exception as e:
                print("[writer] EXCEPTION:", repr(e))
                traceback.print_exc()
                time.sleep(0.05)


