import numpy as np
import sensor_core.memory as mm
from sensor_core.serial import SerialManager
from sensor_core.utils import DictManager
from sensor_core.memory.strg_manager import StorageManager


class DataManager(SerialManager, DictManager, StorageManager):
    def __init__(self, static_args_dict: dict, dynamic_args_queue, virtual_ser_port: bool = False,
                 save_data: bool = False, filepath: str = None, overwrite_data: bool = True):
        """ Initialize Online Data Manager - handles serial port initialization and management
        of data for the online (real-time) use case ONL
        :param static_args_dict: dictionary containing key parameters for initialization
        :param dynamic_args_queue: queue used to send dictionary with dynamic parameters
        :param virtual_ser_port: boolean, if True will not initialize serial port, instead will rely on user-defined
        custom function to generate simulated data
        :param save_data: boolean, determines whether to save data
        :param filepath: filepath to save data to
        :param overwrite_data: boolean, decides whether to overwrite existing saved data
        """
        self.static_args_dict = static_args_dict
        self.dynamic_args_queue = dynamic_args_queue
        self.save_data = save_data

        # Initialize DictManager Subclass
        DictManager.__init__(self)

        # Unpack static_args_dict
        self.select_dictionary(args_dict=self.static_args_dict,
                               dict_type="static")

        self.unpack_selected_dict()

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
        """ Update data in real time
        :save_data: if flag is True, will continuously save data to hdf5 or sqlite3 file
        :param func: optional custom function for serial data acquisition handler
        """
        accumulated_frames = 0

        while True:
            try:
                if self.dynamic_args_queue.empty():
                    pass
                else:
                    self.dynamic_args_dict = self.dynamic_args_queue.get()
                    self.select_dictionary(args_dict=self.dynamic_args_dict,
                                           dict_type="dynamic")
                    self.unpack_selected_dict()
            except:
                self.window_size = 1

            ys = self.acquire_data(func=func)
            if ys is not None:
                serial_window_length = np.shape(ys)[0]
                shm = mm.SharedMemory(self.shm_name)
                # Acquire mutex
                mm.acquire_mutex(self.mutex)
                # Load shared memory object
                data_shared = np.ndarray(shape=self.shape, dtype=self.dtype,
                                         buffer=shm.buf)

                if self.save_data:
                    accumulated_frames += serial_window_length
                    if accumulated_frames < self.num_points:
                        self._online_update_data(curr_data=data_shared, new_data=ys,
                                                 serial_window_length=serial_window_length)
                    else:
                        data = self._online_update_data(curr_data=data_shared, new_data=ys,
                                                        serial_window_length=serial_window_length)
                        for i in range(self.shape[0] - 1):
                            save_data = data[i + 1][:]
                            self.append_serial_channel(key=self.ser_channel_key[i],
                                                       data=save_data)
                        accumulated_frames = 0

                else:
                    self._online_update_data(curr_data=data_shared, new_data=ys,
                                             serial_window_length=serial_window_length)

                # Release mutex
                mm.release_mutex(self.mutex)

    def _online_update_data(self, curr_data, new_data, serial_window_length):
        """ Function that handles rolling buffer update for the online_update_data
        :param curr_data: current/original data to update
        :param new_data: serial/new data to append
        :param serial_window_length: number of time points to update (append/pop)

        """
        for i in range(self.shape[0] - 1):
            curr_data[i + 1][:-serial_window_length] = curr_data[i + 1][serial_window_length:]
            curr_data[i + 1][-serial_window_length:] = new_data[:, i]
        return curr_data
