import numpy as np
import MABOS_core.memory as mm
from MABOS_core.serial import SerialManager
from MABOS_core.utils import DictManager
from MABOS_core.memory.strg_manager import create_serial_database


class OnlineDataManager(SerialManager, DictManager):
    def __init__(self, static_args_dict, dynamic_args_queue=None,
                 save_data: bool = False, multiproc: bool = True):
        """ Initialize Online Data Manager - handles serial port initialization and management
        of data for the online (real-time) use case ONLY
        """
        self.static_args_dict = static_args_dict
        self.dynamic_args_queue = dynamic_args_queue
        self.save_data = save_data
        self.multiproc = multiproc

        # Initialize DictManager Subclass
        DictManager.__init__(self, online=True,
                             multiproc=self.multiproc)

        # Unpack static_args_dict
        self.update_dictionary(args_dict=self.static_args_dict,
                               dict_type="static")
        if self.online:
            self.unpack_online_static_dict()
        else:
            self.unpack_offline_static_dict()

        # Unpack dynamic_args_dict if we need online plotting and it exists
        if self.online is True and self.dynamic_args_queue is not None:
            try:
                self.dynamic_args_dict = self.dynamic_args_queue.get()
                self.num_points = self.dynamic_args_dict["num_points"]
                self.window_size = self.dynamic_args_dict["window_size"]

            except:
                raise ValueError(f"Queue {self.dynamic_args_queue} initialization failed. "
                                 f"Please restart process")
            self.update_dictionary(args_dict=self.dynamic_args_dict,
                                   dict_type="dynamic")
            self.select_dict_to_unpack()

        # Start serial port
        if self.online:
            self.start_serial()

        # Create serial database
        if save_data:
            create_serial_database(channel_key=self.channel_key, num_points=self.num_points, overwrite=True)

    def start_serial(self):
        """ Initialize SerialManager subclass, and setup serial port

        """
        SerialManager.__init__(self, commport=self.commport,
                               baudrate=self.baudrate,
                               num_channel=self.num_channel,
                               window_size=self.window_size,
                               EOL=self.EOL)
        self.setup_serial()

    def online_update_data(self):
        """ Update data in real time
        multiproc: if flag is True, will use shared memory object to share data between processes
        save_data: if flag is True, will continuously save data to hdf5 file
        """
        accumulated_frames = 0

        while True:
            if self.dynamic_args_queue.empty():
                pass
            else:
                self.dynamic_args_dict = self.dynamic_args_queue.get()
                self.num_points = self.dynamic_args_dict["num_points"]
                self.window_size = self.dynamic_args_dict["window_size"]

            ys = self.acquire_data()
            if ys is not None:
                serial_window_length = np.shape(ys)[0]
                if self.multiproc:
                    shm = mm.SharedMemory(self.shm_name)
                    mm.acquire_mutex(self.mutex)
                    data_shared = np.ndarray(shape=self.shape, dtype=self.dtype,
                                             buffer=shm.buf)
                else:
                    data_shared = None
                    pass
                    # TODO add way to run same class wtihout need for multiprocessing

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
                            mm.append_serial_channel(key=self.channel_key[i],
                                                     data=save_data)
                        accumulated_frames = 0
                else:
                    self._online_update_data(curr_data=data_shared, new_data=ys,
                                             serial_window_length=serial_window_length)

                if self.multiproc:
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
