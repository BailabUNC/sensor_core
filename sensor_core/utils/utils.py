import numpy as np
import sys
import multiprocessing


def setup_process_start_method():
    """ Set up multiprocessing start method based on operating system

    """
    if sys.platform.startswith('win'):
        multiprocessing.set_start_method("spawn", force=True)
        os_flag = 'win'
    elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin') or sys.platform.startswith('darwin'):
        multiprocessing.set_start_method("fork", force=True)
        os_flag = 'unix'
    else:
        raise EnvironmentError('Unsupported platform')
    return os_flag


class DictManager(object):
    def __init__(self, args_dict: dict = None, dict_type: str = None,
                 online: bool = False, multiproc: bool = False):
        """ Initialize Dictionary Manager class - handles unpacking of argument dicts.

        :param args_dict: argument dictionary. Can be static or dynamic parameter dictionary
        :param dict_type: denotes what type of dictionary to unpack (static or dynamic)
        :param online: bool flag, denotes whether to unpack online-specific arguments
        :param multiproc: bool flag. denotes whether to unpack multiprocessing specific arguments
        """
        self.EOL = None
        self.num_channel = None
        self.baudrate = None
        self.commport = None
        self.channel_key = None
        self.shape = None
        self.dtype = None
        self.mutex = None
        self.shm_name = None
        self.num_points = None
        self.window_size = None
        self.args_dict = args_dict
        self.dict_type = dict_type
        self.online = online
        self.multiproc = multiproc

    def update_dictionary(self, args_dict: dict, dict_type: str):
        """ Function to update class attribute values for args dict and type
        :param args_dict: argument dictionary. Can be static or dynamic parameter dictionary
        :param dict_type: denotes what type of dictionary to unpack (static or dynamic)
        """
        self.args_dict = args_dict
        self.dict_type = dict_type

    def unpack_selected_dict(self):
        """ Selects which dictionary to unpack based upon input arguments
        Key Arguments: online (real-time) flag and dict_type (static or dynamic)

        """

        if self.online:
            if self.dict_type == "static":
                self.output_dict = self.unpack_online_static_dict()
            elif self.dict_type == "dynamic":
                self.output_dict = self.unpack_dynamic_dict()
        else:
            self.output_dict = self.unpack_offline_static_dict()

    def unpack_online_static_dict(self):
        """ Unpack static parameter dictionary for online/real-time use

        """
        if self.multiproc:
            try:
                self.shm_name = self.args_dict["shm_name"]
                self.mutex = self.args_dict["mutex"]
                self.shape = self.args_dict["shape"]
                self.dtype = self.args_dict["dtype"]
                self.channel_key = self.args_dict["channel_key"]
                self.commport = self.args_dict["commport"]
                self.baudrate = self.args_dict["baudrate"]
                self.num_channel = np.shape(self.channel_key)[0]
                self.EOL = self.args_dict["EOL"]
                self.num_points = self.args_dict["num_points"]
            except ValueError:
                raise ValueError(f"static_args dict {self.args_dict}"
                                 f"should contain the following keys: \n"
                                 f"shm_name, mutex, shape, dtype, channel_key, "
                                 f"commport, baudrate, and EOL")
        else:
            try:
                self.channel_key = self.args_dict["channel_key"]
                self.commport = self.args_dict["commport"]
                self.baudrate = self.args_dict["baudrate"]
                self.num_channel = np.shape(self.channel_key)[0]
                self.EOL = self.args_dict["EOL"]
                self.num_points = self.args_dict["num_points"]
            except ValueError:
                raise ValueError(f"static_args dict {self.args_dict}"
                                 f"should contain the following keys: \n"
                                 f"channel_key, commport, baudrate, and EOL")

    def unpack_offline_static_dict(self):
        """ Unpack static parameter dictionary for offline use

        """
        try:
            self.channel_key = self.args_dict["channel_key"]
            self.num_channel = np.shape(self.channel_key)[0]
            self.num_points = self.args_dict["num_points"]
        except ValueError:
            raise ValueError(f"static_args dict {self.args_dict}"
                             f"should contain the following key: \n"
                             f"channel_key")

    def unpack_dynamic_dict(self):
        """ Unpack dynamic parameter dictionary (only for online use)

        """
        try:
            self.num_points = self.args_dict["num_points"]
            self.window_size = self.args_dict["window_size"]
        except ValueError:
            raise ValueError(f"dynamic args dict {self.args_dict}"
                             f"should contain the following keys: \n"
                             f"num_points and window_size")
