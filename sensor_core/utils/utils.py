import numpy as np
import sys
import multiprocessing
from typing import *


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
    def __init__(self, args_dict: dict = None, dict_type: str = None):
        """ Initialize Dictionary Manager class - handles unpacking of argument dicts.

        :param args_dict: argument dictionary. Can be static or dynamic parameter dictionary
        :param dict_type: denotes what type of dictionary to unpack (static or dynamic)
         """
        self.args_dict = args_dict
        self.dict_type = dict_type

    @staticmethod
    def create_static_dict(channel_key: Union[np.ndarray, str], commport: str, baudrate: int,
                           **kwargs):
        valid_keys = ['channel_key', 'commport', 'baudrate', 'mutex',
                      'shm_name', 'shape', 'dtype', 'EOL',
                      'num_points', 'num_channel']
        static_args_dict = {
            "channel_key": channel_key,
            "commport": commport,
            "baudrate": baudrate
        }
        for key in kwargs:
            if key in valid_keys:
                static_args_dict[f"{key}"] = kwargs[f"{key}"]
            else:
                Warning(f"Key {key} is not an acceptable input in static_args_dict.\n"
                        f"omitted from dictionary.")
        return static_args_dict

    @staticmethod
    def update_static_dict(static_args_dict: dict, **kwargs):
        valid_keys = ['channel_key', 'commport', 'baudrate', 'mutex',
                      'shm_name', 'shape', 'dtype', 'EOL',
                      'num_points', 'num_channel']
        for key in kwargs:
            if key in valid_keys:
                static_args_dict[f"{key}"] = kwargs[f"{key}"]
            else:
                Warning(f"Key {key} is not an acceptable input in static_args_dict.\n"
                        f"omitted from dictionary.")
        return static_args_dict

    def select_dictionary(self, args_dict: dict, dict_type: str):
        """ Function to update class attribute values for args dict and type
        :param args_dict: argument dictionary. Can be static or dynamic parameter dictionary
        :param dict_type: denotes what type of dictionary to unpack (static or dynamic)
        """
        self.args_dict = args_dict
        self.dict_type = dict_type

    def unpack_selected_dict(self):
        """ Selects which dictionary to unpack based upon input arguments
        Key Arguments: dict_type (static or dynamic)
        """
        if self.dict_type == "static":
            self.unpack_online_static_dict()
        elif self.dict_type == "dynamic":
            self.unpack_dynamic_dict()

    def unpack_online_static_dict(self):
        """ Unpack static parameter dictionary for online/real-time use
        :return: adds attributes to self for each key in dict
        """
        essential_keys = ['channel_key', 'commport', 'baudrate', 'mutex',
                          'shm_name', 'shape', 'dtype']
        optional_keys = ['EOL', 'num_points', 'num_channel']

        for key in essential_keys:
            try:
                setattr(self, f"{key}", self.args_dict[f"{key}"])
            except KeyError:
                raise KeyError(f"selected dictionary doesn't have key {key}\n"
                               f"dictionary must include the following keys:\n"
                               f"{essential_keys}")

        for key in optional_keys:
            try:
                setattr(self, f"{key}", self.args_dict[f"{key}"])
            except KeyError:
                setattr(self, f"{key}", None)

    def unpack_dynamic_dict(self):
        """ Unpack dynamic parameter dictionary (only for online use)
        :return: adds attributes to self for each key in dict
        """
        essential_keys = ['num_points', 'window_size']

        for key in essential_keys:
            try:
                setattr(self, f"{key}", self.args_dict[f"{key}"])
            except KeyError:
                raise KeyError(f"selected dictionary doesn't have key {key}\n"
                               f"dictionary must include the following keys:\n"
                               f"{essential_keys}")
