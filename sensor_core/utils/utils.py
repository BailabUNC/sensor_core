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


def create_static_dict(
    ser_channel_key,
    plot_channel_key,
    commport: str,
    baudrate: int,
    shm_name: str,
    shape: Tuple[int, ...],
    dtype=np.float32,
    ring_capacity: int = 4096,
    num_points: int = 1000,
    *,
    data_mode: str = "line",                 # <— NEW
    frame_shape: Optional[Tuple[int, ...]] = None  # <— NEW (logical shape)
) -> Dict:
    """Build the static args dict that both processes read."""
    return {
        "ser_channel_key": ser_channel_key,
        "plot_channel_key": plot_channel_key,
        "commport": commport,
        "baudrate": baudrate,
        "shm_name": shm_name,
        "shape": tuple(shape),
        "dtype": dtype,
        "ring_capacity": int(ring_capacity),
        "num_points": int(num_points),
        "data_mode": str(data_mode),
        "frame_shape": tuple(frame_shape) if frame_shape else None,
    }



def update_static_dict(static_args_dict: dict, **kwargs):
    """ Update existing static args dict
    :param static_args_dict: dictionary to update params
    :param kwargs: all parameters you wish to update
    :return: static_args_dict. Dictionary containing all static parameters
    """
    valid_keys = ['ser_channel_key', 'plot_channel_key',
                  'commport', 'baudrate', 'shm_name',
                  'shape', 'dtype', 'EOL', 'ring_capacity',
                  'num_points', 'num_channel', 'plot_target_fps',
                  'plot_catch_up_max', 'plot_catchup_boost',
                  'plot_lag_frames', 'data_mode']
    for key in kwargs:
        if key in valid_keys:
            static_args_dict[f"{key}"] = kwargs[f"{key}"]
        else:
            Warning(f"Key {key} is not an acceptable input in static_args_dict.\n"
                    f"omitted from dictionary.")
    return static_args_dict


def _coerce(val, default):
    return val if isinstance(val, (int, float)) and not isinstance(val, bool) else default

class DictManager(object):
    def __init__(self, args_dict: dict = None, dict_type: str = None):
        """ Initialize Dictionary Manager class - handles unpacking of argument dicts.

        :param args_dict: argument dictionary. Can be static or dynamic parameter dictionary
        :param dict_type: denotes what type of dictionary to unpack (static or dynamic)
         """
        self.args_dict = args_dict
        self.dict_type = dict_type

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
        self.unpack_online_static_dict()


    def unpack_online_static_dict(self):
        """ Unpack static parameter dictionary for online/real-time use
        :return: adds attributes to self for each key in dict
        """
        essential_keys = ['ser_channel_key', "plot_channel_key", 'commport',
                          'baudrate', 'shm_name', 'shape', 'dtype', 'ring_capacity',
                          'data_mode']
        optional_keys = ['EOL', 'num_points', 'num_channel', 'plot_target_fps',
                         'plot_catchup_base_max', 'plot_catchup_boost']

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
                if key == "num_channel":
                    num_channel = np.shape(self.args_dict["plot_channel_key"])[0] * \
                                  np.shape(self.args_dict["plot_channel_key"])[1]
                    setattr(self, f"{key}", num_channel)
                else:
                    setattr(self, f"{key}", None)

