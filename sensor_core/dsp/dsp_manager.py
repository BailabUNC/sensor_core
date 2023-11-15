import sensor_core.memory.mem_manager as mm
from sensor_core.utils.utils import DictManager

import numpy as np
import scipy.signal as signal

class DSPManager(DictManager):
    def __init__(self, channel_key, commport: str, baudrate: int, window_size: int = 1):

        self.channel_key = channel_key
        self.commport = commport
        self.baudrate = baudrate
        self.window_size = window_size
        self.update_dictionary(args_dict=static_args_dict,
            dict_type="static")
        
        static_args_dict = {
            "channel_key": channel_key,
            "commport": commport,
            "baudrate": baudrate
        }

    def select_dsp_mod(dsp_module, data, N):
        if (dsp_module == 'moving_avg'):
            pad = str(input("Enter 'min' or 'percentile' to pad array with"))
            return DSPManager.moving_avg_filter(data, N, pad)
        
        elif (dsp_module == 'butterworth'):
            min_frq = int(input("Enter min frequency cutoff"))
            max_frq = int(input("Enter max frequency cutoff"))
            btype = str(input("Enter 'lowpass', 'highpass', 'bandpass', or 'bandstop'"))
            return DSPManager.butter_filter(data, min_frq, max_frq, N, btype)
        else:
            raise ValueError("Specify 'moving_average' or 'butterworth'")
            

    @staticmethod
    def moving_avg_filter(data, N, pad):
        if(pad == 'min'):
            data_arr = np.concatenate(([np.min(data)] * (N), data))
        elif(pad == 'percentile'):
            data_arr = np.concatenate(([np.percentile(data, 10)] * (N), data))
        else:
            raise ValueError("Specify 'min' or 'percentile' to pad array with")
        
        cumsum = np.cumsum(data_arr)
        filtered = (cumsum[N:] - cumsum[:-N]) / N
        return filtered
        
    @staticmethod
    def butter_filter(data, min_frq, max_frq, N, btype):
        if((btype == 'lowpass') or (btype == 'highpass') or (btype == 'bandpass') or (btype == 'bandstop')):
            b, a = signal.butter(N, [min_frq, max_frq], btype = btype, analog = False, output = 'ba', fs = 100)
            filtered = signal.filtfilt(b, a, data)
            return filtered
        else:
            raise ValueError("Specify btype 'lowpass', 'highpass', 'bandpass', or 'bandstop'")