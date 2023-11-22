import sensor_core.memory.mem_manager as mm
from sensor_core.utils.utils import DictManager

import numpy as np
import scipy.signal as signal

class DSPManager(DictManager):
    def __init__(self, dsp_module):

        self.dsp_module = dsp_module

        if (dsp_module != 'moving_avg' or dsp_module != 'butterworth'):
            raise ValueError('Must enter valid dsp_module')

        self.update_dictionary(args_dict=dsp_dict,
            dict_type="static")

        dsp_dict = {
            "moving_avg": {"data"},
            "butterworth": { "order": 4, "N": 100000000}
        }
    
    #some function(dsp_list, **kwargs)
    #for module in dsp_list:
        #check for key in kwargs for subset of input

    def select_dsp_mod(dsp_module, data, N, pad):
        if (dsp_module == 'moving_avg'):
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
        else:
            raise ValueError("Specify btype 'lowpass', 'highpass', 'bandpass', or 'bandstop'")
        
        return filtered