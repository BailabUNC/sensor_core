from sensor_core.utils.utils import DictManager

import numpy as np
import scipy.signal as signal

class DSPManager(DictManager):
    def __init__(self, dsp_module):
        valid_modules = ['moving_avg', 'butterworth']

        if dsp_module in valid_modules:
            self.dsp_module = dsp_module

        else:
            Warning(f"{dsp_module} is not an acceptable input for dsp_module.\n")

        """dsp_dict = {
            "moving_avg": {"N": int, "pad": str},
            "butterworth": {"min_frq": int, "max_frq": int, "order": int, "btype": str }
        }
        """

    def set_attributes(dsp_module, **kwargs):
        ma_keys = ['data', 'N', 'pad']
        bw_keys = ['data', 'order', 'min_frq', 'max_frq', 'btype']
  
        if (dsp_module == 'moving_avg'):
            for key, value in kwargs.items():
                if key in ma_keys:
                    setattr(self, key, value)    

        elif (dsp_module == 'butterworth'):
            for key, value in kwargs.items():
                if key in bw_keys:
                    setattr(bw_module, key, value)
                else:
                    raise KeyError("Invalid butterworth parameter input") 
           
    
    def select_dsp_mod(dsp_module,**kwargs):
        
        #return getattr(ma_module, 'moving_avg_filter')(ma_module.data, ma_module.N, ma_module.pad)
        #getattr(bw_module, 'butter_filter')(bw_module.data, bw_module.N)  

        if (dsp_module == 'moving_avg'):
            return DSPManager.moving_avg_filter()
        
        if (dsp_module == 'butterworth'):
            return DSPManager.butter_filter()
            
           
            

    @staticmethod
    def moving_avg_filter(data, N, pad):
        """N = Window size"""
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
    def butter_filter(data, min_frq, max_frq, order, btype):
        if((btype == 'lowpass') or (btype == 'highpass') or (btype == 'bandpass') or (btype == 'bandstop')): 
            b, a = signal.butter(order, [min_frq, max_frq], btype = btype, analog = False, output = 'ba', fs = 100)
            filtered = signal.filtfilt(b, a, data)
        else:
            raise ValueError("Specify btype 'lowpass', 'highpass', 'bandpass', or 'bandstop'")
        
        return filtered