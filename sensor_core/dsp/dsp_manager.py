from sensor_core.utils.utils import DictManager
import numpy as np
import scipy.signal as signal

class DSPManager(DictManager):
    def _init_(self, dsp_module_queue: np.array([]), dsp_modules: dict = None):
        """Initialize empty array and create empty dictionary
        :param dsp_modules: Array of dsp_module objects, identified by user-defined name, target algo, and key parameters
        :param dsp_dict: master dsp_module dictionary
        :param dict_type: Dynamic dictionary """

        self.dsp_modules = {}
        self.dsp_module_queue = []

    def add_dsp_module(self, name: str, target_algo: str, **kwargs):
        ma_keys = {'N': int, 'pad': str}
        bw_keys = {'order': int, 'min_frq': int, 'max_frq': int, 'btype': str}

        if (target_algo == 'moving_average_filter'):
            if not set(ma_keys).issubset(kwargs.keys()):
                raise ValueError(f"Missing required parameters: {ma_keys}")
            unexpected_params = set(kwargs.keys()) - set(ma_keys)
            if unexpected_params:
                raise ValueError(f"Invalid moving average input, Unexpected parameters: {unexpected_params}")
            
            self.dsp_modules[name] = {'algo': target_algo, **kwargs}
            self.dsp_module_queue.append(name)

        elif (target_algo == 'butterworth_filter'):
            if not set(bw_keys).issubset(kwargs.keys()):
                raise ValueError(f"Missing required parameters: {bw_keys}")
            
            unexpected_params = set(kwargs.keys()) - set(bw_keys)
            if unexpected_params:
                raise ValueError(f"Invalid moving average input, Unexpected parameters: {unexpected_params}")
            
            self.dsp_modules[name] = {'algo': target_algo, **kwargs}
            self.dsp_module_queue.append(name)

    @staticmethod
    def moving_average_filter(data, N, pad):
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
    def butterworth_filter(data, min_frq, max_frq, order, btype):
        if((btype == 'lowpass') or (btype == 'highpass') or (btype == 'bandpass') or (btype == 'bandstop')):
            b, a = signal.butter(order, [min_frq, max_frq], btype = btype, analog = False, output = 'ba', fs = 100)
            filtered = signal.filtfilt(b, a, data)
        else:
            raise ValueError("Specify btype 'lowpass', 'highpass', 'bandpass', or 'bandstop'")
        return filtered