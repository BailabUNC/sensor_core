from sensor_core.utils.utils import DictManager
import numpy as np
import scipy.signal as signal

class DSPManager(DictManager):
    def _init_(self, dsp_modules_queue: np.array([]), dsp_modules: dict = None):
        """Initialize empty array and create empty dictionary
        :param dsp_modules: Array of dsp_module objects, identified by user-defined name, target algo, and key parameters
        :param dsp_dict: master dsp_module dictionary
        :param dict_type: Dynamic dictionary """

        self.dsp_modules = {}
        self.dsp_modules_queue = []

    @staticmethod
    def add_dsp_module(self, name: str, target_algo: str, **kwargs):
        #TODO: Make sure no duplicate names, 
        # Make sure input parameters are correct data type
        # Currently only checks after run_dsp_module executes

        ma_keys = {'window_size': int, 'pad': str}
        bw_keys = {'order': int, 'min_frq': int, 'max_frq': int, 'btype': str}

        if target_algo == 'moving_average_filter':
            if not set(ma_keys).issubset(kwargs.keys()):
                raise ValueError(f"Missing required parameters: {ma_keys}")
            unexpected_params = set(kwargs.keys()) - set(ma_keys)
            if unexpected_params:
                raise ValueError(f"Invalid moving average input, unexpected parameters: {unexpected_params}")
            
            self.dsp_modules[name] = {'algo': target_algo, **kwargs}
            self.dsp_modules_queue.append(name)

        elif target_algo == 'butterworth_filter':
            if not set(bw_keys).issubset(kwargs.keys()):
                raise ValueError(f"Missing required parameters: {bw_keys}")
            
            unexpected_params = set(kwargs.keys()) - set(bw_keys)
            if unexpected_params:
                raise ValueError(f"Invalid moving average input, unexpected parameters: {unexpected_params}")
            
            self.dsp_modules[name] = {'algo': target_algo, **kwargs}
            self.dsp_modules_queue.append(name)

    @staticmethod
    def remove_dsp_module(self, name: str):
        if name not in self.dsp_modules_queue:
            raise ValueError(f"Specified module does not exist")
        else:
            rm_index = self.dsp_modules_queue.index(name)
            self.dsp_modules_queue.pop(rm_index)
            print(f"Update list: ", self.dsp_modules_queue)

    def run_dsp_modules(self, data):
        for name in self.dsp_modules_queue:
            module = self.dsp_modules[name]
            target_algo = module['algo']

            if target_algo == 'moving_average_filter':
                filtered_data = self.moving_average_filter(data, **module)
            elif target_algo == 'butterworth_filter':
                filtered_data = self.butterworth_filter(data, **module)
            else:
                raise ValueError(f"Unsupported module algorithm: {target_algo}")
        
        print("DSP Modules successfully run")
        return filtered_data

    @staticmethod
    def moving_average_filter(data, window_size: int, pad: str):
        if pad == 'min':
            data_arr = np.concatenate(([np.min(data)] * (window_size), data))
        elif pad == 'percentile':
            data_arr = np.concatenate(([np.percentile(data, 10)] * (), data))
        else:
            raise ValueError("Invalid moving_average_filter parameter, remove module and specify 'min' or 'percentile' to pad array with")
        
        cumsum = np.cumsum(data_arr)
        filtered = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
        return filtered
    
    @staticmethod
    def butterworth_filter(data, min_frq: int, max_frq: int, order: int, btype: str):
        if (btype == 'lowpass') or (btype == 'highpass') or (btype == 'bandpass') or (btype == 'bandstop'):
            b, a = signal.butter(order, [min_frq, max_frq], btype = btype, analog = False, output = 'ba', fs = 100)
            filtered = signal.filtfilt(b, a, data)
        else:
            raise ValueError("Invalid butterworth parameter, remove module, and specify btype 'lowpass', 'highpass', 'bandpass', or 'bandstop'")
        return filtered