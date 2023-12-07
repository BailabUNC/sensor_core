from sensor_core.utils.utils import DictManager
import numpy as np
import scipy.signal as signal

class DSPManager:
    def __init__(self):
        """Initialize empty array and create empty dictionary
        :param dsp_modules_queue: Array of dsp_module object names
        :param dsp_modules: master dsp_module dictionary of dsp_modules containing, name, algo, and key parameters
        """
        self.dsp_modules = {}
        self.dsp_modules_queue = []

    def add_dsp_module(self, name: str, target_algo: str, **kwargs):
        ma_keys = {'window_size': int, 'pad': str}
        bw_keys = {'order': int, 'min_frq': int, 'max_frq': int}

        if name not in self.dsp_modules_queue:
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

        else:
            raise ValueError(f"Inputted name already exists: {name}")


    def remove_dsp_module(self, name: str):
        if name not in self.dsp_modules_queue:
            raise ValueError(f"Specified module does not exist: {name}")
        else:
            self.dsp_modules_queue.remove(name)
            del self.dsp_modules[name]

    def run_dsp_modules(self, data):
        new_data = np.copy(data)

        for name in self.dsp_modules_queue:
            module = self.dsp_modules[name]
            target_algo = module['algo']

            if target_algo == 'moving_average_filter':
                window_size = module['window_size']
                pad = module['pad']
                filtered_data = self.moving_average_filter(new_data, window_size, pad)

            elif target_algo == 'butterworth_filter':
                min_frq = module['min_frq']
                max_frq = module['max_frq']
                order = module['order']
                filtered_data = self.butterworth_filter(new_data, min_frq, max_frq, order)
            else:
                raise ValueError(f"Unsupported module algorithm: {target_algo}")
            
            new_data = filtered_data
        
        return new_data

    @staticmethod
    def moving_average_filter(data, window_size: int, pad: str):
        if pad == 'min':
            data_arr = np.concatenate(([np.min(data)] * (window_size), data))
        elif pad == 'percentile':
            data_arr = np.concatenate(([np.percentile(data, 10)] * (window_size), data))
        else:
            raise ValueError("Invalid moving_average_filter parameter, remove module and specify 'min' or 'percentile' to pad array with")
        
        cumsum = np.cumsum(data_arr)
        filtered = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
        return filtered
    
    @staticmethod
    def butterworth_filter(data, min_frq: int, max_frq: int, order: int):
        #Bandpass implementation
        b, a = signal.butter(order, [min_frq, max_frq], btype = 'bandpass', analog = False, output = 'ba', fs = 100)
        filtered = signal.filtfilt(b, a, data)

        return filtered