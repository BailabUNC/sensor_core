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
        bw_keys = {'order': int, 'min_frq': int, 'max_frq': int, 'fs': int}

        if name in self.dsp_modules_queue:
            raise ValueError(f"Inputted name already exists: {name}")

        if target_algo == 'moving_average_filter':
            if not set(ma_keys).issubset(kwargs.keys()):
                raise ValueError(f"Missing required parameters: {ma_keys}")
            unexpected_params = set(kwargs.keys()) - set(ma_keys)
            if unexpected_params:
                raise ValueError(f"Invalid moving average input, unexpected parameters: {unexpected_params}")

            # basic validation
            if int(kwargs["window_size"]) < 1:
                raise ValueError("window_size must be >= 1")
            if kwargs["pad"] not in ("min", "percentile"):
                raise ValueError("pad must be 'min' or 'percentile'")

            self.dsp_modules[name] = {'algo': target_algo, **kwargs}
            self.dsp_modules_queue.append(name)
            return

        if target_algo == 'butterworth_filter':
            if not set(bw_keys).issubset(kwargs.keys()):
                raise ValueError(f"Missing required parameters: {bw_keys}")
            unexpected_params = set(kwargs.keys()) - set(bw_keys)
            if unexpected_params:
                raise ValueError(f"Invalid butterworth input, unexpected parameters: {unexpected_params}")

            fs = int(kwargs["fs"])
            min_frq = float(kwargs["min_frq"])
            max_frq = float(kwargs["max_frq"])
            order = int(kwargs["order"])

            if fs <= 0:
                raise ValueError("fs must be > 0")
            nyq = fs / 2.0
            if not (0.0 < min_frq < max_frq < nyq):
                raise ValueError(f"Butterworth requires 0 < min_frq < max_frq < fs/2. Got min={min_frq}, max={max_frq}, fs={fs}")
            if order < 1:
                raise ValueError("order must be >= 1")

            self.dsp_modules[name] = {'algo': target_algo, **kwargs}
            self.dsp_modules_queue.append(name)
            return

        raise ValueError(f"Unsupported module algorithm: {target_algo}")

    def remove_dsp_module(self, name: str):
        if name not in self.dsp_modules_queue:
            raise ValueError(f"Specified module does not exist: {name}")
        self.dsp_modules_queue.remove(name)
        del self.dsp_modules[name]

    def run_dsp_modules(self, data):
        new_data = np.copy(data)
        for name in self.dsp_modules_queue:
            module = self.dsp_modules[name]
            target_algo = module['algo']

            if target_algo == 'moving_average_filter':
                filtered_data = self.moving_average_filter(
                    new_data,
                    int(module['window_size']),
                    str(module['pad'])
                )

            elif target_algo == 'butterworth_filter':
                filtered_data = self.butterworth_filter(
                    new_data,
                    float(module['min_frq']),
                    float(module['max_frq']),
                    int(module['order']),
                    int(module['fs'])
                )
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
            raise ValueError("Invalid moving_average_filter pad; use 'min' or 'percentile'")

        cumsum = np.cumsum(data_arr)
        filtered = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
        return filtered

    @staticmethod
    def butterworth_filter(data, min_frq: float, max_frq: float, order: int, fs: int):
        b, a = signal.butter(order,
                             [min_frq, max_frq],
                             btype='bandpass',
                             analog=False,
                             output='ba',
                             fs=fs)
        return signal.filtfilt(b, a, data)