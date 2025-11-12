# Moving Average Filter Comparison: Python, Numba, and Cython
import numpy as np
import time
from numba import jit
import matplotlib.pyplot as plt

class PythonFilter:
    @staticmethod
    def moving_average_filter(data, window_size: int, pad: str):
        if pad == 'min':
            data_arr = np.concatenate(([np.min(data)] * window_size, data))
        elif pad == 'percentile':
            data_arr = np.concatenate(([np.percentile(data, 10)] * window_size, data))
        else:
            raise ValueError("Invalid moving_average_filter parameter, remove module and specify 'min' or 'percentile' to pad array with")
        
        cumsum = np.cumsum(data_arr)
        filtered = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
        return filtered


@jit(nopython=True, cache = False)
def numba_moving_average_filter(data, window_size, pad_value):
    """
    pad_value must be pre-computed since numba doesn't support np.min() or np.percentile() in nopython.
    """
    n = len(data)
    # create padded array
    data_arr = np.empty(n + window_size)
    
    # fill padding
    for i in range(window_size):
        data_arr[i] = pad_value

    for i in range(n):
        data_arr[i + window_size] = data[i]
    
    # compute cumsum
    cumsum = np.empty_like(data_arr)
    cumsum[0] = data_arr[0]
    for i in range(1, len(data_arr)):
        cumsum[i] = cumsum[i-1] + data_arr[i]
    
    # compute moving average
    filtered = np.empty(n)
    for i in range(n):
        filtered[i] = (cumsum[i + window_size] - cumsum[i]) / window_size
    
    return filtered

class NumbaFilter:
    @staticmethod
    def moving_average_filter(data, window_size: int, pad: str):
        if pad == 'min':
            pad_value = np.min(data)
        elif pad == 'percentile':
            pad_value = np.percentile(data, 10)
        else:
            raise ValueError("Invalid moving_average_filter parameter, remove module and specify 'min' or 'percentile' to pad array with")
        
        return numba_moving_average_filter(data, window_size, pad_value)
    
class CythonStyleFilter:
    @staticmethod
    def moving_average_filter(data, window_size: int, pad: str):
        """
        Cython-optimized style implementation using numpy arrays efficiently.
        In actual Cython, this would use typed memoryviews and C loops.
        """
        
        data = np.array(data)
        
        n = len(data)
        
        # Compute pad value
        if pad == 'min':
            pad_value = data.min()
        elif pad == 'percentile':
            pad_value = np.percentile(data, 10)
        else:
            raise ValueError("Invalid moving_average_filter parameter, remove module and specify 'min' or 'percentile' to pad array with")
        
        # Pre-allocate arrays for better performance
        data_arr = np.empty(n + window_size, dtype=data.dtype)
        data_arr[:window_size] = pad_value
        data_arr[window_size:] = data
        
        # Use numpy's optimized cumsum
        cumsum = np.cumsum(data_arr)
        
        # Vectorized computation
        filtered = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
        
        return filtered
