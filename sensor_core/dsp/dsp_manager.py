import numpy as np
import scipy.signal as signal

#Moving average filter
def moving_avg_filter(data, N):
    data_arr = np.concatenate(([np.min(data)] * (N), data))
    cumsum = np.cumsum(data_arr)
    filtered = (cumsum[N:] - cumsum[:-N]) / N
    return filtered
    
#Butterworth filter
def butter_filter(data, min_frq, max_frq, N):
    b, a = signal.butter(N, [min_frq, max_frq], 'band', analog = False, output = 'ba', fs = 100)
    filtered = signal.filtfilt(b, a, data)
    return filtered