import numpy as np
import scipy.signal as signal

#Moving average filter
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
    
#Butterworth filter
def butter_filter(data, min_frq, max_frq, N, btype):
    if(btype != 'lowpass' or 'highpass' or 'bandpass' or 'bandstop'):
        raise ValueError("Specify btype 'lowpass', 'highpass', 'bandpass', or 'bandstop'")
    
    b, a = signal.butter(N, [min_frq, max_frq], btype, analog = False, output = 'ba', fs = 100)
    filtered = signal.filtfilt(b, a, data)
    return filtered