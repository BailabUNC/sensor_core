# save as moving_average_cython.pyx
import numpy as np
cimport numpy as cnp
cimport cython

cnp.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cython_moving_average_filter_trailingrolling(data, int window_size, str pad):
    cdef cnp.ndarray[cnp.double_t, ndim=1] data_arr = np.array(data, dtype=np.float64)

    cdef Py_ssize_t n = data_arr.shape[0]

    if n == 0:
        raise ValueError("Input array is empty")

    cdef double pad_value = data_arr[0]
    cdef int i

    if pad == 'min':
        for i in range(1, n):
            if data_arr[i] < pad_value:
                pad_value = data_arr[i]
    elif pad == 'percentile':
        pad_value = np.percentile(data_arr, 10)
    else:
        raise ValueError("Invalid pad parameter")

    cdef int w = window_size

    cdef cnp.ndarray[cnp.double_t, ndim=1] result = np.empty(n, dtype=np.float64)
    cdef double window_sum = 0.0
    cdef Py_ssize_t j

    for j in range(w - 1):
        window_sum += pad_value
    window_sum += data_arr[0]
    result[0] = window_sum / w

    for j in range(1, n):
        if j - w >= 0:
            window_sum -= data_arr[j - w]
        else:
            window_sum -= pad_value
        window_sum += data_arr[j]
        result[j] = window_sum / w

    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cython_moving_average_filter_centeredrolling(data, int window_size, str pad):
    cdef cnp.ndarray[cnp.double_t, ndim=1] data_arr = np.array(data, dtype=np.float64)

    cdef Py_ssize_t n = data_arr.shape[0]

    if n == 0:
        raise ValueError("Input array is empty")

    cdef double pad_value = data_arr[0]
    cdef int i

    if pad == 'min':
        for i in range(1, n):
            if data_arr[i] < pad_value:
                pad_value = data_arr[i]
    elif pad == 'percentile':
        pad_value = np.percentile(data_arr, 10)
    else:
        raise ValueError("Invalid pad parameter")

    cdef int w = window_size
    cdef int half_w = w // 2

    cdef cnp.ndarray[cnp.double_t, ndim=1] result = np.empty(n, dtype=np.float64)
    cdef double window_sum = 0.0
    cdef Py_ssize_t j
    cdef int left_idx, right_idx

    # Initialize the window sum for position 0
    # Window for position 0 spans from (-half_w) to (+half_w)
    for j in range(-half_w, half_w + 1):
        if j < 0 or j >= n:
            window_sum += pad_value
        else:
            window_sum += data_arr[j]

    result[0] = window_sum / w

    # Slide the window for positions 1 through n-1
    for i in range(1, n):
        # Remove the leftmost element of the previous window
        left_idx = (i - 1) - half_w  # leftmost index of previous window
        if left_idx < 0 or left_idx >= n:
            window_sum -= pad_value
        else:
            window_sum -= data_arr[left_idx]

        # Add the rightmost element of the current window
        right_idx = i + half_w  # rightmost index of current window
        if right_idx < 0 or right_idx >= n:
            window_sum += pad_value
        else:
            window_sum += data_arr[right_idx]

        result[i] = window_sum / w

    return result


@staticmethod
def moving_average_filter(cnp.ndarray[cnp.float64_t, ndim=1] data,
                          int window_size,
                          str pad):
    """
    Apply moving average filter with padding.

    Parameters:
    -----------
    data : 1D numpy array (float64)
    window_size : int
    pad : str, either 'min' or 'percentile'

    Returns:
    --------
    filtered : 1D numpy array with moving average applied
    """
    cdef:
        cnp.ndarray[cnp.float64_t, ndim=1] data_arr
        cnp.ndarray[cnp.float64_t, ndim=1] cumsum
        cnp.ndarray[cnp.float64_t, ndim=1] filtered
        double pad_value
        int n = data.shape[0]
        int total_size = n + window_size

    # Determine padding value
    if pad == 'min':
        pad_value = np.min(data)
    elif pad == 'percentile':
        pad_value = np.percentile(data, 10)
    else:
        raise ValueError("Invalid moving_average_filter parameter, remove module and specify 'min' or 'percentile' to pad array with")

    # Create padded array
    data_arr = np.empty(total_size, dtype=np.float64)
    data_arr[:window_size] = pad_value
    data_arr[window_size:] = data

    # Compute cumulative sum
    cumsum = np.cumsum(data_arr)

    # Compute moving average
    filtered = (cumsum[window_size:] - cumsum[:-window_size]) / window_size

    return filtered