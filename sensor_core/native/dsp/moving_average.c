#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

double find_min(
    const double *data, 
    int n
) {
    double min_val = data[0];
    for (int i = 1; i < n; i++)
        if (data[i] < min_val)
            min_val = data[i];
    return min_val;
}

static int partition(double *arr, int left, int right, int pivotIndex) {
    double pivotValue = arr[pivotIndex];
    double tmp = arr[pivotIndex];
    arr[pivotIndex] = arr[right];
    arr[right] = tmp;

    int storeIndex = left;
    for (int i = left; i < right; i++) {
        if (arr[i] < pivotValue) {
            double t = arr[i];
            arr[i] = arr[storeIndex];
            arr[storeIndex] = t;
            storeIndex++;
        }
    }

    tmp = arr[right];
    arr[right] = arr[storeIndex];
    arr[storeIndex] = tmp;
    return storeIndex;
}

static double quickselect(double *arr, int left, int right, int k) {
    while (left < right) {
        int pivotIndex = left + (right - left) / 2;
        int newIndex = partition(arr, left, right, pivotIndex);

        if (newIndex == k)
            break;
        else if (newIndex > k)
            right = newIndex - 1;
        else
            left = newIndex + 1;
    }
    return arr[k];
}

double find_10th_percentile(const double *data, int n) {
    if (n <= 0) return NAN;

    double *copy = (double*) malloc(n * sizeof(double));
    if (!copy) return NAN;
    memcpy(copy, data, n * sizeof(double));

    int k = (int) floor(0.10 * (n - 1));
    double val = quickselect(copy, 0, n - 1, k);

    free(copy);
    return val;
}

double *moving_average_filter(
    const double *data,
    int n,
    int window_size,
    const char *pad,
    int *out_len
) {
    double pad_val;
    if (strcmp(pad, "percentile") == 0) {
        pad_val = find_10th_percentile(data, n);
    } else {
        pad_val = find_min(data, n);
    }

    int padded_len = n + window_size;
    double *padded_data = (double*) malloc(padded_len * sizeof(double));
    for (int i = 0; i < window_size; i++) {
        padded_data[i] = pad_val;
    }
    for (int i = 0; i < n; i++) {
        padded_data[i + window_size] = data[i];
    }

    double *filtered = (double*) malloc(n * sizeof(double));

    double cumsum = 0.0;
    for (int i = 0; i < window_size; i++) {
        cumsum += padded_data[i];
    }
    for (int i = window_size; i < padded_len; i++) {
        cumsum += padded_data[i] - padded_data[i - window_size];
        filtered[i - window_size] = cumsum / window_size;
    }

    free(padded_data);

    *out_len = n;
    return filtered;
}
