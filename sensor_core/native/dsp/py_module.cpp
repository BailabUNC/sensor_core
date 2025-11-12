#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
extern "C" {
#include "moving_average.c"
}

namespace py = pybind11;

py::array_t<double> moving_average_filter_py(
    py::array_t<double, py::array::c_style | py::array::forcecast> data,
    int window_size,
    std::string pad
) {
    py::buffer_info buf = data.request();
    int n = static_cast<int>(buf.size);
    const double* data_ptr = static_cast<double*>(buf.ptr);

    int out_len = 0;
    double* result = moving_average_filter(data_ptr, n, window_size, pad.c_str(), &out_len);

    py::array_t<double> output(out_len);
    py::buffer_info out_buf = output.request();
    memcpy(out_buf.ptr, result, out_len * sizeof(double));
    free(result);
    return output;
}

PYBIND11_MODULE(dsp, m) {
    m.def("moving_average_filter", &moving_average_filter_py,
        py::arg("data"), py::arg("window_size"), py::arg("pad"),
        R"pbdoc(
            Apply moving average filter with choice of padding (percentile or minimum)
            :param data: input signal
            :param window_size: window size for moving average
            :param pad: padding method ('percentile' or 'min')
            :return: filtered signal
        )pbdoc");
}