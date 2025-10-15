#include <array>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "ring.hpp"
namespace py = pybind11;

PYBIND11_MODULE(fastring, m) {
    py::class_<ShmRing>(m, "Ring")
        .def_static("create", [](const std::string& name, size_t cap, size_t fbytes) {
            return ShmRing::create(name.c_str(), cap, fbytes);
        })
        .def_static("open", [](const std::string& name, size_t cap, size_t fbytes) {
            return ShmRing::open(name.c_str(), cap, fbytes);
        })
        .def_property_readonly("frame_bytes", [](const ShmRing& r){ return r.frame_bytes; })
        .def_property_readonly("capacity", [](const ShmRing& r){ return r.capacity; })
        .def_property_readonly("write_idx", [](const ShmRing& r) {
            return (uint64_t) r.hdr->write_idx.load(std::memory_order_acquire);
        })
        .def("publish", [](ShmRing& r, py::array arr) {
            py::gil_scoped_release release;
            if (!(arr.flags() & py::array::c_style))
                throw std::runtime_error("array must be C-contiguous");
            size_t nbytes = (size_t)arr.nbytes();
            if (nbytes % r.frame_bytes != 0)
                throw std::runtime_error("size not multiple of frame_bytes");
            r.publish(arr.data(), nbytes / r.frame_bytes);
        })
        .def("view_frame", [](ShmRing& r, uint64_t logical_idx, py::ssize_t C, py::ssize_t S) {
            size_t slot = (size_t)(logical_idx % r.capacity);
            void* ptr = r.data + slot * r.frame_bytes;
            const py::ssize_t itemsize = (py::ssize_t)sizeof(float);
            const char* fmt = "f";
            std::array<py::ssize_t, 2> shape   { C, S };
            std::array<py::ssize_t, 2> strides { (py::ssize_t)(S * sizeof(float)), (py::ssize_t)sizeof(float) };
            return py::memoryview::from_buffer(ptr, itemsize, fmt, shape, strides, /*readonly=*/true);
        })
        .def("view_window", [](ShmRing& r, uint64_t start, size_t frames, py::ssize_t C, py::ssize_t S) {
            size_t slot = (size_t)(start % r.capacity);
            if (slot + frames > r.capacity)
                throw std::runtime_error("window wraps ring; split into two calls");
            void* ptr = r.data + slot * r.frame_bytes;
            const py::ssize_t itemsize = (py::ssize_t)sizeof(float);
            const char* fmt = "f";
            std::array<py::ssize_t, 3> shape   { (py::ssize_t)frames, C, S };
            std::array<py::ssize_t, 3> strides {
                (py::ssize_t)r.frame_bytes,
                (py::ssize_t)(S * sizeof(float)),
                (py::ssize_t)sizeof(float)
            };
            return py::memoryview::from_buffer(ptr, itemsize, fmt, shape, strides, /*readonly=*/true);
        });
}
