#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "ring.hpp"
namespace py = pybind11;

PYBIND11_MODULE(_fastring, m) {
    py::class_<ShmRing>(m, "Ring")
        .def_static("create", [](const std::string& name, size_t cap, size_t fbytes) {
            return ShmRing::create(name.c_str(), cap, fbytes);
        })
        .def_static("open", [](const std::string& name, size_t cap, size_t fbytes) {
            return ShmRing::open(name.c_str(), cap, fbytes);
        })
        .def_static("unlink", [](const std::string& name) {
            ShmRing::unlink(name.c_str());
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
        // Read-only bytes of `frames` consecutive frames starting at logical index `start`, as a
        // uint8 array. The caller applies the dtype and shape, so every dtype works. The array's base
        // is this ring, so the ring stays mapped for as long as the array or any view of it exists.
        .def("view_bytes", [](py::object self, uint64_t start, size_t frames) {
            ShmRing& r = self.cast<ShmRing&>();
            size_t slot = (size_t)(start % r.capacity);
            if (frames > r.capacity - slot)
                throw std::runtime_error("window wraps the ring; split it into two calls");
            py::array bytes(py::dtype::of<uint8_t>(),
                            {(py::ssize_t)(frames * r.frame_bytes)},
                            {(py::ssize_t)1},
                            r.data + slot * r.frame_bytes,
                            self);
            bytes.attr("setflags")(py::arg("write") = false);
            return bytes;
        });
}