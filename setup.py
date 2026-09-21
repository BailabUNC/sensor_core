import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

if sys.platform == "win32":
    extra_compile_args = ["/O2"]
else:
    extra_compile_args = ["-O3", "-fvisibility=hidden", "-fPIC"]

ext_modules = [
    Pybind11Extension(
        "sensor_core._fastring",
        sources=["sensor_core/native/fastring/py_module.cpp"],
        include_dirs=["sensor_core/native/fastring"],
        depends=["sensor_core/native/fastring/ring.hpp"],
        cxx_std=17,
        define_macros=[("PYBIND11_DETAILED_ERROR_MESSAGES", "1")],
        extra_compile_args=extra_compile_args,
    ),
]

setup(ext_modules=ext_modules, cmdclass={"build_ext": build_ext})