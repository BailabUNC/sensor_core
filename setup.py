from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext
import numpy as np

install_requires = [
    "numpy",
    "pygfx>=0.1.13",
    "jupyterlab",
    "pyserial",
    "fastplotlib",
    "h5py",
]

ext_modules = [
    Pybind11Extension(
        "fastring",
        sources=[
            "sensor_core/native/fastring/py_module.cpp",
        ],
        include_dirs=[
            "sensor_core/native/fastring",
            np.get_include(),
        ],
        cxx_std=17,
        define_macros=[("PYBIND11_DETAILED_ERROR_MESSAGES", "1")],
        extra_compile_args=["-O3", "-fvisibility=hidden", "-fPIC"],
    ),
]

setup(
    name="sensor_core",
    version="1.0.0",
    packages=find_packages(),
    install_requires=install_requires,
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    description="Core API for plotting sensor data in real-time",
)
