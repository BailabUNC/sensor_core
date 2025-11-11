from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# Define the extension module
extensions = [
    Extension(
        name="moving_average_cython",  # Module name
        sources=["moving_average_cython.pyx"],  # Source file
        include_dirs=[np.get_include()]# Suppress NumPy warnings
    )
]

setup(
    name="moving_average_cython",
    version="1.0.0",
    description="moving average filter implemented in Cython",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": 3,  # Use Python 3 syntax
            "boundscheck": False,  # Disable bounds checking for performance
            "wraparound": False,   # Disable negative index wrapping
        }
    ),
    zip_safe=False,
    install_requires=[
        "numpy",
        "Cython",
    ],
    python_requires=">=3.6",
)