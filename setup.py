from setuptools import setup, find_packages

install_requires = [
    'numpy >= 2.0',
    'pygfx>=0.1.13',
    'jupyterlab',
    'pyserial',
    'fastplotlib',
    'h5py',
    ]

setup(
    name='MABOS_core',
    version='1.0',
    packages=find_packages(),
    install_requires=install_requires,
    url='',
    license='',
    author='Arjun Putcha',
    author_email='',
    description='Core API for plotting sensor data in real-time'
)
