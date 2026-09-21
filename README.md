# sensor_core
A Python-based package for the acquisition, digital signal processing, plotting, and storage of sensor data in realtime.
*Please see [fastplotlib](https://github.com/kushalkolar/fastplotlib/tree/), developed by Kushal Kolar, to learn more about the plotting library we primarily use.*

# Key Features
1.) **Custom Serial Acquisition** - Users can write and pass their own acquisition handler into the sensor_core pipeline. Refer to the [custom serial handler notebook](examples/custom_serial_acquisition.ipynb).

2.) **Digital Signal Processing Integration** - sensor_core has a DSPManager hook enabling custom or predefined DSP algorithms to be applied prior to visualization without affecting the underlying datastream. Refer to the [DSP notebok](examples/virtual_serial_port_line_dsp.ipynb).

3.) **High-Speed Visualization** - using fastplotlib, we can reliably visualize 2- and 3-D data at high speed. We have thus far tested only in Jupyter Notebooks. Refer to the [line](examples/virtual_serial_port_line.ipynb) and [image](examples/virtual_serial_port_image.ipynb) notebooks for visualization examples.

4.) **High-Bandwidth Storage** - sensor_core creates temporary .bin files to stream data rapidly to before offloading to a sqlite file, enabling stable long term storage while imposing minimal delay in the real-time processing pipeline. Refer to the [line](examples/virtual_serial_port_line.ipynb) and [image](examples/virtual_serial_port_image.ipynb) notebooks for storage examples.
## Installation
sensor_core requires Python 3.10 or newer. It is published on PyPI as `sensor-pipeline` and imported as `sensor_core`:
```
pip install sensor-pipeline
```
The example notebooks plot live data in Jupyter, which requires the notebook extras:
```
pip install "sensor-pipeline[notebook]"
```
Prebuilt wheels cover Python 3.10–3.13 on Linux (x86_64), macOS (Apple silicon), and Windows (x64). On other platforms, pip builds from source, which requires a C++17 compiler.

## Developer Installation
```
git clone https://github.com/BailabUNC/sensor_core
cd sensor_core
pip install -e ".[notebook]"
```
Python changes take effect immediately. After editing the C++ sources in `sensor_core/native/fastring/`, rerun the install command to rebuild the extension.

## Acquiring, Plotting, and Saving Data in Real-Time
The following data was captured by [MABOS](https://github.com/BailabUNC/MABOS/tree/master): a proprietary biosensor we developed. 

https://github.com/BailabUNC/MABOS_core/assets/96029511/cbcf4896-62dc-4e1d-8ed4-9be6ac47196a