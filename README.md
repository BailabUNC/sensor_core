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

## Storing and Loading Data
With `start_stream_ingest=True`, acquired frames are stored in the SQLite database at `sqlite_path`. Call `sm.flush()` to store everything acquired so far without stopping, and `sm.close()` when finished (or create the manager with `with SensorManager(...) as sm:`). Each run is stored as a session, and every frame keeps the time it was acquired:
```python
from sensor_core.memory.strg_manager import StorageManager

db = "serial_db.sqlite3"
sessions = StorageManager.list_sessions(db)                   # runs stored in this database
red = StorageManager.load_serial_channel("red", db, session=-1)  # every sample of one channel, latest run
times = StorageManager.load_frame_times(db, session=-1)          # acquisition time of each frame, ns since the Unix epoch
```
A line-mode frame is one acquisition of `window_size` samples, so each timestamp covers that many samples of every channel. Timestamps come from the host's monotonic clock, which all processes on the machine share, so streams recorded on the same computer can be aligned; `load_frame_times(..., clock="monotonic")` returns those values directly. The database is plain SQLite, readable from any language: a `sessions` table holds each run's settings and clock anchor, and a `frames` table holds one row per frame with its index, timestamp, and raw bytes. The format is described in [`strg_manager.py`](sensor_core/memory/strg_manager.py). For a complete script that acquires, stores, and checks data without a display, see [`examples/virtual_serial_port_line.py`](examples/virtual_serial_port_line.py).

## Benchmarks
[`benchmarks/`](benchmarks/README.md) measures how many frames per second are acquired, written to disk, stored in SQLite, and drawn by the live plot, for line and image data. It checks that every stored frame arrived intact, and it produces the performance figure in the paper; its README lists the exact commands.

## Running the Tests
```
pip install -e ".[test]"
pytest
```
The ([![tests](https://github.com/BailabUNC/sensor_core/actions/workflows/tests.yml/badge.svg)](https://github.com/BailabUNC/sensor_core/actions/workflows/tests.yml)) run on Linux, macOS, and Windows for every push and pull request. They cover acquisition, digital signal processing, the shared-memory ring buffer, storage, and the example script. Tests that render figures, including the example notebooks, run when `SENSOR_CORE_RENDER_TESTS=1` is set; they need the notebook extras and a GPU or a software renderer (on Linux, `sudo apt-get install mesa-vulkan-drivers`), and CI runs them on Linux. Tests marked `processes` start worker processes; skip them with `pytest -m "not processes"`.

## Acquiring, Plotting, and Saving Data in Real-Time
The following data was captured by [MABOS](https://github.com/BailabUNC/MABOS/tree/master): a proprietary biosensor we developed. 

https://github.com/BailabUNC/MABOS_core/assets/96029511/cbcf4896-62dc-4e1d-8ed4-9be6ac47196a

## Contributing
Questions, bug reports, and suggestions are welcome as [GitHub issues](https://github.com/BailabUNC/sensor_core/issues). [CONTRIBUTING.md](CONTRIBUTING.md) explains how to set up a development environment, run the tests, and propose changes, and [CHANGELOG.md](CHANGELOG.md) lists the changes in each release.

## Citing sensor_core
If you use sensor_core in your research, please cite it. The citation is in [CITATION.cff](CITATION.cff); on GitHub, "Cite this repository" in the sidebar gives it in APA and BibTeX formats.

