# Changelog

Notable changes to sensor_core are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/). Changes before 1.0.0 are not recorded.

## [Unreleased]

To be released as 2.0.0: the storage format and several interfaces changed, and databases written by
1.x cannot be read.

### Added

- Automated tests, run in CI on Linux, macOS, and Windows, including the example notebooks and the
  benchmark.
- `SensorManager.flush()`, `stop()`, and `close()`, and `with SensorManager(...) as sm:`, so a session can
  end with everything it acquired stored. Managers still open are closed when Python exits.
- Each stored frame keeps the time it was acquired, which can be loaded as monotonic or wall-clock time.
- `StorageManager.list_sessions()`, `load_frame_times()`, `load_frames()`, and `load_images()`, and a
  `session` argument for `load_serial_channel()`.
- `SensorManager.create_plot()`, which returns the live figure.
- The `writer_dropped_frames` metric: frames the writer could not copy before the ring buffer overwrote
  them.
- `SensorManager` arguments `stream_dir`, `ring_capacity`, `shm_name`, and `rotate_bytes`.
- A benchmark (`benchmarks/`) that measures throughput and checks every stored frame.
- An example script that runs without a display, `examples/virtual_serial_port_line.py`.
- Installation extras: `notebook` (Jupyter plotting), `test`, and `benchmark`.

### Changed

- Storage uses plain SQLite tables, `sessions` and `frames`, instead of `sqlitedict`. Each run is stored
  as its own session, appending no longer slows down as the database grows, and any SQLite client can
  read the data.
- Frames stream to numbered segment files instead of two alternating files, so data is never
  overwritten before it is stored, and segments left behind by a crash are stored in the next session.
- Each slot of the ring buffer holds one whole acquisition, so `ring_capacity` and `rotate_frames` count
  acquisitions.
- Each `SensorManager` has its own shared-memory ring buffer, so several can run at once, for example one
  per serial port.
- `scipy` is now a dependency; `jupyterlab` and `jupyter_rfb` moved to the `notebook` extra; `sqlitedict`
  is no longer used.
- The compiled extension is now part of the package, as `sensor_core._fastring`.
- The example notebooks were rewritten for the current interface.

### Deprecated

- `SensorManager.setup_plotting_process()`; use `create_plot()`.
- The `save_data` and `filepath` arguments of `update_data_process()`; storage is set up with
  `start_stream_ingest` and `sqlite_path` when creating the `SensorManager`.
- The `fast_stream_path_a` and `fast_stream_path_b` arguments of `SensorManager`, which are now ignored;
  use `stream_dir`.

### Removed

- The `sqlitedict` storage functions `load_serial_database()`, `append_serial_channel()`, `create_sqlite()`,
  and `load_sqlite()`.
- `requirements.txt`; dependencies are listed in `pyproject.toml`.

### Fixed

- Stored line data did not match the acquired data, and image frames written together were stored as
  copies of one another.
- Timestamps were not stored.
- The live line plot mixed samples from every channel into each subplot.
- Frames overwritten in the ring buffer before the writer copied them were skipped without being
  reported, and under heavy load frames could be stored damaged. They are now dropped and counted.
- Rotating stream files could discard data that had not been stored yet.
- On Windows, frames acquired before the storage processes started were lost.
- `SensorManager` failed with NumPy 2.4 and later.
- Plotting a subset of the channels rejected every acquisition.
- Errors raised inside a custom acquisition function were reported as a signature error.
- The built-in serial reader returned frames of the wrong shape.
- Arrays read from the ring buffer could outlive it and crash Python.
- Worker processes and shared memory were left behind after a crash or a notebook kernel restart.
- Plotting stored data failed.

## [1.0.0] - 2026-01-13

First release on PyPI, as `sensor-pipeline`.

[Unreleased]: https://github.com/BailabUNC/sensor_core/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/BailabUNC/sensor_core/releases/tag/v1.0.0
