# Contributing to sensor_core

Thank you for helping improve sensor_core. This guide covers how to get help, report problems, and
contribute changes.

## Getting help and reporting problems

Questions, bug reports, and feature requests all go in
[GitHub issues](https://github.com/BailabUNC/sensor_core/issues). For a bug, please include:

- what you ran: a short script or notebook cell, ideally using a simulated sensor as in the
  [examples](examples/) so others can run it without your hardware
- what happened, with the full error message
- the output of `sm.get_metrics()`, if a SensorManager was running
- your operating system, Python version, and sensor_core version (`pip show sensor-pipeline`)

## Development setup

```
git clone https://github.com/BailabUNC/sensor_core
cd sensor_core
pip install -e ".[notebook,test,benchmark]"
```

The shared-memory ring buffer is a C++ extension, so installing from source needs a C++17 compiler:
Visual Studio Build Tools on Windows, the Xcode Command Line Tools on macOS, or GCC or Clang on Linux.
Python changes take effect immediately; after changing `sensor_core/native/fastring/`, rerun the install
command to rebuild the extension.

How the pieces fit together, following the data:

- `sensor_core/serial/` and `sensor_core/data/`: acquisition. Your acquisition function runs in its own
  process (a thread on Windows) and publishes each frame to the ring buffer.
- `sensor_core/native/fastring/` and `sensor_core/memory/ring_adapter.py`: the ring buffer in shared memory.
- `sensor_core/memory/stream_logger.py`: the writer, which copies frames from the ring buffer into
  numbered segment files.
- `sensor_core/memory/db_ingester.py` and `strg_manager.py`: the ingester, which stores segments in
  SQLite, and the storage format and loading functions.
- `sensor_core/plot/` and `sensor_core/dsp/`: live plotting and filtering.
- `sensor_core/sensor_manager.py`: `SensorManager`, which starts and stops all of the above.

## Running the tests

```
pytest
```

- `pytest -m "not processes"` skips the tests that start worker processes, for a quicker run.
- Tests that render figures, including the example notebooks, run when `SENSOR_CORE_RENDER_TESTS=1` is
  set. They need the notebook extras and a GPU or a software renderer (on Linux,
  `sudo apt-get install mesa-vulkan-drivers`).
- The benchmark has its own instructions in [benchmarks/README.md](benchmarks/README.md).

CI runs the tests for every push and pull request: on Linux, macOS, and Windows with Python 3.10
(and NumPy 1.26) and Python 3.14; against a wheel built and installed outside the source tree; and, on
Linux with a software renderer, the notebooks, plots, and benchmark. CI cannot test real serial hardware
or rendering on macOS and Windows, so please check those by hand if your change affects them.

## Submitting changes

1. For a large change, open an issue first so we can agree on the approach.
2. Create a branch from `main`.
3. Add or update tests. A bug fix should come with a test that fails without the fix.
4. Run `pytest` and make sure it passes.
5. For any change users will notice, add a line under "Unreleased" in [CHANGELOG.md](CHANGELOG.md).
6. Open a pull request that says what changed and why, and link the issue it addresses. CI must pass
   before it is merged.

## Releasing (maintainers)

1. Update `version` in `pyproject.toml`, and `version` and `date-released` in `CITATION.cff`.
2. In `CHANGELOG.md`, rename "Unreleased" to the new version and date.
3. Merge to `main`, then push a version tag, for example `git tag v2.0.0 && git push origin v2.0.0`.
   The tag runs the workflow that builds wheels for Linux, macOS, and Windows and publishes them to PyPI.
4. Create a GitHub release from the tag. With the repository's Zenodo integration turned on, this also
   archives the release and gives it a DOI.
