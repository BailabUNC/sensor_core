"""Shared fixtures for the sensor_core test suite.

Most tests run in a single process against small shared-memory rings that are
created and removed per test. Tests marked `processes` start worker processes.
"""
import os
import uuid

import numpy as np
import pytest

from harness import StoragePipeline, unlink_shared_memory


def pytest_report_header(config):
    import scipy
    import sensor_core

    return [
        f"numpy {np.__version__}, scipy {scipy.__version__}",
        f"sensor_core imported from {os.path.dirname(sensor_core.__file__)}",
    ]


@pytest.fixture
def shm_name():
    """A unique shared-memory name for one test, removed afterwards."""
    name = f"/sc_test_{uuid.uuid4().hex[:12]}"
    yield name
    unlink_shared_memory(name)


@pytest.fixture
def storage_pipeline(tmp_path, shm_name):
    """Factory for a StoragePipeline that is closed after the test."""
    pipelines = []

    def make(**options):
        pipeline = StoragePipeline(str(tmp_path), shm_name, **options)
        pipelines.append(pipeline)
        return pipeline

    yield make
    for pipeline in pipelines:
        pipeline.close()


@pytest.fixture
def data_manager(shm_name):
    """Factory for a DataManager on a virtual serial port, configured the way SensorManager does it."""
    from sensor_core.data import DataManager
    from sensor_core.memory.mem_utils import initialize_ring
    from sensor_core.utils.utils import create_static_dict

    held = []

    def make(ser_channel_key, frame_shape, plot_channel_key=None, data_mode="line", dtype=np.float32):
        ring, shape = initialize_ring(ser_channel_key, dtype, shm_name=shm_name, frames_capacity=16,
                                      data_mode=data_mode, frame_shape=frame_shape)
        static_args = create_static_dict(
            ser_channel_key=ser_channel_key,
            plot_channel_key=plot_channel_key or [ser_channel_key],
            commport=None,
            baudrate=115200,
            shm_name=shm_name,
            shape=shape,
            dtype=dtype,
            ring_capacity=16,
            data_mode=data_mode,
            frame_shape=shape,
        )
        manager = DataManager(static_args_dict=static_args, virtual_ser_port=True)
        held.extend([ring, manager])
        return manager

    yield make
    held.clear()
