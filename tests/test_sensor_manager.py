"""SensorManager configuration and process start-up."""
import time

import pytest

from harness import unlink_shared_memory
from sensor_core import SensorManager

KEYS = ["red", "infrared", "violet"]


def test_plot_keys_default_to_one_row_with_every_channel():
    ser_keys, plot_keys = SensorManager.setup_channel_keys(KEYS)
    assert ser_keys == KEYS
    assert plot_keys == [KEYS]


def test_plot_keys_can_arrange_channels_in_a_grid():
    grid = [["violet"], ["red"]]
    assert SensorManager.setup_channel_keys(KEYS, plot_channel_key=grid)[1] == grid


def test_plot_keys_must_be_serial_keys():
    with pytest.raises(KeyError):
        SensorManager.setup_channel_keys(KEYS, plot_channel_key=[["red", "green"]])


def test_serial_keys_must_be_one_dimensional():
    with pytest.raises(ValueError, match="one-dimensional"):
        SensorManager.setup_channel_keys([["red", "infrared"]])


@pytest.mark.processes
def test_sensor_manager_starts_its_stream_writer(tmp_path, monkeypatch):
    # SensorManager always uses this shared-memory name and never removes it (#57),
    # so clear any leftover object first; macOS cannot resize an existing one.
    unlink_shared_memory("/sensor_ring")
    monkeypatch.chdir(tmp_path)  # the writer creates ./serial_stream_a.bin and ./serial_stream_b.bin
    manager = SensorManager(ser_channel_key=KEYS, commport=None, frame_shape=(100, 10, 3))
    try:
        # The constructor only prints a message if the writer fails to start.
        assert manager._stream_proc is not None
        deadline = time.monotonic() + 30
        while not manager.writer_metrics_proxy.get("writer_active_bin") and time.monotonic() < deadline:
            assert manager._stream_proc.is_alive(), manager.writer_metrics_proxy.get("writer_last_error")
            time.sleep(0.05)
        assert manager.writer_metrics_proxy.get("writer_alive") is True
        assert manager.writer_metrics_proxy.get("writer_active_bin").startswith(str(tmp_path))
    finally:
        if manager._stream_proc is not None:
            manager._stream_proc.terminate()
            manager._stream_proc.join(timeout=10)
        manager._mp_manager.shutdown()
        manager.ring = None
        unlink_shared_memory("/sensor_ring")
