"""Live plotting: each subplot must show its own channel's most recent samples.

Rendering needs a GPU, so these tests check the data the plotter extracts
from the ring, not the figure itself.
"""
import numpy as np
import pytest

from harness import line_acquisitions, plotted_line_traces
from sensor_core.memory.mem_utils import initialize_ring

KEYS = ["red", "infrared", "violet"]
LINE_SHAPE = (100, 10, 3)


@pytest.mark.xfail(strict=True, reason="#57: the plotter reads each trace from interleaved ring slots, so "
                                       "every subplot mixes samples from all channels")
@pytest.mark.parametrize("plot_channel_key", [
    [["red", "infrared", "violet"]],
    [["violet", "red", "infrared"]],
])
def test_each_subplot_shows_its_own_channel_in_order(shm_name, plot_channel_key):
    ring, shape = initialize_ring(KEYS, np.float32, shm_name=shm_name, frames_capacity=4096,
                                  data_mode="line", frame_shape=LINE_SHAPE)
    acquisitions = line_acquisitions(40, window=10, channels=3)
    for acquisition in acquisitions:
        ring.publish(acquisition)
    traces = plotted_line_traces(ring, shape, plot_channel_key)
    for key, trace in traces.items():
        channel_samples = acquisitions[:, :, KEYS.index(key)].ravel()
        assert np.isin(trace, channel_samples).all(), f"the {key} subplot shows other channels' samples"
        assert (np.diff(trace) == len(KEYS)).all(), f"the {key} subplot's samples are not consecutive"
