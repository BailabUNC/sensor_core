"""Live plotting: each subplot must show its own channel's most recent samples.

Rendering needs a GPU, so these tests check the data the plotter takes from
the ring (latest_line_traces), not the figure itself.
"""
import numpy as np
import pytest

from harness import line_acquisitions
from sensor_core.memory.mem_utils import initialize_ring
from sensor_core.plot.plot_utils import latest_line_traces

KEYS = ["red", "infrared", "violet"]
LINE_SHAPE = (100, 10, 3)  # 100 points per trace, 10 samples per acquisition, 3 channels
LAG = 16  # the plotter's default: skip the newest 16 frames


@pytest.fixture
def ring(shm_name):
    ring, _ = initialize_ring(KEYS, np.float32, shm_name=shm_name, frames_capacity=4096,
                              data_mode="line", frame_shape=LINE_SHAPE)
    return ring


@pytest.mark.parametrize("plot_channel_key", [
    [["red", "infrared", "violet"]],
    [["violet", "red", "infrared"]],
    [["red"], ["violet"]],
])
def test_each_subplot_shows_its_own_channels_latest_samples(ring, plot_channel_key):
    acquisitions = line_acquisitions(40, window=10, channels=3)
    for acquisition in acquisitions:
        ring.publish(acquisition)
    traces = latest_line_traces(ring, KEYS, plot_channel_key, num_points=100, lag=LAG)
    assert set(traces) == {key for row in plot_channel_key for key in row}
    for key, trace in traces.items():
        channel = KEYS.index(key)
        assert len(trace) == 100
        # ends at the newest frame outside the lag, and every sample is the next one from this channel
        assert trace[-1] == acquisitions[-1 - LAG, -1, channel]
        assert (np.diff(trace) == len(KEYS)).all(), f"the {key} subplot's samples are not consecutive"


def test_no_traces_until_enough_frames_arrive(ring):
    for acquisition in line_acquisitions(5, window=10, channels=3):
        ring.publish(acquisition)
    assert latest_line_traces(ring, KEYS, [KEYS], num_points=100, lag=LAG) is None
