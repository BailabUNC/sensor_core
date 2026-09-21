import numpy as np
from typing import *

def _fastplotlib():
    try:
        import fastplotlib as fpl
        return fpl
    except ImportError as e:
        raise ImportError(
            "Plotting backend could not be initialized. "
            "If you are running headless, install a supported backend (e.g., `pip install glfw` "
            "or `pip install pyside6`) and ensure a display is available."
        ) from e

def create_fig(plot_channel_key: Union[np.ndarray, str]):
    """ Create fastplotlib Figure (collection of subplot(s))

    :param plot_channel_key: names of subplot(s)
    :return: GridPlot object
    """
    fpl = _fastplotlib()
    grid_shape = np.shape(plot_channel_key)

    fig = fpl.Figure(
        shape=grid_shape,
        names=plot_channel_key
    )
    return fig


def initialize_fig_data(num_channel: int, num_points: int):
    """ Initialize GridPlot data

    :param num_channel: number of distinct channels
    :param num_points: number of 'time' points [num_points = time(s) * Hz]
    :return: x and y arrays [y array has shape (num_channel, num_points)]
    """
    ys = np.ones((num_channel, num_points)) * np.linspace(0, 1, num_points)
    return ys


def latest_line_samples(ring, num_points: int, lag: int = 0):
    """ Most recent samples of every channel, oldest first

    :param ring: line-mode RingBuffer, whose frames are shaped (window_size, channels)
    :param num_points: number of samples per channel to return
    :param lag: number of the newest frames to skip
    :return: array shaped (num_points, channels), or None until enough frames have been published
    """
    window, channels = ring.slot_shape
    frames = min(-(-int(num_points) // window), ring.capacity)
    start = int(ring.write_idx) - int(lag) - frames
    if start < 0:
        return None
    samples = ring.read_window(start, frames).reshape(-1, channels)
    return samples[-int(num_points):]


def latest_line_traces(ring, ser_channel_key, plot_channel_key, num_points: int, lag: int = 0):
    """ Most recent samples of each plotted channel, keyed by channel name

    :param ring: line-mode RingBuffer
    :param ser_channel_key: names of the channels in the ring, in order
    :param plot_channel_key: grid of channel names, one per subplot
    :param num_points: number of samples per trace
    :param lag: number of the newest frames to skip
    :return: dict mapping each plotted channel name to its samples, or None until enough frames have been published
    """
    samples = latest_line_samples(ring, num_points, lag)
    if samples is None:
        return None
    column = {key: i for i, key in enumerate(ser_channel_key)}
    return {key: samples[:, column[key]] for row in plot_channel_key for key in row}
