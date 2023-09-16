import numpy as np
from typing import *
from fastplotlib import Plot, GridPlot


def create_plot(channel_key: Union[np.ndarray, str]):
    """ Create fastplotlib Plot

    :param channel_key: name of plot (should only have one element)
    :return: Plot object
    """
    if len(channel_key) > 1:
        raise ValueError(f"Channel Key {channel_key} should only have one name")

    plot = Plot(
        name=channel_key[0]
    )
    return plot


def create_grid_plot(channel_key: Union[np.ndarray, str]):
    """ Create fastplotlib GridPlot (collection of subplots)

    :param channel_key: names of subplots
    :return: GridPlot object
    """
    grid_shape = (len(channel_key), 1)
    names = np.transpose([channel_key]).tolist()

    grid_plot = GridPlot(
        shape=grid_shape,
        names=names
    )
    return grid_plot


def initialize_plot_data(num_points: int):
    """ Initialize Plot data

    :param num_points: number of 'time' points [num_points = time(s) * Hz]
    :return: x and y arrays
    """
    xs = [np.linspace(0, num_points - 1, num_points)]
    ys = np.ones(num_points) * np.linspace(0, 1, num_points)
    return xs, ys


def initialize_grid_plot_data(num_channel: int, num_points: int):
    """ Initialize GridPlot data

    :param num_channel: number of distinct channels
    :param num_points: number of 'time' points [num_points = time(s) * Hz]
    :return: x and y arrays [y array has shape (num_channel, num_points)]
    """
    xs = [np.linspace(0, num_points - 1, num_points)]
    ys = np.ones((num_channel, num_points)) * np.linspace(0, 1, num_points)
    return xs, ys
