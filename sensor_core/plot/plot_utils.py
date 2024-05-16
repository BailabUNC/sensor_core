import numpy as np
from typing import *
import fastplotlib as fpl


def create_fig(plot_channel_key: Union[np.ndarray, str]):
    """ Create fastplotlib Figure (collection of subplot(s))

    :param plot_channel_key: names of subplot(s)
    :return: GridPlot object
    """
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
    xs = [np.linspace(0, num_points - 1, num_points)]
    ys = np.ones((num_channel, num_points)) * np.linspace(0, 1, num_points)
    return xs, ys
