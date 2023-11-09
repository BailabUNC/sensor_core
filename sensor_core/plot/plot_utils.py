import numpy as np
from typing import *
from fastplotlib import Plot, GridPlot
from sensor_core.memory.strg_manager import StorageManager


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


def create_grid_plot(plot_channel_key: Union[np.ndarray, str]):
    """ Create fastplotlib GridPlot (collection of subplots)

    :param plot_channel_key: names of subplots
    :return: GridPlot object
    """
    grid_shape = np.shape(plot_channel_key)

    grid_plot = GridPlot(
        shape=grid_shape,
        names=plot_channel_key
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


def offline_initialize_data(filepath: str, channel_key: Union[np.ndarray, str]):
    """ Extract offline sensor data for set of keys
    :param filepath: define path to database to read data from
    :param channel_key: define set of keys in database to plot data
    :return: x and y values
    """
    ys = []
    for key in channel_key:
        data = StorageManager.load_serial_channel(key=key, filepath=filepath)
        ys.append(data)

    num_points = len(ys[0])
    xs = [np.linspace(0, num_points - 1, num_points)]
    return xs, ys


def offline_plot_data(filepath: str, channel_key: Union[np.ndarray, str] = None):
    """ Initialize plot for offline data
    :param filepath: define path to database to read data from
    :param channel_key: define set of keys in databaes to plot data
    :return: return plot object 
    """
    if channel_key is None:
        database = StorageManager.load_serial_database(filepath=filepath)
        channel_key = []
        with database as db:
            for key in db.keys():
                channel_key.append(key)
        channel_keys = list(np.transpose(channel_key))
    else:
        channel_keys = channel_key

    xs, ys = offline_initialize_data(filepath, channel_key=channel_keys)
    plot = create_grid_plot(channel_key=channel_keys)

    for i, subplot in enumerate(plot):
        data = np.dstack([xs, ys[i]])[0]
        subplot.add_line(data=data, name=channel_keys[i], cmap='jet')
        subplot.auto_scale(maintain_aspect=False)

    plot.show()

    return plot
