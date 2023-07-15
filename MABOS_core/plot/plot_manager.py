import MABOS_core.memory.mem_manager as mm
from ._plot_utils import *
from multiprocessing.shared_memory import SharedMemory


def initialize_plot(channel_key, num_points, grid_plot_flag):
    """ Handler for initializing Plot/GridPlot

    :param channel_key: user-defined channel name
    :param num_points: number of 'time' points [num_points = time(s) * Hz]
    :param grid_plot_flag: boolean, determines usage of Plot or GridPlot
    :return: Calls function to initialize Plot or GridPlot
    """

    if grid_plot_flag:
        grid_plot, data = _initialize_grid_plot(channel_key=channel_key, num_points=num_points)
        return grid_plot, data
    else:
        if len(channel_key) != 1:
            raise ValueError(f"Channel key {channel_key} must have only one element if using Plot")
        else:
            plot, data = _initialize_plot(channel_key=channel_key, num_points=num_points)
            return plot, data


def _initialize_plot(channel_key, num_points):
    """ Initialize Plot object and data

    :param channel_key: user-defined channel name
    :param num_points: number of 'time' points [num_points = time(s) * Hz]
    :return: Plot object and data
    """
    plot = create_plot(channel_key=channel_key)
    xs, ys = initialize_plot_data(num_points=num_points)
    plot_data = np.dstack([xs, ys])[0]
    plot.add_line(data=plot_data, name=channel_key[0], cmap='jet')
    plot.auto_scale(maintain_aspect=False)
    data = np.vstack((xs, ys))
    return plot, data


def _initialize_grid_plot(channel_key, num_points):
    """ Initialize GridPlot object and data

    :param channel_key: user-defined channel name
    :param num_points: number of 'time' points [num_points = time(s) * Hz]
    :return: GridPlot object and data
    """
    grid_plot = create_grid_plot(channel_key=channel_key)
    xs, ys = initialize_grid_plot_data(num_channel=len(channel_key), num_points=num_points)
    for i, subplot in enumerate(grid_plot):
        plot_data = np.dstack([xs, ys[i]])[0]
        subplot.add_line(data=plot_data, name=channel_key[i], cmap='jet')
    data = np.vstack((xs, ys))
    return grid_plot, data


def obtain_plot_data(args_dict):
    """ Update Plot data with shared memory object data

    :param args_dict: dictionary containing kwargs for memory and plot management
    :return: no explicit return. Updates Plot data, next render cycle will show updated data
    """
    # Unpack dictionary
    plot = args_dict["plot"]
    mutex = args_dict["mutex"]
    shm_name = args_dict["shm_name"]
    shape = args_dict["shape"]
    dtype = args_dict["dtype"]
    channel_key = args_dict["channel_key"]
    # Acquire Shared Memory Object data
    mm.acquire_mutex(mutex)
    shm = SharedMemory(shm_name)
    data_shared = np.ndarray(shape=shape, dtype=dtype,
                             buffer=shm.buf)
    data = np.dstack([data_shared[0], data_shared[1]])[0]
    plot[channel_key[0]].data = data
    plot.auto_scale(maintain_aspect=False)
    mm.release_mutex(mutex)


def obtain_grid_plot_data(args_dict):
    """ Update GridPlot data with shared memory object data

    :param args_dict: dictionary containing kwargs for memory and plot management
    :return: no explicit return. Updates GridPlot data, next render cycle will show updated data
    """
    # Unpack dictionary
    grid_plot = args_dict["plot"]
    mutex = args_dict["mutex"]
    shm_name = args_dict["shm_name"]
    shape = args_dict["shape"]
    dtype = args_dict["dtype"]
    channel_key = args_dict["channel_key"]
    # Acquire Shared Memory Object data
    mm.acquire_mutex(mutex)
    shm = SharedMemory(shm_name)
    data_shared = np.ndarray(shape=shape, dtype=dtype,
                             buffer=shm.buf)
    for i, subplot in enumerate(grid_plot):
        data = np.dstack([data_shared[0], data_shared[i + 1]])[0]
        subplot[channel_key[i]].data = data
        subplot.auto_scale(maintain_aspect=False)
    mm.release_mutex(mutex)
