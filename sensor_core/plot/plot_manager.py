import sensor_core.memory.mem_utils as mm
from sensor_core.memory.strg_manager import StorageManager
from .plot_utils import *
from typing import *
from multiprocessing.shared_memory import SharedMemory
from sensor_core.utils.utils import DictManager


class PlotManager(DictManager):
    def __init__(self, static_args_dict):
        """ Initialize Plot Manager Class
        :param static_args_dict: dictionary containing key parameters for initialization
        """
        DictManager.__init__(self)

        self.select_dictionary(args_dict=static_args_dict,
                               dict_type="static")
        # Unpack static_args_dict
        self.unpack_selected_dict()
        self.fig = None

        # Initialize figure
        self.initialize_fig()

        # Add animation function
        self.fig.add_animations(self.online_plot_data)


    def initialize_fig(self):
        """ Initialize Figure object and data
        :return: Figure object and data
        """
        fig = create_fig(plot_channel_key=self.plot_channel_key)
        xs, ys = initialize_fig_data(num_channel=self.num_channel, num_points=self.num_points)
        for i, subplot in enumerate(fig):
            idx = divmod(i, np.shape(self.plot_channel_key)[1])
            plot_data = np.dstack([xs, ys[i]])[0]
            subplot.add_line(data=plot_data, name=self.plot_channel_key[idx[0]][idx[1]], cmap='jet')
        self.fig = fig
        return fig


    def online_plot_data(self):
        """ Update subplot data with shared memory object data
        :return: no explicit return. Updates GridPlot data, next render cycle will show updated data
        """
        # Acquire Shared Memory Object data
        mm.acquire_mutex(self.mutex)
        shm = SharedMemory(self.shm_name)
        data_shared = np.ndarray(shape=self.shape, dtype=self.dtype,
                                 buffer=shm.buf)
        for i, subplot in enumerate(self.fig):
            idx = divmod(i, np.shape(self.plot_channel_key)[1])
            data = np.dstack([data_shared[0], data_shared[i + 1]])[0]
            subplot[self.plot_channel_key[idx[0]][idx[1]]].data = data
            subplot.auto_scale(maintain_aspect=False)
        mm.release_mutex(self.mutex)

    @staticmethod
    def offline_initialize_data(filepath: str, plot_channel_key: Union[np.ndarray, str]):
        """ Extract offline sensor data for set of keys
        :param filepath: define path to database to read data from
        :param plot_channel_key: define set of keys in database to plot data
        :return: x and y values
        """
        ys = []
        plot_shape = np.shape(plot_channel_key)
        for key in np.reshape(plot_channel_key, newshape=(1, plot_shape[0] * plot_shape[1]))[0]:
            data = StorageManager.load_serial_channel(key=key, filepath=filepath)
            ys.append(data)

        num_points = len(ys[0])
        xs = [np.linspace(0, num_points - 1, num_points)]
        return xs, ys

    @classmethod
    def offline_plot_data(cls, filepath: str, plot_channel_key: Union[np.ndarray, str] = None):
        """ Initialize plot for offline data
        :param filepath: define path to database to read data from
        :param plot_channel_key: define set of keys in database to plot data
        :return: return plot object
        """
        if plot_channel_key is None:
            database = StorageManager.load_serial_database(filepath=filepath)
            channel_key = []
            with database as db:
                for key in db.keys():
                    channel_key.append(key)
            plot_channel_keys = [channel_key]
        else:
            plot_channel_keys = plot_channel_key

        xs, ys = cls.offline_initialize_data(filepath=filepath, plot_channel_key=plot_channel_keys)
        for i in range(np.shape(plot_channel_keys)[0]*np.shape(plot_channel_keys)[1]):
            if not ys[i][:]:
                ys[i][:] = np.ones(1000) * np.linspace(0, 1, 1000)

        fig = create_fig(plot_channel_key=plot_channel_keys)

        for i, subplot in enumerate(fig):
            idx = divmod(i, np.shape(plot_channel_keys)[1])
            data = np.dstack([xs, ys[i]])[0]
            subplot.add_line(data=data, name=plot_channel_keys[idx[0]][idx[1]], cmap='jet')
            subplot.auto_scale(maintain_aspect=False)

        fig.show()

        return fig
