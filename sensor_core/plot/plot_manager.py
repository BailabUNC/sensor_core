from sensor_core.utils.metrics import RingMetrics, timer
from sensor_core.memory.ring_adapter import RingBuffer
from sensor_core.memory.mem_utils import _assert_ring_layout
import time
from .plot_utils import *
from sensor_core.utils.utils import DictManager
from sensor_core.memory.strg_manager import StorageManager


def run_once(f):
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return f(*args, **kwargs)
    wrapper.has_run = False
    return wrapper


class PlotManager(DictManager):
    def __init__(self, static_args_dict, metrics_proxy=None):
        """ Initialize Plot Manager Class
        :param static_args_dict: dictionary with static parameters
        :param metrics_proxy: shared proxy for metrics to pass to parent class
        """
        DictManager.__init__(self)

        self.select_dictionary(args_dict=static_args_dict, dict_type="static")
        self.unpack_selected_dict()

        # Open ring (consumer)
        self.ring = RingBuffer(self.shm_name, int(self.ring_capacity),
                               tuple(self.shape), self.dtype, create=False)
        _assert_ring_layout(self.ring, tuple(self.shape), self.dtype)

        self.metrics = RingMetrics()

        self.fig = None
        self.initialize_fig()

        # Initialize staging once
        self._init_ring_plotting()

        # Register animation once (you had this twice)
        self.fig.add_animations(self.online_plot_data)

        self._metrics_proxy = metrics_proxy
        self._last_push = 0.0

    def initialize_fig(self):
        fig = create_fig(plot_channel_key=self.plot_channel_key)
        ys = initialize_fig_data(num_channel=self.num_channel, num_points=self.num_points)
        for i, subplot in enumerate(fig):
            idx = divmod(i, np.shape(self.plot_channel_key)[1])
            plot_data = ys[i]
            subplot.add_line(data=plot_data, name=self.plot_channel_key[idx[0]][idx[1]], cmap='jet')
        self.fig = fig
        return fig

    def _init_ring_plotting(self):
        self._lines = {}
        self._meta  = {}
        ncols = int(np.shape(self.plot_channel_key)[1])
        for i, subplot in enumerate(self.fig):
            r, c = divmod(i, ncols)
            ch_key = self.plot_channel_key[r][c]
            if hasattr(subplot, "graphics") and subplot.graphics:
                line = subplot.graphics[-1]
            else:
                line = None
            if line is None:
                continue
            self._lines[ch_key] = line
            pos0 = line.data.value
            S, D = pos0.shape
            stage = np.empty((S, D), dtype=np.float32)
            x0 = pos0[:, 0].astype(np.float32, copy=True)
            z0 = pos0[:, 2].astype(np.float32, copy=True) if D > 2 else None
            stage[:, 0] = x0
            stage[:, 1] = pos0[:, 1]
            if D > 2:
                stage[:, 2] = z0
            self._meta[ch_key] = {"S": S, "D": D, "x0": x0, "z0": z0, "stage": stage}

    def online_plot_data(self, *_, **__):
        with timer(lambda ms: self.metrics.note_plot_tick(ms, write_idx=int(self.ring.write_idx))):
            try:
                wi = int(self.ring.write_idx)
                if wi <= 0:
                    return
                self.metrics.last_write_idx = wi

                # Assemble a rolling (C, S) from the last K frames
                C, L = int(self.shape[0]), int(self.shape[1])
                S    = int(self.num_points)
                K    = int(np.ceil(S / max(1, L)))
                lag  = int(getattr(self, "plot_lag_frames", 16))

                start = wi - lag - K
                if start < 0:
                    return

                # Read K frames as a contiguous logical window (may need split)
                cap  = int(self.ring.capacity)
                slot = start % cap
                first = min(K, cap - slot)
                win1  = self.ring.view_window(start, first)
                rest  = K - first
                if rest:
                    win2 = self.ring.view_window(start + first, rest)
                    win  = np.concatenate((win1, win2), axis=0)
                else:
                    win  = win1

                self.metrics.update_drop_estimate(write_idx_now=wi, frames_read_this_tick=K)

                # Concatenate along time to (C, K*L) and take the last S samples
                block  = np.concatenate([win[i] for i in range(win.shape[0])], axis=1)  # (C, K*L)
                yblock = block[:, -S:]                                                  # (C, S)

                # One owning, C-contiguous copy just before GPU upload
                yblock = np.require(yblock, dtype=np.float32, requirements=["C"])
                if not yblock.flags["OWNDATA"]:
                    yblock = yblock.copy()

                self.metrics.last_read_idx = int(start)
                self.metrics.frames_lag = int(wi - start)

                # Update all subplots
                ncols = int(np.shape(self.plot_channel_key)[1])
                per_tick_gpu_ms = 0.0
                for i, subplot in enumerate(self.fig):
                    r, c   = divmod(i, ncols)
                    ch_key = self.plot_channel_key[r][c]
                    if ch_key not in self._lines:
                        continue
                    line   = self._lines[ch_key]
                    meta   = self._meta[ch_key]
                    S_line, D = meta["S"], meta["D"]
                    stage  = meta["stage"]
                    x0, z0 = meta["x0"], meta["z0"]

                    y = yblock[i]
                    if y.shape[0] != S_line:
                        y = y[-S_line:] if y.shape[0] > S_line else np.pad(y, (S_line - y.shape[0], 0))

                    stage[:, 0] = x0
                    stage[:, 1] = y
                    if D > 2 and z0 is not None:
                        stage[:, 2] = z0

                    # Single contiguous upload to GPU buffer
                    now = time.perf_counter()
                    line.data[:S_line] = stage[:S_line]
                    per_tick_gpu_ms += (time.perf_counter() - now) * 1000.0

                    if self._metrics_proxy is not None and (now - self._last_push) > 0.5:
                        self._metrics_proxy.update(self.metrics.snapshot())
                        self._last_push = now

                self.metrics.add_gpu_upload_ms(per_tick_gpu_ms)

            except Exception as e:
                import traceback, sys
                print("[plot] exception:", e, file=sys.stderr)
                traceback.print_exc()


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

        ys = cls.offline_initialize_data(filepath=filepath, plot_channel_key=plot_channel_keys)
        for i in range(np.shape(plot_channel_keys)[0]*np.shape(plot_channel_keys)[1]):
            if not ys[i][:]:
                ys[i][:] = np.ones(1000) * np.linspace(0, 1, 1000)

        fig = create_fig(plot_channel_key=plot_channel_keys)

        for i, subplot in enumerate(fig):
            idx = divmod(i, np.shape(plot_channel_keys)[1])
            subplot.add_line(data=ys[i], name=plot_channel_keys[idx[0]][idx[1]], cmap='jet')
            subplot.auto_scale(maintain_aspect=False)

        fig.show()

        return fig
