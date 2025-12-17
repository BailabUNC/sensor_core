from sensor_core.utils.metrics import RingMetrics, timer
from sensor_core.memory.ring_adapter import RingBuffer
from sensor_core.memory.mem_utils import _assert_ring_layout
import time
from .plot_utils import *
from sensor_core.utils.utils import DictManager, _coerce
from sensor_core.memory.strg_manager import StorageManager
from typing import Union
from sensor_core.dsp.dsp_manager import DSPManager


def run_once(f):
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return f(*args, **kwargs)
    wrapper.has_run = False
    return wrapper

class PlotManager(DictManager):
    def __init__(self, static_args_dict, metrics_proxy=None, plot_dsp_proxy=None):
        """ Initialize Plot Manager Class
        :param static_args_dict: dictionary with static parameters
        :param metrics_proxy: shared proxy for metrics to pass to parent class
        """
        DictManager.__init__(self)

        self.select_dictionary(args_dict=static_args_dict, dict_type="static")
        self.unpack_selected_dict()
        # Set consumer params
        self.plot_target_fps = _coerce(getattr(self, "plot_target_fps", None), 60.0)
        self.plot_catchup_base_max = int(_coerce(getattr(self, "plot_catchup_base_max", None), 2048))
        self.plot_catchup_boost = _coerce(getattr(self, "plot_catchup_boost", None), 2.5)

        # DSP Manager setup
        self._plot_dsp_proxy = plot_dsp_proxy
        self._plot_dsp_local = DSPManager()
        self._plot_dsp_version_seen = -1

        # Open ring (consumer)
        self.ring = RingBuffer(self.shm_name, int(self.ring_capacity),
                               tuple(self.shape), self.data_mode, self.dtype, create=False)
        _assert_ring_layout(self.ring, tuple(self.shape), self.dtype)

        self.metrics = RingMetrics()

        self.fig = None
        self.initialize_fig()

        # Initialize staging once
        if self.data_mode == 'line':
            self._init_ring_plotting_line()
        else:
            self._init_ring_plotting_image()

        self._metrics_proxy = metrics_proxy
        self._last_push = 0.0
        self._last_read_idx = None

        # Register animation once (you had this twice)
        self.fig.add_animations(self.online_plot_data)

    def _sync_plot_dsp_if_needed(self):
        if self._plot_dsp_proxy is None:
            return

        try:
            version = int(self._plot_dsp_proxy.get("version", 0))
        except Exception:
            version = 0

        if version == self._plot_dsp_version_seen:
            return

        modules = self._plot_dsp_proxy.get("modules", {})
        queue = self._plot_dsp_proxy.get("queue", [])

        # rebuild local pipeline from plain dict/list
        self._plot_dsp_local.dsp_modules = dict(modules) if modules else {}
        self._plot_dsp_local.dsp_modules_queue = list(queue) if queue else []

        self._plot_dsp_version_seen = version


    def initialize_fig(self):
        if self.data_mode == "line":
            fig = create_fig(plot_channel_key=self.plot_channel_key)
            ys = initialize_fig_data(num_channel=self.shape[2], num_points=self.shape[0])
            for i, subplot in enumerate(fig):
                idx = divmod(i, np.shape(self.plot_channel_key)[1])
                plot_data = ys[i]
                subplot.add_line(data=plot_data, name=self.plot_channel_key[idx[0]][idx[1]], cmap='jet')
            self.fig = fig
        else:
            fig = create_fig(plot_channel_key=self.plot_channel_key)  # 1x1 layout
            H, W, Cimg = (self.shape[0], self.shape[1], self.shape[2] if len(self.shape) == 3 else 1)
            if Cimg == 1:
                imbuf = np.zeros((H, W), dtype=self.dtype)
            else:
                imbuf = np.zeros((H, W, Cimg), dtype=self.dtype)

            for i, subplot in enumerate(fig):
                im = subplot.add_image(data=imbuf, name="image", cmap="gray")
                im.vmax=255
                im.vmin=0
            self.fig = fig
        return self.fig

    def _init_ring_plotting_line(self):
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
            pos0 = line.data.value  # (S,D)
            S, D = pos0.shape
            stage = np.empty((S, D), dtype=np.float32)
            x0 = pos0[:, 0].astype(np.float32, copy=True)
            stage[:, 0] = x0
            stage[:, 1] = pos0[:, 1]
            if D > 2:
                z0 = pos0[:, 2].astype(np.float32, copy=True)
                stage[:, 2] = z0
            else:
                z0 = None
            self._meta[ch_key] = {"S": S, "D": D, "x0": x0, "z0": z0, "stage": stage}

    def _init_ring_plotting_image(self):
        # store handle to the image graphic
        self._image = None
        for i, subplot in enumerate(self.fig):
            if hasattr(subplot, "graphics") and subplot.graphics:
                for g in subplot.graphics:
                    if getattr(g, "kind", "").lower() == "image" or g.__class__.__name__.lower().startswith("image"):
                        self._image = g
                if self._image is None:
                    self._image = subplot.graphics[-1]

    def online_plot_data(self, *_, **__):
        target_fps = float(getattr(self, "plot_target_fps", 60.0))
        min_dt = 1.0/max(1e-6, target_fps)
        now = time.perf_counter()
        last = getattr(self, "_last_present", 0.0)
        if now - last < min_dt:
            return
        tick_start = time.perf_counter()
        with timer(lambda ms: self.metrics.note_plot_tick(ms, write_idx=int(self.ring.write_idx))):
            try:
                wi = int(self.ring.write_idx)
                if wi <= 0:
                    return
                self.metrics.last_write_idx = wi

                lag = int(getattr(self, "plot_lag_frames", 16))
                cap = int(self.ring.capacity)
                end = wi - lag
                if end < 0:
                    return

                if self.data_mode == "line":
                    N, S = int(self.shape[0]), int(self.shape[1])
                    K = int(np.ceil(N / max(1, S)))
                    start = end - K + 1
                    if start < 0:
                        return

                    slot = start % cap
                    first = min(K, cap - slot)
                    win1 = self.ring.view_window(start, first)
                    rest = K - first
                    if rest:
                        win2 = self.ring.view_window(start + first, rest)
                        win = np.concatenate((win1, win2), axis=0)
                    else:
                        win = win1

                    self.metrics.update_drop_estimate(write_idx_now=wi, frames_read_this_tick=K)

                    block = np.concatenate([win[i] for i in range(win.shape[0])], axis=1)  # (C, K*N)
                    yblock = block[:, -N:]
                    yblock = np.require(yblock, dtype=np.float32, requirements=["C"])
                    if not yblock.flags["OWNDATA"]:
                        yblock = yblock.copy()

                    ncols = int(np.shape(self.plot_channel_key)[1])
                    per_tick_gpu_ms = 0.0
                    for i, subplot in enumerate(self.fig):
                        r, c = divmod(i, ncols)
                        ch_key = self.plot_channel_key[r][c]
                        if ch_key not in self._lines:
                            continue
                        line = self._lines[ch_key]
                        meta = self._meta[ch_key]
                        S_line, D = meta["S"], meta["D"]
                        stage = meta["stage"]
                        x0, z0 = meta["x0"], meta["z0"]

                        # Check DSP Pipeline
                        self._sync_plot_dsp_if_needed()
                        y = yblock[i]
                        if self._plot_dsp_local.dsp_modules_queue:
                            y = np.asarray(y, dtype=np.float32)
                            try:
                                y = self._plot_dsp_local.run_dsp_modules(y)
                            except Exception as e:
                                pass

                        # Ensure correct length for the GPU stage buffer
                        if y.shape[0] != S_line:
                            y = y[-S_line:] if y.shape[0] > S_line else np.pad(y, (S_line - y.shape[0], 0))

                        stage[:, 0] = x0
                        stage[:, 1] = y

                        if D > 2 and z0 is not None:
                            stage[:, 2] = z0

                        t0 = time.perf_counter()
                        line.data[:S_line] = stage[:S_line]
                        per_tick_gpu_ms += (time.perf_counter() - t0) * 1000.0

                    self.metrics.add_gpu_upload_ms(per_tick_gpu_ms)
                    self.metrics.last_read_idx = int(start)
                    self.metrics.frames_lag = int(wi - start)

                else:
                    CATCH_MAX = int(getattr(self, "plot_catch_up_max", 8))

                    if self._last_read_idx is None:
                        frames_to_read = 1
                        start = end
                    else:
                        available = max(0, end - self._last_read_idx)
                        frames_to_read = min(available, CATCH_MAX) if available > 0 else 0
                        start = end - frames_to_read + 1 if frames_to_read > 0 else None

                    if not frames_to_read:
                        self._last_present = time.perf_counter()
                        return

                    # handle wrap case
                    slot = start % cap
                    first = min(frames_to_read, cap - slot)
                    win1 = self.ring.view_window(start, first)
                    rest = frames_to_read - first
                    if rest:
                        win2 = self.ring.view_window(start + first, rest)
                        win = np.concatenate((win1, win2), axis=0)
                    else:
                        win = win1

                    self.metrics.update_drop_estimate(write_idx_now=wi, frames_read_this_tick=frames_to_read)
                    latest = win[-1]

                    if latest.ndim == 3 and latest.shape[2] == 1:
                        latest = latest[:, :, 0]
                    latest = np.require(latest, dtype=np.float32, requirements=["C"])
                    if not latest.flags["OWNDATA"]:
                        latest = latest.copy()

                    # upload to the single image graphic
                    t0 = time.perf_counter()
                    self._image.data[...] = latest
                    self.metrics.add_gpu_upload_ms((time.perf_counter() - t0) * 1000.0)

                    self._last_read_idx = end
                    self.metrics.last_read_idx = int(end)
                    self.metrics.frames_lag = int(wi - end)

                # Bookkeeping & metrics proxy (rate-limited to ~2 Hz)
                self._last_seen_wi = wi
                now = time.perf_counter()
                last_push = getattr(self, "_last_push", 0.0)
                if self._metrics_proxy is not None and (now - last_push) > 0.5:
                    self._metrics_proxy.update(self.metrics.snapshot())
                    self._last_push = now

            except Exception as e:
                import traceback, sys
                print("[plot] exception:", e, file=sys.stderr)
                traceback.print_exc()
                return

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
