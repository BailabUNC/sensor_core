from collections import deque
from contextlib import contextmanager
import time
import numpy as np

# Helper functions for mean and 95% percentile
def _avg(seq):  return float(np.mean(seq)) if seq else 0.0
def _p95(seq):  return float(np.percentile(seq, 95)) if seq else 0.0

class RingMetrics:
    """
    Keeps rolling timing metrics and simple rates for producer (writer) and consumer (plot).
    All durations are in milliseconds.
    """
    def __init__(self, window: int = 500):
        """ Initialize Ring Metrics Class

        :param window: number of samples used to calculate metric
        """
        self.publish_ms = deque(maxlen=window)
        self.plot_ms    = deque(maxlen=window)
        self.gpu_ms     = deque(maxlen=window)
        self.acquire_ms = deque(maxlen=window)

        self._last_pub_t  = None
        self._last_plot_t = None
        self.producer_fps = 0.0
        self.consumer_fps = 0.0

        # ring-related live counters
        self.last_write_idx = 0
        self.last_read_idx  = 0
        self.frames_lag     = 0
        self.drops_est      = 0

        # for crude drop estimation across ticks
        self._prev_write_idx = None

    def note_publish(self, ms: float, write_idx: int | None = None):
        self.publish_ms.append(ms)
        now = time.perf_counter()
        if self._last_pub_t is not None:
            dt = now - self._last_pub_t
            if dt > 0:
                # EWMA for stability
                self.producer_fps = 0.9 * self.producer_fps + 0.1 * (1.0 / dt)
        self._last_pub_t = now
        if write_idx is not None:
            self.last_write_idx = int(write_idx)

    def note_plot_tick(self, ms: float, write_idx: int | None = None):
        self.plot_ms.append(ms)
        now = time.perf_counter()
        if self._last_plot_t is not None:
            dt = now - self._last_plot_t
            if dt > 0:
                self.consumer_fps = 0.9 * self.consumer_fps + 0.1 * (1.0 / dt)
        self._last_plot_t = now
        if write_idx is not None:
            self.last_write_idx = int(write_idx)

    def add_gpu_upload_ms(self, ms: float):
        self.gpu_ms.append(ms)

    def add_acquire_ms(self, ms: float):
        self.acquire_ms.append(ms)

    def update_drop_estimate(self, write_idx_now: int, frames_read_this_tick: int):
        # If writer advanced by more than we consumed, the excess are "drops" at this visualization rate.
        if self._prev_write_idx is not None:
            advanced = max(0, int(write_idx_now) - int(self._prev_write_idx))
            excess   = max(0, advanced - frames_read_this_tick)
            self.drops_est += excess
        self._prev_write_idx = int(write_idx_now)

    def snapshot(self) -> dict:
        return dict(
            producer_fps     = round(self.producer_fps, 2),
            consumer_fps     = round(self.consumer_fps, 2),
            publish_avg_ms   = round(_avg(self.publish_ms), 3),
            publish_p95_ms   = round(_p95(self.publish_ms), 3),
            plot_tick_avg_ms = round(_avg(self.plot_ms), 3),
            plot_tick_p95_ms = round(_p95(self.plot_ms), 3),
            gpu_upload_avg_ms= round(_avg(self.gpu_ms), 3),
            gpu_upload_p95_ms= round(_p95(self.gpu_ms), 3),
            acquire_avg_ms   = round(_avg(self.acquire_ms), 3),
            acquire_p95_ms   = round(_p95(self.acquire_ms), 3),
            write_idx        = int(self.last_write_idx),
            read_idx         = int(self.last_read_idx),
            frames_lag       = int(self.frames_lag),
            drops_est        = int(self.drops_est),
        )

@contextmanager
def timer(cb):
    """
    Usage:
        with timer(lambda ms: metrics.note_publish(ms)):
            do_work()
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        cb((time.perf_counter() - t0) * 1000.0)
