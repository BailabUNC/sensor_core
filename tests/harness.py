"""Test data generators and an in-process harness for the storage and plotting paths."""
import os
import sys
import time

import numpy as np

from sensor_core.memory import db_ingester
from sensor_core.memory.mem_utils import initialize_ring
from sensor_core.memory.stream_logger import BinaryStreamWriter
from sensor_core.memory.strg_manager import StorageManager


def unlink_shared_memory(name):
    """Remove a POSIX shared-memory object; sensor_core never unlinks its rings (#57)."""
    if sys.platform == "win32":
        return  # Windows releases a mapping when its last handle closes
    import _posixshmem

    try:
        _posixshmem.shm_unlink(name)
    except FileNotFoundError:
        pass


def line_acquisitions(count, window, channels):
    """`count` line acquisitions, each shaped (window, channels).

    Every value is unique and increases along the stream, so any reordering,
    duplication, loss, or mixing of channels changes the data.
    """
    return np.arange(count * window * channels, dtype=np.float32).reshape(count, window, channels)


def image_frames(count, shape, dtype):
    """`count` image frames whose pixels differ within and between frames."""
    values = np.arange(count * int(np.prod(shape))).reshape((count, *shape))
    if np.issubdtype(np.dtype(dtype), np.integer):
        values = values % 251
    return values.astype(dtype)


class StoragePipeline:
    """Runs the storage stages in one process: ring buffer -> binary stream files -> SQLite.

    In the package, these stages are endless loops in worker processes
    (stream_logger.dump_loop and db_ingester.ingest_loop). drain() and ingest()
    repeat one iteration of each loop, so a test controls exactly when frames
    are written, sealed, and ingested. Once SensorManager can stop and flush
    (#57), this class should drive that API instead; the tests describe
    behavior and should not need to change.
    """

    def __init__(self, directory, shm_name, keys, frame_shape, dtype, data_mode="line", capacity=64):
        self.keys = list(keys)
        self.ring, shape = initialize_ring(self.keys, dtype, shm_name=shm_name, frames_capacity=capacity,
                                           data_mode=data_mode, frame_shape=frame_shape)
        self.files = [os.path.join(directory, "stream_a.bin"), os.path.join(directory, "stream_b.bin")]
        self.sqlite_path = os.path.join(directory, "db.sqlite3")
        self.writer = BinaryStreamWriter(*self.files, shm_name, capacity, shape, dtype,
                                         data_mode=data_mode, rotate_frames=10**9, overwrite=True)
        self._last_idx = int(self.ring.write_idx)

    def publish(self, frames):
        for frame in frames:
            self.ring.publish(frame)

    def drain(self):
        """One polling step of the stream writer (dump_loop)."""
        wi = int(self.ring.write_idx)
        cap = self.ring.capacity
        n = (wi - self._last_idx) % cap
        if n == 0:
            return
        start = (wi - n) % cap
        ts_ns = time.time_ns()
        first = min(n, cap - start)
        self.writer.write_frames(self.ring.view_window_bytes(start, first), self.ring.frame_bytes,
                                 start, first, ts_ns)
        if n > first:
            self.writer.write_frames(self.ring.view_window_bytes(0, n - first), self.ring.frame_bytes,
                                     0, n - first, ts_ns)
        self._last_idx = wi

    def rotate(self):
        """Seal the active stream file and switch to the other one."""
        self.writer._rotate()

    def ingest(self):
        """One scan of the ingester (ingest_loop): ingest every sealed file."""
        for path in self.files:
            seal = path + ".seal"
            if not os.path.exists(seal):
                continue
            with open(path, "rb") as fh:
                _, header, ver_b, len_b, payload = db_ingester._read_header(fh)
            dtype = np.dtype(header["dtype"])
            shape = tuple(header["frame_shape"])
            counts = {}
            if header.get("data_mode", "line") == "line":
                N, _, C = shape
                db_ingester._ingest_file_line(path, self.sqlite_path, self.keys, 32, dtype, N, C, counts)
            else:
                db_ingester._ingest_file_image(path, self.sqlite_path, shape, 32, dtype, counts)
            os.remove(seal)
            with open(path, "wb") as out:  # the ingester truncates each file back to its header
                out.write(db_ingester.MAGIC + ver_b + len_b + payload)

    def stored(self, key):
        return np.asarray(StorageManager.load_serial_channel(key, filepath=self.sqlite_path))

    def stored_images(self):
        shape = tuple(StorageManager.load_serial_channel("image_shape", filepath=self.sqlite_path))
        return self.stored("image").reshape((-1, *shape))

    def close(self):
        self.writer._fh.close()
        self.ring = None


def plotted_line_traces(ring, frame_shape, plot_channel_key, lag=16):
    """The trace each subplot shows, computed as PlotManager.online_plot_data does in line mode.

    Mirrors plot_manager.py (lines 164-205) without creating a figure, which
    needs a GPU. Once #57 separates this extraction from rendering, call it
    directly instead.
    """
    N, S = int(frame_shape[0]), int(frame_shape[1])
    end = int(ring.write_idx) - lag
    K = int(np.ceil(N / max(1, S)))
    start = end - K + 1
    first = min(K, ring.capacity - start % ring.capacity)
    win = ring.view_window(start, first)
    if K > first:
        win = np.concatenate((win, ring.view_window(start + first, K - first)), axis=0)
    yblock = np.concatenate([win[i] for i in range(win.shape[0])], axis=1)[:, -N:]
    ncols = int(np.shape(plot_channel_key)[1])
    traces = {}
    for i in range(int(np.size(plot_channel_key))):
        row, col = divmod(i, ncols)
        traces[plot_channel_key[row][col]] = yblock[i]
    return traces
