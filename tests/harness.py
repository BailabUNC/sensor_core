"""Test data generators and an in-process harness for the storage path."""
import os
import sys

import numpy as np

from sensor_core.memory.db_ingester import ingest_sealed_file
from sensor_core.memory.mem_utils import initialize_ring
from sensor_core.memory.stream_logger import BinaryStreamWriter, drain_ring
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

    In the package, these stages run as endless loops in worker processes
    (stream_logger.dump_loop and db_ingester.ingest_loop). Each loop iteration
    calls drain_ring or ingest_sealed_file; this class calls the same functions,
    so a test controls exactly when frames are written, sealed, and ingested.
    """

    def __init__(self, directory, shm_name, keys, frame_shape, dtype, data_mode="line", capacity=64):
        self.keys = list(keys)
        self.ring, shape = initialize_ring(self.keys, dtype, shm_name=shm_name, frames_capacity=capacity,
                                           data_mode=data_mode, frame_shape=frame_shape)
        self.files = [os.path.join(directory, "stream_a.bin"), os.path.join(directory, "stream_b.bin")]
        self.sqlite_path = os.path.join(directory, "db.sqlite3")
        self.writer_metrics = {}
        self.writer = BinaryStreamWriter(*self.files, shm_name, capacity, shape, dtype,
                                         data_mode=data_mode, rotate_frames=10**9, overwrite=True,
                                         metrics_proxy=self.writer_metrics)
        self._last_idx = int(self.ring.write_idx)

    def publish(self, frames):
        for frame in frames:
            self.ring.publish(frame)

    def drain(self):
        """One polling step of the stream writer (dump_loop)."""
        self._last_idx, _ = drain_ring(self.ring, self.writer, self._last_idx)

    def rotate(self):
        """Seal the active stream file and switch to the other one."""
        self.writer._rotate()

    def ingest(self):
        """One scan of the ingester (ingest_loop): ingest every sealed file."""
        for path in self.files:
            if os.path.exists(path + ".seal"):
                ingest_sealed_file(path, self.sqlite_path, self.keys)

    def stored(self, key):
        return np.asarray(StorageManager.load_serial_channel(key, filepath=self.sqlite_path))

    def stored_images(self):
        shape = tuple(StorageManager.load_serial_channel("image_shape", filepath=self.sqlite_path))
        return self.stored("image").reshape((-1, *shape))

    def close(self):
        self.writer.close()
        self.ring = None
