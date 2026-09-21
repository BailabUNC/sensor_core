"""Test data generators and an in-process harness for the storage path."""
import os

import numpy as np

from sensor_core.memory.db_ingester import ingest_pending_segments
from sensor_core.memory.mem_utils import initialize_ring
from sensor_core.memory.ring_adapter import unlink_ring
from sensor_core.memory.stream_logger import BinaryStreamWriter, drain_ring, new_session, sealed_segments
from sensor_core.memory.strg_manager import load_channel, load_frame_times, load_images


def unlink_shared_memory(name):
    """Remove a ring buffer's shared-memory name (a no-op on Windows, which frees it automatically)."""
    unlink_ring(name)


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

    In the package, these stages run as loops in worker processes
    (stream_logger.dump_loop and db_ingester.ingest_loop). Each loop iteration
    calls drain_ring or ingest_pending_segments; this class calls the same
    functions, so a test controls exactly when frames are written, sealed, and
    ingested.
    """

    def __init__(self, directory, shm_name, keys, frame_shape, dtype, data_mode="line", capacity=64):
        self.keys = list(keys)
        self.ring, shape = initialize_ring(self.keys, dtype, shm_name=shm_name, frames_capacity=capacity,
                                           data_mode=data_mode, frame_shape=frame_shape)
        self.stream_dir = os.path.join(directory, "stream")
        self.sqlite_path = os.path.join(directory, "db.sqlite3")
        self.writer_metrics = {}
        self.session = new_session()
        self.writer = BinaryStreamWriter(self.stream_dir, shm_name, capacity, shape, dtype,
                                         data_mode=data_mode, rotate_frames=10**9,
                                         metrics_proxy=self.writer_metrics,
                                         channels=self.keys, session=self.session)
        self._last_idx = 0  # like dump_loop: start from the ring's first frame

    def publish(self, frames, timestamps=None):
        """Publish frames, timestamped now or with the given acquisition times (time.perf_counter_ns)."""
        for i, frame in enumerate(frames):
            self.ring.publish(frame, None if timestamps is None else timestamps[i])

    def drain(self):
        """One polling step of the stream writer (dump_loop)."""
        self._last_idx, _ = drain_ring(self.ring, self.writer, self._last_idx)

    def rotate(self):
        """Seal the active segment and start a new one."""
        self.writer.rotate()

    def ingest(self):
        """One pass of the ingester (ingest_loop): store every sealed segment, oldest first."""
        ingest_pending_segments(self.stream_dir, self.sqlite_path)

    def sealed(self):
        """Sealed segments waiting to be ingested, as (sequence number, path)."""
        return sealed_segments(self.stream_dir)

    def stored(self, key):
        return load_channel(self.sqlite_path, key)

    def stored_images(self):
        return load_images(self.sqlite_path)

    def stored_times(self, clock="monotonic"):
        return load_frame_times(self.sqlite_path, clock=clock)

    def close(self):
        self.writer.close()
        self.ring = None
