import numpy as np
from sensor_core import _fastring as fastring


def unlink_ring(name: str):
    """
    Remove a ring buffer's name so no new process can open it; existing mappings stay valid until released
    """
    fastring.Ring.unlink(name)


class RingBuffer:
    """
    Python adapter for the shared-memory ring buffer in sensor_core._fastring.

    Each slot of the ring holds one frame:
      line mode:  one acquisition, shaped (window_size, channels)
      image mode: one image, shaped (height, width, channels)
    """
    def __init__(self,
                 name,
                 capacity_frames,
                 frame_shape,
                 data_mode,
                 dtype,
                 create=False):
        self.name = name
        self.capacity = int(capacity_frames)
        self.logical_shape = tuple(int(x) for x in frame_shape)
        self.frame_shape = self.logical_shape
        self.dtype = np.dtype(dtype)
        self._mode = str(data_mode)

        if self._mode == "line":
            N, S, C = self.logical_shape
            self._N, self._S, self._C = N, S, C
            self.slot_shape = (S, C)
        elif self._mode == "image":
            H, W, C = self.logical_shape
            self._H, self._W, self._Cimg = H, W, C
            self.slot_shape = (H, W, C)
        else:
            raise ValueError(f"data_mode must be 'line' or 'image', got {data_mode!r}")

        frame_bytes = int(np.prod(self.slot_shape)) * self.dtype.itemsize
        maker = fastring.Ring.create if create else fastring.Ring.open
        self._ring = maker(self.name, self.capacity, frame_bytes)

    @property
    def write_idx(self) -> int:
        return int(self._ring.write_idx)

    @property
    def frame_bytes(self) -> int:
        return int(self._ring.frame_bytes)

    def publish(self, arr):
        """
        Publish one frame, or a batch of image frames, to the ring buffer
        :param arr: line mode: (window_size, channels), or (channels, window_size) which is transposed;
                    image mode: (height, width, channels) or a batch (n, height, width, channels)
        """
        a = np.asarray(arr)

        if self._mode == "line":
            if a.shape != self.slot_shape and a.shape == (self._C, self._S):
                a = a.T
            if a.shape != self.slot_shape:
                raise ValueError(f"publish LINE expects (S,C) = {self.slot_shape}, got {a.shape}")
            self._ring.publish(np.ascontiguousarray(a, dtype=self.dtype))
            return

        if a.shape == self.slot_shape or (a.ndim == 4 and a.shape[1:] == self.slot_shape):
            self._ring.publish(np.ascontiguousarray(a, dtype=self.dtype))
            return
        raise ValueError(f"publish Image expects (H,W,C) or (N,H,W,C) with (H,W,C) = {self.slot_shape}, got {a.shape}")

    def view_window(self, start: int, frames: int):
        """
        Read-only NumPy view of consecutive frames, shaped (frames, *slot_shape)
        :param start: logical index of the first frame
        :param frames: number of frames; the window must not wrap past the end of the ring (see read_window)
        """
        frames = int(frames)
        if frames <= 0:
            return np.empty((0, *self.slot_shape), dtype=self.dtype)
        raw = self._ring.view_bytes(int(start), frames)  # uint8 array that keeps the ring mapped
        return raw.view(self.dtype).reshape((frames, *self.slot_shape))

    def read_window(self, start: int, frames: int):
        """
        Copy of consecutive frames, shaped (frames, *slot_shape), wrapping around the ring as needed
        :param start: logical index of the first frame
        :param frames: number of frames (at most the ring capacity)
        """
        start, frames = int(start), int(frames)
        first = min(frames, self.capacity - start % self.capacity)
        parts = [self.view_window(start, first)]
        if frames > first:
            parts.append(self.view_window(start + first, frames - first))
        return np.concatenate(parts, axis=0)

    def view_window_bytes(self, start: int, frames: int):
        """
        Read-only bytes of consecutive frames, for writing them to disk without a copy
        """
        if frames <= 0:
            return memoryview(b"")
        return memoryview(self._ring.view_bytes(int(start), int(frames)))
