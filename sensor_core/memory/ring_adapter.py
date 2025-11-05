import numpy as np
import fastring
from typing import Tuple, Optional

class RingBuffer:
    """
    Python adapter for C++ fastring class
    Internally the native ring is always (C_flat, S_flat) with S_flat=1 for images.
    """

    def __init__(self, name, capacity_frames, frame_shape, dtype, create=False):
        self.name = name
        self.capacity = int(capacity_frames)
        self.logical_shape = tuple(frame_shape)
        self.dtype = np.dtype(dtype)

        # Infer mode and flatten geometry used by the C layer for line and image case
        if len(self.logical_shape) == 2:
            self._mode = "line"
            C, S = self.logical_shape
            self._C, self._S = int(C), int(S)
            self.frame_shape = (self._C, self._S)
        elif len(self.logical_shape) == 3:
            self._mode = "image"
            H, W, C = self.logical_shape
            self._H, self._W, self._Cimg = int(H), int(W), int(C)
            self._S = self._H * self._W * self._Cimg
            self.frame_shape = (self._H, self._W, self._Cimg)
        else:
            raise ValueError(f"frame_shape must be (C,S) or (H,W,C), got {self.logical_shape}")

        maker = fastring.Ring.create if create else fastring.Ring.open
        self._ring = maker(self.name, int(self.capacity), int(self._S * self.dtype.itemsize))

    @property
    def write_idx(self) -> int:
        return int(self._ring.write_idx)

    @property
    def frame_bytes(self) -> int:
        return int(self._ring.frame_bytes)

    def publish(self, arr):
        """
        Publish array to ring buffer
        :param arr: input array to store
        """
        a = np.asarray(arr)

        if self._mode == "line":
            if a.ndim == 2:
                if a.shape != (self._C, self._S):
                    raise ValueError(f"publish LINE expects (C,S) got {a.shape}")
                frame = np.ascontiguousarray(a, dtype=self.dtype)
                self._ring.publish(frame)
                return

            if a.ndim == 3 and a.shape[1:] == (self._C, self._S):
                a = np.ascontiguousarray(a, dtype=self.dtype)
                for i in range(a.shape[0]):
                    self._ring.publish(a[i])
                return

            raise ValueError(f"publish Line expects (C,S) or (N,C,S), got {a.shape}")

        else:
            # image mode
            if a.ndim == 3:
                if a.shape != (self._H, self._W, self._Cimg):
                    raise ValueError(f"publish Image expects (H,W,C), got {a.shape}")
                frame = np.ascontiguousarray(a, dtype=self.dtype)
                self._ring.publish(frame)
                return

            if a.ndim == 4 and a.shape[1:] == (self._H, self._W, self._Cimg):
                a = np.ascontiguousarray(a, dtype=self.dtype)
                for i in range(a.shape[0]):
                    self._ring.publish(a[i])
                return

            raise ValueError(f"publish Image expects (H,W,C) or (N,H,W,C), got {a.shape}")

    def view_window(self, start: int, frames: int):
        """
        Return a NumPy view of consecutive frames starting at start
          Falls back to one copy if the underlying memoryview is not C-contiguous.
        :param start: logical start index for view
        :param frames: number of frames to show
        """
        start = int(start)
        frames = int(frames)

        def _call_view(start_i, frames_i, dim0=None, dim1=None):
            try:
                return self._ring.view_window(start_i, frames_i)
            except TypeError:
                if dim0 is None or dim1 is None:
                    raise
                return self._ring.view_window(start_i, frames_i, int(dim0), int(dim1))

        if self._mode == "line":
            C, S = self._C, self._S
            mv = _call_view(start, frames, C, S)
            try:
                arr = np.frombuffer(mv, dtype=self.dtype, count=frames * C * S)
            except BufferError:
                arr = np.frombuffer(mv.tobytes(), dtype=self.dtype, count=frames * C * S)
            return arr.reshape(frames, C, S)

        H, W, Cimg = self._H, self._W, self._Cimg
        mv = _call_view(start, frames, H, W * Cimg)

        elem_count = frames * H * W * Cimg
        try:
            arr = np.frombuffer(mv, dtype=self.dtype, count=elem_count)
            arr = arr.reshape(frames, H, W, Cimg)
            return arr
        except BufferError:
            arr = np.frombuffer(mv.tobytes(), dtype=self.dtype, count=elem_count)
            return arr.reshape(frames, H, W, Cimg)