import numpy as np
import fastring  # native pybind11 module (ShmRing)

class RingBuffer:
    """Typed, shape-safe adapter around the native shared-memory ring."""
    def __init__(self, name: str, capacity_frames: int, frame_shape, dtype=np.float32, create: bool = True):
        self.name = name
        self.dtype = np.dtype(dtype)
        self.frame_shape = tuple(frame_shape)  # (C, S)
        self.frame_bytes = int(self.dtype.itemsize * int(np.prod(self.frame_shape)))
        self._capacity = int(capacity_frames)
        maker = fastring.Ring.create if create else fastring.Ring.open
        self._ring = maker(name, self._capacity, self.frame_bytes)

    @property
    def capacity(self) -> int:
        return int(self._ring.capacity)

    @property
    def write_idx(self) -> int:
        return int(self._ring.write_idx)

    def publish(self, frames: np.ndarray) -> None:
        """Publish (C,S) or (N,C,S) frames to the ring."""
        arr = np.asarray(frames)
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        if arr.ndim != 3:
            raise ValueError(f"publish expects (C,S) or (N,C,S), got {arr.shape}")
        if tuple(arr.shape[1:]) != tuple(self.frame_shape):
            raise ValueError(f"frame shape mismatch: got {arr.shape[1:]}, expected {self.frame_shape}")
        if arr.dtype != self.dtype:
            arr = arr.astype(self.dtype, copy=False)
        arr = np.ascontiguousarray(arr)
        self._ring.publish(arr)

    def view_frame(self, logical_idx: int) -> np.ndarray:
        C, S = self.frame_shape
        mv = self._ring.view_frame(int(logical_idx), int(C), int(S))
        return np.asarray(mv)

    def view_window(self, start: int, frames: int) -> np.ndarray:
        C, S = self.frame_shape
        mv = self._ring.view_window(int(start), int(frames), int(C), int(S))
        return np.asarray(mv)
