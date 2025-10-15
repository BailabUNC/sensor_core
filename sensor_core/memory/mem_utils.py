import numpy as np
from typing import Tuple
try:
    from sensor_core.memory.ring_adapter import RingBuffer
except Exception as e:
    RingBuffer = None  # will raise when used

def initialize_ring(
    ser_channel_key,
    window_size: int,
    frames_capacity: int = 4096,
    dtype = np.float32,
    shm_name: str = "/sensor_ring",
    create: bool = True
) -> Tuple["RingBuffer", Tuple[int, int]]:
    """Create/open the shared-memory ring buffer.
    :param ser_channel_key: list of serial channel names
    :param window_size: for 1D data, number of time points to acquire before passing
    :param frames_capacity: number of frames ring buffer can hold
    :param dtype: data type (default 32-bit float)
    :param shm_name: name of ring_buffer
    :param create: flag to initialize (true) or reference (false) ring buffer
    Returns (ring, frame_shape) where frame_shape == (C, S).
    """
    if RingBuffer is None:
        raise RuntimeError("RingBuffer adapter not available; build/install native 'fastring' first")

    C = int(np.prod(np.shape(ser_channel_key)))
    S = int(window_size)
    frame_shape = (C, S)
    dtype = np.float32
    ring = RingBuffer(shm_name, frames_capacity, frame_shape, dtype=dtype, create=create)
    return ring, frame_shape

def _assert_ring_layout(ring, expected_shape, expected_dtype):
    """ Safety lock for Ring Buffer
    :param ring: ring buffer object
    :param expected_shape: expected shape of ring buffer frame
    :param expected_dtype: expected data type of ring buffer
    """
    expected_bytes = np.dtype(expected_dtype).itemsize * int(np.prod(expected_shape))
    assert int(ring.frame_bytes) == expected_bytes, (
        f"frame_bytes mismatch: ring={ring.frame_bytes}, "
        f"expected={expected_bytes} for shape={expected_shape}, dtype={expected_dtype}"
    )
