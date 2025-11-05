import numpy as np
from typing import Tuple, Optional
try:
    from sensor_core.memory.ring_adapter import RingBuffer
except Exception as e:
    RingBuffer = None  # will raise when used

def _normalize_image_shape(shape):
    """
    Normalize image shape for C frames
    :param shape: input image shape
    """
    if len(shape) == 2:
        H, W = shape
        return (int(H), int(W), 1)
    elif len(shape) == 3:
        H, W, C = shape
        return (int(H), int(W), int(C))
    else:
        raise ValueError(f"image frame_shape must be (H,W) or (H,W,C), got {shape}")

def _assert_ring_layout(ring: RingBuffer, logical_shape: Tuple[int, ...], dtype) -> None:
    """
    Check to catch misconfiguration
    :param ring: Ring buffer
    :param logical_shape: input shape
    :param dtype: data type of input
    """
    if ring.dtype != np.dtype(dtype):
        raise TypeError(f"Ring dtype mismatch: ring={ring.dtype}, expected={np.dtype(dtype)}")
    if ring.logical_shape != tuple(logical_shape):
        raise ValueError(f"Ring logical shape mismatch: ring={ring.logical_shape}, expected={logical_shape}")

def initialize_ring(ser_channel_key, window_size, dtype, shm_name="/sensor_ring",
                    frames_capacity=4096, data_mode="line", frame_shape=None):
    """
    Initialize Ring Buffer
    :param ser_channel_key: serial channel key names
    :param window_size: number of fames per acquisition
    :param dtype: data type
    :param shm_name: name of ring buffer object
    :param frames_capacity: max frame capacity of ring buffer
    :param data_mode: allow for line or image data
    :param frame_shape: input shape of each frame
    """
    data_mode = (data_mode or "line").lower()
    if data_mode == "line":
        C = len(ser_channel_key)
        S = int(window_size)
        logical = (C, S)
    else:
        if frame_shape is None:
            raise ValueError("frame_shape is required for data_mode='image'")
        logical = _normalize_image_shape(tuple(frame_shape))

    ring = RingBuffer(shm_name, int(frames_capacity), tuple(logical), np.dtype(dtype), create=True)
    return ring, logical
