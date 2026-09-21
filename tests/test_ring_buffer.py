"""Shared-memory ring buffer: publishing, reading back, wrap-around, and layout checks."""
import gc

import numpy as np
import pytest

from harness import image_frames, line_acquisitions
from sensor_core.memory.mem_utils import initialize_ring
from sensor_core.memory.ring_adapter import RingBuffer

IMAGE_SHAPE = (6, 4, 1)
KEYS = ["red", "infrared", "violet"]
LINE_SHAPE = (100, 10, 3)  # (num_points, window_size, channels), as in the example notebooks


def make_ring(shm_name, data_mode="image", frame_shape=IMAGE_SHAPE, dtype=np.float32, capacity=16, keys=("camera",)):
    ring, _ = initialize_ring(list(keys), dtype, shm_name=shm_name, frames_capacity=capacity,
                              data_mode=data_mode, frame_shape=frame_shape)
    return ring


def test_image_frames_read_back_as_published(shm_name):
    ring = make_ring(shm_name)
    frames = image_frames(5, IMAGE_SHAPE, np.float32)
    ring.publish(frames)
    assert ring.write_idx == 5
    np.testing.assert_array_equal(ring.view_window(0, 5), frames)


def test_ring_wraps_around_and_keeps_the_newest_frames(shm_name):
    ring = make_ring(shm_name, capacity=4)
    frames = image_frames(6, IMAGE_SHAPE, np.float32)
    ring.publish(frames)
    assert ring.write_idx == 6  # the write index keeps counting past the capacity
    newest = np.concatenate([ring.view_window(2, 2), ring.view_window(4, 2)])
    np.testing.assert_array_equal(newest, frames[2:])


def test_read_window_wraps_around_the_end_of_the_ring(shm_name):
    ring = make_ring(shm_name, capacity=4)
    frames = image_frames(6, IMAGE_SHAPE, np.float32)
    ring.publish(frames)
    np.testing.assert_array_equal(ring.read_window(2, 4), frames[2:])


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_single_image_frames_read_back_for_any_dtype(shm_name, dtype):
    ring = make_ring(shm_name, dtype=dtype)
    frame = image_frames(1, IMAGE_SHAPE, dtype)
    ring.publish(frame)
    np.testing.assert_array_equal(ring.view_window(0, 1), frame)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_image_windows_read_back_for_any_dtype(shm_name, dtype):
    ring = make_ring(shm_name, dtype=dtype)
    frames = image_frames(8, IMAGE_SHAPE, dtype)
    ring.publish(frames)
    np.testing.assert_array_equal(ring.view_window(0, 8), frames)


@pytest.mark.parametrize("dtype", [np.float32, np.int16, np.float64])
def test_each_line_acquisition_fills_one_slot(shm_name, dtype):
    ring = make_ring(shm_name, data_mode="line", frame_shape=LINE_SHAPE, dtype=dtype, keys=KEYS)
    acquisitions = line_acquisitions(5, window=10, channels=3).astype(dtype)
    for acquisition in acquisitions:
        ring.publish(acquisition)
    assert ring.write_idx == 5
    np.testing.assert_array_equal(ring.view_window(0, 5), acquisitions)


def test_line_frames_can_be_published_in_either_orientation(shm_name):
    ring = make_ring(shm_name, data_mode="line", frame_shape=LINE_SHAPE, keys=KEYS)
    frame = line_acquisitions(1, window=10, channels=3)[0]
    ring.publish(frame)
    ring.publish(frame.T)
    np.testing.assert_array_equal(ring.view_window(0, 2), [frame, frame])


def test_square_line_frames_are_not_transposed(shm_name):
    # window_size == channels, so the (S, C) and (C, S) orientations have the same shape
    ring = make_ring(shm_name, data_mode="line", frame_shape=(9, 3, 3), keys=KEYS)
    frame = line_acquisitions(1, window=3, channels=3)[0]
    ring.publish(frame)
    np.testing.assert_array_equal(ring.view_window(0, 1)[0], frame)


def test_line_frames_with_the_wrong_shape_are_rejected(shm_name):
    ring = make_ring(shm_name, data_mode="line", frame_shape=LINE_SHAPE, keys=KEYS)
    with pytest.raises(ValueError, match="publish LINE expects"):
        ring.publish(np.zeros((9, 3), np.float32))


def test_image_frames_with_the_wrong_shape_are_rejected(shm_name):
    ring = make_ring(shm_name)
    with pytest.raises(ValueError, match="publish Image expects"):
        ring.publish(np.zeros((5, 4, 1), np.float32))


def test_opening_a_ring_with_a_different_layout_fails(shm_name):
    make_ring(shm_name, capacity=16)
    with pytest.raises(RuntimeError, match="layout mismatch"):
        RingBuffer(shm_name, 8, IMAGE_SHAPE, "image", np.float32)


def test_views_keep_the_ring_mapped_after_it_is_released(shm_name):
    ring = make_ring(shm_name)
    frames = image_frames(2, IMAGE_SHAPE, np.float32)
    ring.publish(frames)
    view = ring.view_window(0, 2)
    del ring
    gc.collect()
    np.testing.assert_array_equal(view, frames)
