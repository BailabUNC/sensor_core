"""Shared-memory ring buffer: publishing, reading back, and wrap-around."""
import numpy as np
import pytest

from harness import image_frames, line_acquisitions
from sensor_core.memory.mem_utils import initialize_ring

IMAGE_SHAPE = (6, 4, 1)
KEYS = ["red", "infrared", "violet"]


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


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_single_image_frames_read_back_for_any_dtype(shm_name, dtype):
    ring = make_ring(shm_name, dtype=dtype)
    frame = image_frames(1, IMAGE_SHAPE, dtype)
    ring.publish(frame)
    np.testing.assert_array_equal(ring.view_window(0, 1), frame)


FLOAT_ONLY_VIEWS = "#57: the C++ ring views assume 4-byte items, so multi-frame windows of other dtypes are corrupted"


@pytest.mark.parametrize("dtype", [
    pytest.param(np.uint8, marks=pytest.mark.xfail(strict=True, reason=FLOAT_ONLY_VIEWS)),
    pytest.param(np.uint16, marks=pytest.mark.xfail(strict=True, reason=FLOAT_ONLY_VIEWS)),
    np.float32,
])
def test_image_windows_read_back_for_any_dtype(shm_name, dtype):
    ring = make_ring(shm_name, dtype=dtype)
    frames = image_frames(8, IMAGE_SHAPE, dtype)
    ring.publish(frames)
    np.testing.assert_array_equal(ring.view_window(0, 8), frames)


def test_line_frames_can_be_published_in_either_orientation(shm_name):
    ring = make_ring(shm_name, data_mode="line", frame_shape=(100, 10, 3), keys=KEYS)
    frame = line_acquisitions(1, window=10, channels=3)[0]
    ring.publish(frame)
    after_first = ring.write_idx
    ring.publish(frame.T)
    assert 0 < after_first < ring.write_idx


def test_line_frames_with_the_wrong_shape_are_rejected(shm_name):
    ring = make_ring(shm_name, data_mode="line", frame_shape=(100, 10, 3), keys=KEYS)
    with pytest.raises(ValueError, match="publish LINE expects"):
        ring.publish(np.zeros((9, 3), np.float32))


def test_image_frames_with_the_wrong_shape_are_rejected(shm_name):
    ring = make_ring(shm_name)
    with pytest.raises(ValueError, match="publish Image expects"):
        ring.publish(np.zeros((5, 4, 1), np.float32))
