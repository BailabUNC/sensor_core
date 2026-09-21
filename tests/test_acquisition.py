"""Acquisition: how frames from a custom acquisition function are validated and normalized."""
import numpy as np
import pytest

KEYS = ["red", "infrared", "violet"]
LINE_SHAPE = (100, 10, 3)  # (num_points, window_size, channels), as in the example notebooks


def test_custom_function_receives_the_serial_handle_and_frame_shape(data_manager):
    manager = data_manager(KEYS, LINE_SHAPE)
    seen = {}

    def acquire(ser, frame_shape):
        seen.update(ser=ser, frame_shape=frame_shape)
        return np.zeros((10, 3))

    manager.acquire_data(func=acquire)
    assert seen == {"ser": None, "frame_shape": LINE_SHAPE}  # a virtual port has no serial handle


def test_line_frames_are_returned_as_float32_window_by_channel(data_manager):
    manager = data_manager(KEYS, LINE_SHAPE)
    frame = manager.acquire_data(func=lambda ser, frame_shape: np.ones((10, 3), dtype=np.float64))
    assert frame.shape == (10, 3)
    assert frame.dtype == np.float32


def test_channel_major_line_frames_are_transposed(data_manager):
    manager = data_manager(KEYS, LINE_SHAPE)
    data = np.arange(30.0).reshape(3, 10)
    frame = manager.acquire_data(func=lambda ser, frame_shape: data)
    np.testing.assert_array_equal(frame, data.T)


def test_line_frames_with_the_wrong_shape_are_rejected(data_manager):
    manager = data_manager(KEYS, LINE_SHAPE)
    with pytest.raises(ValueError, match="LINE data shape"):
        manager.acquire_data(func=lambda ser, frame_shape: np.zeros((9, 3)))


def test_returning_none_means_no_data_yet(data_manager):
    manager = data_manager(KEYS, LINE_SHAPE)
    assert manager.acquire_data(func=lambda ser, frame_shape: None) is None


def test_image_frames_keep_their_dtype_and_gain_a_channel_axis(data_manager):
    manager = data_manager(["camera"], (6, 4), data_mode="image", dtype=np.uint8)
    frame = manager.acquire_data(func=lambda ser, frame_shape: np.zeros((6, 4), np.uint8), data_mode="image")
    assert frame.shape == (6, 4, 1)
    assert frame.dtype == np.uint8


def test_image_frames_with_the_wrong_shape_are_rejected(data_manager):
    manager = data_manager(["camera"], (6, 4), data_mode="image", dtype=np.uint8)
    with pytest.raises(ValueError, match="IMAGE data shape"):
        manager.acquire_data(func=lambda ser, frame_shape: np.zeros((5, 4, 1), np.uint8), data_mode="image")


@pytest.mark.xfail(strict=True, reason="#57: any exception inside a custom acquisition function "
                                       "is replaced by a misleading 'must accept: ser, frame_shape' error")
def test_errors_inside_a_custom_function_are_not_masked(data_manager):
    manager = data_manager(KEYS, LINE_SHAPE)

    def acquire(ser, frame_shape):
        raise RuntimeError("sensor unplugged")

    with pytest.raises(RuntimeError, match="sensor unplugged"):
        manager.acquire_data(func=acquire)


@pytest.mark.xfail(strict=True, reason="#57: num_channel is derived from plot_channel_key, so plotting "
                                       "a subset of channels rejects every acquisition")
def test_plotting_a_subset_of_channels_still_acquires_every_channel(data_manager):
    manager = data_manager(KEYS, LINE_SHAPE, plot_channel_key=[["red", "infrared"]])
    frame = manager.acquire_data(func=lambda ser, frame_shape: np.zeros((10, 3)))
    assert frame.shape == (10, 3)


@pytest.mark.xfail(strict=True, reason="#57: the built-in reader returns (num_points, channels), "
                                       "but validation expects (window_size, channels)")
def test_builtin_reader_returns_a_valid_line_frame(data_manager):
    manager = data_manager(KEYS, LINE_SHAPE)
    assert manager.acquire_data().shape == (10, 3)
