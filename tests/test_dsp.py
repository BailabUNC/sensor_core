"""Digital signal processing: filter behavior, module validation, and pipeline order."""
import re

import numpy as np
import pytest

from sensor_core.dsp import DSPManager


def test_moving_average_is_a_trailing_mean_of_the_window():
    data = np.random.default_rng(0).normal(size=200)
    out = DSPManager.moving_average_filter(data, window_size=5, pad="min")
    assert out.shape == data.shape
    np.testing.assert_allclose(out[4:], np.convolve(data, np.ones(5) / 5, mode="valid"))


def test_moving_average_pads_the_start_with_the_minimum():
    data = np.array([4.0, 6.0, 8.0, 10.0])
    out = DSPManager.moving_average_filter(data, window_size=2, pad="min")
    np.testing.assert_allclose(out, [4.0, 5.0, 7.0, 9.0])


def test_moving_average_can_pad_with_the_10th_percentile():
    data = np.array([4.0, 6.0, 8.0, 10.0])
    out = DSPManager.moving_average_filter(data, window_size=2, pad="percentile")
    assert out[0] == pytest.approx((np.percentile(data, 10) + data[0]) / 2)


def test_butterworth_keeps_the_passband_and_removes_the_stopband():
    fs = 1000
    t = np.arange(2 * fs) / fs
    in_band = np.sin(2 * np.pi * 20 * t)
    out_of_band = np.sin(2 * np.pi * 200 * t)
    out = DSPManager.butterworth_filter(in_band + out_of_band, min_frq=5, max_frq=50, order=4, fs=fs)
    assert out.shape == t.shape
    middle = slice(fs // 2, -fs // 2)  # skip the edges, where filtering transients live
    np.testing.assert_allclose(out[middle], in_band[middle], atol=0.02)


@pytest.mark.parametrize("params, message", [
    ({"window_size": 5}, "Missing required parameters"),
    ({"window_size": 5, "pad": "min", "stride": 2}, "unexpected parameters"),
    ({"window_size": 0, "pad": "min"}, "window_size must be >= 1"),
    ({"window_size": 5, "pad": "mean"}, "pad must be 'min' or 'percentile'"),
])
def test_moving_average_parameters_are_validated(params, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        DSPManager().add_dsp_module("smooth", "moving_average_filter", **params)


@pytest.mark.parametrize("params, message", [
    ({"order": 4, "min_frq": 5, "max_frq": 50}, "Missing required parameters"),
    ({"order": 4, "min_frq": 5, "max_frq": 50, "fs": 1000, "gain": 2}, "unexpected parameters"),
    ({"order": 4, "min_frq": 5, "max_frq": 50, "fs": 0}, "fs must be > 0"),
    ({"order": 4, "min_frq": 50, "max_frq": 5, "fs": 1000}, "0 < min_frq < max_frq < fs/2"),
    ({"order": 4, "min_frq": 5, "max_frq": 600, "fs": 1000}, "0 < min_frq < max_frq < fs/2"),
    ({"order": 0, "min_frq": 5, "max_frq": 50, "fs": 1000}, "order must be >= 1"),
])
def test_butterworth_parameters_are_validated(params, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        DSPManager().add_dsp_module("band", "butterworth_filter", **params)


def test_unknown_algorithms_are_rejected():
    with pytest.raises(ValueError, match="Unsupported module algorithm"):
        DSPManager().add_dsp_module("median", "median_filter", window_size=3)


def test_module_names_must_be_unique():
    dsp = DSPManager()
    dsp.add_dsp_module("smooth", "moving_average_filter", window_size=3, pad="min")
    with pytest.raises(ValueError, match="already exists"):
        dsp.add_dsp_module("smooth", "moving_average_filter", window_size=3, pad="min")


def test_modules_run_in_the_order_they_were_added():
    data = np.random.default_rng(1).normal(size=500)
    dsp = DSPManager()
    dsp.add_dsp_module("smooth", "moving_average_filter", window_size=4, pad="min")
    dsp.add_dsp_module("band", "butterworth_filter", order=2, min_frq=5, max_frq=50, fs=1000)
    smoothed = DSPManager.moving_average_filter(data, 4, "min")
    expected = DSPManager.butterworth_filter(smoothed, 5, 50, 2, 1000)
    np.testing.assert_allclose(dsp.run_dsp_modules(data), expected)


def test_removed_modules_no_longer_run():
    data = np.arange(10.0)
    dsp = DSPManager()
    dsp.add_dsp_module("smooth", "moving_average_filter", window_size=3, pad="min")
    dsp.remove_dsp_module("smooth")
    np.testing.assert_array_equal(dsp.run_dsp_modules(data), data)
    with pytest.raises(ValueError, match="does not exist"):
        dsp.remove_dsp_module("smooth")


def test_running_without_modules_returns_an_unmodified_copy():
    data = np.arange(10.0)
    out = DSPManager().run_dsp_modules(data)
    np.testing.assert_array_equal(out, data)
    assert out is not data
