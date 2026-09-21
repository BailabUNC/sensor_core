"""Live and offline plots drawn with real figures, rendered offscreen.

These tests need the notebook extras and a GPU or a software renderer (on Linux:
`sudo apt-get install mesa-vulkan-drivers`). They run when SENSOR_CORE_RENDER_TESTS=1.
"""
import os
import uuid

import numpy as np
import pytest

from harness import image_frames, line_acquisitions
from sensor_core.memory.mem_utils import initialize_ring
from sensor_core.memory.strg_manager import add_session, connect, insert_frames
from sensor_core.utils.utils import create_static_dict

pytestmark = [
    pytest.mark.skipif(os.environ.get("SENSOR_CORE_RENDER_TESTS") != "1",
                       reason="set SENSOR_CORE_RENDER_TESTS=1 to run tests that render figures"),
    # pygfx 0.15, which fastplotlib 0.6 requires, uses an array pattern that NumPy 2.5 deprecates
    pytest.mark.filterwarnings("ignore:Setting the shape on a NumPy array:DeprecationWarning"),
]

KEYS = ["red", "infrared", "violet"]
LAG = 16  # the plotter's default: skip the newest 16 frames


@pytest.fixture(scope="module", autouse=True)
def offscreen():
    # must be set before fastplotlib is first imported in this process
    os.environ["RENDERCANVAS_FORCE_OFFSCREEN"] = "1"
    pytest.importorskip("fastplotlib")


def make_plot(shm_name, keys, layout, frame_shape, data_mode, dtype, capacity=4096):
    """A ring buffer and a PlotManager reading it, configured the way SensorManager does it."""
    from sensor_core.plot import PlotManager

    ring, shape = initialize_ring(keys, dtype, shm_name=shm_name, frames_capacity=capacity,
                                  data_mode=data_mode, frame_shape=frame_shape)
    static_args = create_static_dict(ser_channel_key=keys, plot_channel_key=layout, commport=None, baudrate=0,
                                     shm_name=shm_name, shape=shape, dtype=dtype, ring_capacity=capacity,
                                     data_mode=data_mode, frame_shape=shape)
    return ring, PlotManager(static_args_dict=static_args)


def shown(plot, key):
    return plot._lines[key].data.value[:, 1].copy()


@pytest.mark.parametrize("layout", [[["violet"], ["red"], ["infrared"]], [["violet", "red"]]])
def test_each_subplot_draws_its_own_channels_latest_samples(shm_name, layout):
    ring, plot = make_plot(shm_name, KEYS, layout, (100, 10, 3), "line", np.float32)
    acquisitions = line_acquisitions(40, window=10, channels=3)
    for acquisition in acquisitions:
        ring.publish(acquisition)
    plot.online_plot_data()
    plot.fig.canvas.draw()
    for key in (key for row in layout for key in row):
        expected = acquisitions[:len(acquisitions) - LAG, :, KEYS.index(key)].ravel()[-100:]
        np.testing.assert_array_equal(shown(plot, key), expected)


def test_the_image_plot_draws_the_newest_frame_outside_the_lag(shm_name):
    ring, plot = make_plot(shm_name, ["camera"], [["camera"]], (48, 64, 1), "image", np.uint8)
    frames = image_frames(30, (48, 64, 1), np.uint8)
    ring.publish(frames)
    plot.online_plot_data()
    plot.fig.canvas.draw()
    np.testing.assert_array_equal(plot._image.data.value, frames[len(frames) - LAG, :, :, 0])


def test_a_stopped_plot_keeps_its_last_frame(shm_name):
    ring, plot = make_plot(shm_name, KEYS, [KEYS], (100, 10, 3), "line", np.float32)
    acquisitions = line_acquisitions(80, window=10, channels=3)
    for acquisition in acquisitions[:40]:
        ring.publish(acquisition)
    plot.online_plot_data()
    first = shown(plot, "red")
    for acquisition in acquisitions[40:60]:
        ring.publish(acquisition)
    plot.online_plot_data()
    second = shown(plot, "red")
    assert not np.array_equal(first, second)  # a running plot follows new data
    plot.stop()
    for acquisition in acquisitions[60:]:
        ring.publish(acquisition)
    plot.online_plot_data()
    np.testing.assert_array_equal(shown(plot, "red"), second)


@pytest.mark.processes
# earlier tests set up the GPU in this process before SensorManager forks its helper process; that is safe
# here, but it is why create_plot() recommends starting acquisition before plotting
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
def test_sensor_manager_creates_plots_and_stops_them_when_closed(tmp_path):
    from sensor_core import SensorManager

    with SensorManager(ser_channel_key=KEYS, commport=None, frame_shape=(100, 10, 3),
                       sqlite_path=str(tmp_path / "db.sqlite3")) as manager:
        figure = manager.create_plot()
        assert [subplot.name for subplot in figure] == KEYS
        with pytest.warns(DeprecationWarning, match="create_plot"):
            _, old_style_figure = manager.setup_plotting_process()
        assert [subplot.name for subplot in old_style_figure] == KEYS
    assert all(plot._stopped for plot in manager._plot_managers)


def test_offline_plots_draw_the_stored_channels(tmp_path):
    from sensor_core.plot import PlotManager

    db = str(tmp_path / "db.sqlite3")
    acquisitions = line_acquisitions(50, window=10, channels=3)
    conn = connect(db)
    with conn:
        session = {"uuid": uuid.uuid4().hex, "started_unix_ns": 0, "started_monotonic_ns": 0}
        session_id = add_session(conn, session, "line", (10, 3), np.float32, KEYS)
        insert_frames(conn, session_id, [(i, i, a.tobytes()) for i, a in enumerate(acquisitions)])
    conn.close()
    figure = PlotManager.offline_plot_data(db)  # every channel of the latest session
    for channel, subplot in enumerate(figure):
        np.testing.assert_array_equal(subplot.graphics[-1].data.value[:, 1], acquisitions[:, :, channel].ravel())
