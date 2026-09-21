"""End to end storage: frames published to the ring must reach SQLite unchanged.

Each test publishes known data, lets the stream writer drain the ring, seals
the stream file, runs the ingester, and compares what SQLite holds with what
was published. "Per poll" is how many frames accumulate in the ring between
two polls of the writer; in the package that depends on timing, so these
tests pin it down.
"""
import numpy as np
import pytest

from harness import image_frames, line_acquisitions

LINE_KEYS = ["red", "infrared", "violet"]
LINE_SHAPE = (100, 10, 3)  # (num_points, window_size, channels), as in the example notebooks
IMAGE_SHAPE = (6, 4, 1)


@pytest.mark.xfail(strict=True, reason="#57: the ingester reads line records with the wrong size, so "
                                       "stored samples do not match what was acquired")
@pytest.mark.parametrize("acquisitions_per_poll", [1, 2, 8])
def test_line_data_reaches_sqlite_unchanged(storage_pipeline, acquisitions_per_poll):
    pipeline = storage_pipeline(keys=LINE_KEYS, frame_shape=LINE_SHAPE, dtype=np.float32)
    acquisitions = line_acquisitions(4 * acquisitions_per_poll, window=10, channels=3)
    for batch in np.split(acquisitions, 4):
        pipeline.publish(batch)
        pipeline.drain()
    pipeline.rotate()
    pipeline.ingest()
    for channel, key in enumerate(LINE_KEYS):
        np.testing.assert_array_equal(pipeline.stored(key), acquisitions[:, :, channel].ravel())


BATCH_CORRUPTION = ("#57: the writer's C++ view assumes 4-byte items, so batches of smaller items "
                    "are stored as copies of earlier frames")


@pytest.mark.parametrize("dtype, frames_per_poll", [
    (np.float32, 1),
    (np.float32, 8),
    (np.uint8, 1),
    (np.uint8, 4),
    pytest.param(np.uint8, 8, marks=pytest.mark.xfail(strict=True, reason=BATCH_CORRUPTION)),
    (np.uint16, 2),
    pytest.param(np.uint16, 4, marks=pytest.mark.xfail(strict=True, reason=BATCH_CORRUPTION)),
])
def test_image_data_reaches_sqlite_unchanged(storage_pipeline, dtype, frames_per_poll):
    pipeline = storage_pipeline(keys=["camera"], frame_shape=IMAGE_SHAPE, dtype=dtype, data_mode="image")
    frames = image_frames(3 * frames_per_poll, IMAGE_SHAPE, dtype)
    for batch in np.split(frames, 3):
        pipeline.publish(batch)
        pipeline.drain()
    pipeline.rotate()
    pipeline.ingest()
    np.testing.assert_array_equal(pipeline.stored_images(), frames)


@pytest.mark.xfail(strict=True, reason="#57: rotating back to a sealed file truncates it even when "
                                       "the ingester has not read it yet")
def test_rotation_keeps_frames_the_ingester_has_not_reached(storage_pipeline):
    pipeline = storage_pipeline(keys=["camera"], frame_shape=IMAGE_SHAPE, dtype=np.float32, data_mode="image")
    frames = image_frames(4, IMAGE_SHAPE, np.float32)
    pipeline.publish(frames[:2])
    pipeline.drain()
    pipeline.rotate()  # file A is sealed, and the ingester falls behind
    pipeline.publish(frames[2:])
    pipeline.drain()
    pipeline.rotate()  # the writer switches back to file A
    pipeline.ingest()
    np.testing.assert_array_equal(pipeline.stored_images(), frames)


@pytest.mark.xfail(strict=True, reason="#57: the ingester discards each record's timestamp")
@pytest.mark.parametrize("data_mode", ["line", "image"])
def test_every_stored_frame_has_a_timestamp(storage_pipeline, data_mode):
    if data_mode == "line":
        pipeline = storage_pipeline(keys=LINE_KEYS, frame_shape=LINE_SHAPE, dtype=np.float32)
        frames = line_acquisitions(5, window=10, channels=3)
    else:
        pipeline = storage_pipeline(keys=["camera"], frame_shape=IMAGE_SHAPE, dtype=np.float32, data_mode="image")
        frames = image_frames(5, IMAGE_SHAPE, np.float32)
    pipeline.publish(frames)
    pipeline.drain()
    pipeline.rotate()
    pipeline.ingest()
    times = pipeline.stored("time")
    assert len(times) == len(frames)
    assert np.all(np.diff(times) >= 0)
