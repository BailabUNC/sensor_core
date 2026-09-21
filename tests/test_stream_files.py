"""Stream segments: the on-disk files between the ring buffer and SQLite."""
import json
import os
import struct

import numpy as np

from harness import image_frames
from sensor_core.memory.db_ingester import _read_header, ingest_pending_segments
from sensor_core.memory.stream_logger import MAGIC, BinaryStreamWriter, new_session, sealed_segments
from sensor_core.memory.strg_manager import StorageManager, list_sessions

KEYS = ["red", "infrared", "violet"]
FRAME_SHAPE = (100, 10, 3)
ONE_FRAME = np.ones((10, 3), np.float32)  # one acquisition for FRAME_SHAPE


def make_writer(stream_dir, channels=KEYS, session=None):
    return BinaryStreamWriter(str(stream_dir), "ring", 16, FRAME_SHAPE, np.float32, rotate_frames=10**9,
                              channels=channels, session=session)


def write_one_frame(writer, index=0):
    writer.write_frames(memoryview(ONE_FRAME.tobytes()), ONE_FRAME.nbytes, index, 1, 0)


def read_segment(path):
    with open(path, "rb") as fh:
        version, header, *_ = _read_header(fh)
        return version, header, fh.read()


def test_sealed_segments_are_numbered_in_order(tmp_path):
    writer = make_writer(tmp_path)
    for index in range(3):
        write_one_frame(writer, index)
        writer.rotate()
    writer.close()
    assert [seq for seq, _ in sealed_segments(tmp_path)] == [1, 2, 3]


def test_empty_segments_are_not_sealed(tmp_path):
    writer = make_writer(tmp_path)
    writer.rotate()
    writer.close()
    assert os.listdir(tmp_path) == []


def test_a_new_session_never_modifies_earlier_segments(tmp_path):
    first = make_writer(tmp_path)
    write_one_frame(first)
    first.close()
    (_, path), = sealed_segments(tmp_path)
    with open(path, "rb") as fh:
        before = fh.read()
    second = make_writer(tmp_path)  # e.g., a later run that uses the same stream directory
    write_one_frame(second)
    second.close()
    with open(path, "rb") as fh:
        assert fh.read() == before
    assert [seq for seq, _ in sealed_segments(tmp_path)] == [1, 2]


def test_a_segment_left_active_by_a_crash_is_recovered(tmp_path):
    crashed = make_writer(tmp_path)
    write_one_frame(crashed)
    crashed._fh.close()  # the process dies before sealing its active segment
    make_writer(tmp_path).close()
    (_, path), = sealed_segments(tmp_path)
    _, _, records = read_segment(path)
    assert len(records) == 16 + ONE_FRAME.nbytes


def test_ingested_segments_are_deleted(storage_pipeline):
    shape = (6, 4, 1)
    pipeline = storage_pipeline(keys=["camera"], frame_shape=shape, dtype=np.float32, data_mode="image")
    pipeline.publish(image_frames(3, shape, np.float32))
    pipeline.drain()
    pipeline.rotate()
    assert len(pipeline.sealed()) == 1
    pipeline.ingest()
    assert pipeline.sealed() == []


def test_a_segment_that_cannot_be_stored_is_set_aside(tmp_path):
    stream_dir = tmp_path / "stream"
    stream_dir.mkdir()
    payload = json.dumps({"frame_shape": list(FRAME_SHAPE), "dtype": "float32", "data_mode": "line"}).encode()
    old = stream_dir / "stream_000001.bin"
    old.write_bytes(MAGIC + struct.pack("<H", 1) + struct.pack("<I", len(payload)) + payload)  # format version 1
    writer = make_writer(stream_dir)
    write_one_frame(writer)
    writer.close()  # a valid segment after the bad one

    db = str(tmp_path / "db.sqlite3")
    rejected = []
    ingest_pending_segments(str(stream_dir), db,
                            on_rejected=lambda seq, path, error: rejected.append((seq, str(error))))
    assert len(rejected) == 1 and rejected[0][0] == 1 and "format version 1" in rejected[0][1]
    assert (stream_dir / "stream_000001.bin.rejected").exists()  # kept for inspection
    assert sealed_segments(stream_dir) == []  # the valid segment after it was still stored
    assert len(StorageManager.load_serial_channel("red", filepath=db)) == 10


def test_records_carry_each_frames_global_index(storage_pipeline):
    shape = (6, 4, 1)
    pipeline = storage_pipeline(keys=["camera"], frame_shape=shape, dtype=np.float32, data_mode="image", capacity=4)
    for batch in np.split(image_frames(10, shape, np.float32), 5):
        pipeline.publish(batch)
        pipeline.drain()
    pipeline.rotate()
    (_, path), = pipeline.sealed()
    _, _, records = read_segment(path)
    record_size = 16 + pipeline.ring.frame_bytes
    indices = [struct.unpack_from("<QQ", records, offset)[1] for offset in range(0, len(records), record_size)]
    assert indices == list(range(10))  # keeps counting past the 4-slot ring, so gaps are detectable


def test_recovered_segments_keep_the_session_and_channels_they_were_written_with(tmp_path):
    crashed_session = new_session()
    crashed = make_writer(tmp_path / "stream", channels=["a", "b", "c"], session=crashed_session)
    write_one_frame(crashed)
    crashed._fh.close()  # the process dies before sealing its active segment
    make_writer(tmp_path / "stream", channels=KEYS).close()  # the next session, with different channel names

    db = str(tmp_path / "db.sqlite3")
    ingest_pending_segments(str(tmp_path / "stream"), db)
    (session,) = list_sessions(db)
    assert session["uuid"] == crashed_session["uuid"]
    assert session["channels"] == ["a", "b", "c"]
    assert session["frames"] == 1


def test_a_segment_whose_header_contradicts_its_frames_is_set_aside(tmp_path):
    writer = make_writer(tmp_path / "stream", channels=["red", "infrared"])  # frames have 3 channels
    write_one_frame(writer)
    writer.close()
    rejected = []
    ingest_pending_segments(str(tmp_path / "stream"), str(tmp_path / "db.sqlite3"),
                            on_rejected=lambda seq, path, error: rejected.append(str(error)))
    assert len(rejected) == 1 and "names 2 channels for frames with 3 channels" in rejected[0]


def test_segments_are_sealed_before_they_outgrow_rotate_bytes(tmp_path):
    record = 16 + ONE_FRAME.nbytes
    writer = BinaryStreamWriter(str(tmp_path), "ring", 16, FRAME_SHAPE, np.float32, rotate_frames=10**9,
                                rotate_bytes=3 * record, channels=KEYS)
    frames = np.ones((7, 10, 3), np.float32)
    writer.write_frames(memoryview(frames.tobytes()), ONE_FRAME.nbytes, 0, 7, 0)
    writer.close()
    sizes = [len(read_segment(path)[2]) // record for _, path in sealed_segments(tmp_path)]
    assert sizes == [3, 3, 1]  # every segment holds at most rotate_bytes of records
