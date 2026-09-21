"""Binary stream files: the on-disk format between the ring buffer and SQLite."""
import json
import struct

import numpy as np
import pytest

from harness import image_frames
from sensor_core.memory.db_ingester import _read_header, ingest_sealed_file
from sensor_core.memory.stream_logger import MAGIC, BinaryStreamWriter

KEYS = ["red", "infrared", "violet"]
ONE_FRAME = np.ones((10, 3), np.float32)  # one acquisition for frame_shape (100, 10, 3)


def read_stream_file(path):
    with open(path, "rb") as fh:
        version, header, *_ = _read_header(fh)
        return version, header, fh.read()


def test_writer_starts_fresh_files_when_the_configuration_changes(tmp_path):
    a, b = str(tmp_path / "a.bin"), str(tmp_path / "b.bin")
    old = BinaryStreamWriter(a, b, "ring", 16, (100, 10, 3), np.float32, overwrite=True)
    old.write_frames(memoryview(ONE_FRAME.tobytes()), ONE_FRAME.nbytes, 0, 1, 0)
    old.close()
    # a later session with a different window size must not append to the old file
    new = BinaryStreamWriter(a, b, "ring", 16, (100, 20, 3), np.float32)
    new.close()
    for path in (a, b):
        _, header, records = read_stream_file(path)
        assert tuple(header["frame_shape"]) == (100, 20, 3)
        assert records == b""


def test_writer_keeps_unsealed_frames_from_a_matching_configuration(tmp_path):
    a, b = str(tmp_path / "a.bin"), str(tmp_path / "b.bin")
    first = BinaryStreamWriter(a, b, "ring", 16, (100, 10, 3), np.float32, overwrite=True)
    first.write_frames(memoryview(ONE_FRAME.tobytes()), ONE_FRAME.nbytes, 0, 1, 0)
    first.close()
    second = BinaryStreamWriter(a, b, "ring", 16, (100, 10, 3), np.float32)
    second.close()
    _, _, records = read_stream_file(a)
    assert len(records) == 16 + ONE_FRAME.nbytes  # still there to be ingested


def test_records_carry_each_frames_global_index(storage_pipeline):
    shape = (6, 4, 1)
    pipeline = storage_pipeline(keys=["camera"], frame_shape=shape, dtype=np.float32, data_mode="image", capacity=4)
    for batch in np.split(image_frames(10, shape, np.float32), 5):
        pipeline.publish(batch)
        pipeline.drain()
    pipeline.writer.close()
    _, _, records = read_stream_file(pipeline.files[0])
    record_size = 16 + pipeline.ring.frame_bytes
    indices = [struct.unpack_from("<QQ", records, offset)[1] for offset in range(0, len(records), record_size)]
    assert indices == list(range(10))  # keeps counting past the 4-slot ring, so gaps are detectable


def test_ingester_refuses_files_in_another_format_version(tmp_path):
    path = tmp_path / "old.bin"
    payload = json.dumps({"frame_shape": [100, 10, 3], "dtype": "float32", "data_mode": "line"}).encode()
    path.write_bytes(MAGIC + struct.pack("<H", 1) + struct.pack("<I", len(payload)) + payload)
    (tmp_path / "old.bin.seal").touch()
    with pytest.raises(ValueError, match="format version 1"):
        ingest_sealed_file(str(path), str(tmp_path / "db.sqlite3"), KEYS)
