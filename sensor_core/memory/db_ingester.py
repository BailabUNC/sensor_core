import os, json, struct, time, traceback
import multiprocessing
from typing import List, Optional, Tuple
import numpy as np
from sqlitedict import SqliteDict
from .strg_manager import StorageManager
from .stream_logger import MAGIC, VERSION as STREAM_VERSION, sealed_segments, _safe_update

MAGIC_LEN = len(MAGIC)
REC_HEADER_SZ = 16

def _read_header(fh):
    magic = fh.read(MAGIC_LEN)
    if magic != MAGIC:
        raise ValueError('Invalid stream magic')
    ver_bytes = fh.read(2);  ver = int.from_bytes(ver_bytes, 'little')
    len_bytes = fh.read(4);  hdr_len = int.from_bytes(len_bytes, 'little')
    payload = fh.read(hdr_len)
    hdr = json.loads(payload.decode('utf-8'))
    return ver, hdr, ver_bytes, len_bytes, payload

def _ensure_sqlite_keys_line(sqlite_path: str, channel_keys: List[str], dtype: np.dtype):
    with SqliteDict(sqlite_path) as db:
        for k in channel_keys:
            if k not in db:
                db[k] = np.array([], dtype=dtype)
        if 'time' not in db:
            db['time'] = np.array([], dtype=np.float64)
        db.commit()

def _ensure_sqlite_keys_image(sqlite_path: str, shape: Tuple[int,int,int], dtype: np.dtype):
    with SqliteDict(sqlite_path) as db:
        if 'image' not in db:
            db['image'] = np.array([], dtype=dtype)  # flattened frames appended
        if 'image_shape' not in db:
            db['image_shape'] = tuple(shape)         # (H,W,Cimg)
        db.commit()

def _ingest_file_line(path: str, sqlite_path: str, channel_keys: List[str],
                      batch_frames: int, dtype: np.dtype, window: int, channels: int,
                      metrics_accum: dict):
    if len(channel_keys) != channels:
        raise ValueError(f"{len(channel_keys)} channel keys given for {channels} channels in {path}")
    _ensure_sqlite_keys_line(sqlite_path, channel_keys, dtype)
    sm = StorageManager(channel_key=channel_keys, filepath=sqlite_path, overwrite=False)
    record_bytes = window * channels * dtype.itemsize
    acc = {k: [] for k in channel_keys}
    frames = 0; bytes_read = 0; batches = 0

    def _flush():
        for key in channel_keys:
            if acc[key]:
                sm.append_serial_channel(key, np.concatenate(acc[key], axis=0))
                acc[key].clear()

    with open(path, 'rb') as fh:
        _read_header(fh)
        while True:
            rec = fh.read(REC_HEADER_SZ)
            if len(rec) < REC_HEADER_SZ:
                break
            ts_ns, wi = struct.unpack('<QQ', rec)
            raw = fh.read(record_bytes)
            if len(raw) < record_bytes:
                break
            frame = np.frombuffer(raw, dtype=dtype).reshape(window, channels)
            for ci, key in enumerate(channel_keys):
                acc[key].append(frame[:, ci])
            frames += 1
            bytes_read += REC_HEADER_SZ + len(raw)
            if len(acc[channel_keys[0]]) >= batch_frames:
                _flush()
                batches += 1
    _flush()
    metrics_accum["frames_ingested"] = metrics_accum.get("frames_ingested", 0) + frames
    metrics_accum["bytes_read"] = metrics_accum.get("bytes_read", 0) + bytes_read
    metrics_accum["batches_flushed"] = metrics_accum.get("batches_flushed", 0) + batches

def _ingest_file_image(path: str, sqlite_path: str, shape: Tuple[int,int,int],
                       batch_frames: int, dtype: np.dtype, metrics_accum: dict):
    H, W, Cimg = shape
    frame_items = H * W * Cimg
    _ensure_sqlite_keys_image(sqlite_path, shape, dtype)
    sm = StorageManager(channel_key=['image'], filepath=sqlite_path, overwrite=False)
    acc = []
    frames = 0; bytes_read = 0; batches = 0
    with open(path, 'rb') as fh:
        ver, hdr, ver_b, len_b, payload = _read_header(fh)
        while True:
            rec = fh.read(REC_HEADER_SZ)
            if len(rec) < REC_HEADER_SZ:
                break
            ts_ns, wi = struct.unpack('<QQ', rec)
            raw = fh.read(frame_items * dtype.itemsize)
            if len(raw) < frame_items * dtype.itemsize:
                break
            arr = np.frombuffer(raw, dtype=dtype, count=frame_items)  # flat
            acc.append(arr)
            frames += 1
            bytes_read += REC_HEADER_SZ + len(raw)
            if len(acc) >= batch_frames:
                block = np.concatenate(acc, axis=0)
                sm.append_serial_channel('image', block)
                acc.clear()
                batches += 1
    if acc:
        block = np.concatenate(acc, axis=0)
        sm.append_serial_channel('image', block)
        acc.clear()
    metrics_accum["frames_ingested"] = metrics_accum.get("frames_ingested", 0) + frames
    metrics_accum["bytes_read"] = metrics_accum.get("bytes_read", 0) + bytes_read
    metrics_accum["batches_flushed"] = metrics_accum.get("batches_flushed", 0) + batches
    return ver_b, len_b, payload

def ingest_segment(path: str, sqlite_path: str, channel_keys: List[str], batch_frames: int = 32):
    """
    Ingest one sealed segment into SQLite, then delete it
    :return: (header, counts) -- the segment's header, and counts of frames, bytes, and batches ingested
    :raises ValueError: if the segment was written in another format or does not match channel_keys
    """
    with open(path, 'rb') as fh:
        ver, hdr, _, _, _ = _read_header(fh)
    if ver != STREAM_VERSION:
        raise ValueError(f"{path} uses stream format version {ver}; this version of sensor_core reads version {STREAM_VERSION}")
    dtype = np.dtype(hdr['dtype'])
    shape = tuple(hdr['frame_shape'])
    mode = hdr.get('data_mode', 'line')
    counts = {"frames_ingested": 0, "bytes_read": 0, "batches_flushed": 0}
    if mode == 'line':
        _, window, channels = shape
        _ingest_file_line(path, sqlite_path, channel_keys, batch_frames, dtype, window, channels, counts)
    elif mode == 'image':
        _ingest_file_image(path, sqlite_path, shape, batch_frames, dtype, counts)
    else:
        raise ValueError(f"{path} has unknown data_mode {mode!r}")
    os.remove(path)
    return hdr, counts


def ingest_pending_segments(stream_dir: str, sqlite_path: str, channel_keys: List[str],
                            batch_frames: int = 32, on_ingested=None, on_rejected=None) -> int:
    """
    Ingest every sealed segment in stream_dir, oldest first
    A segment that cannot be ingested is renamed to *.rejected so it neither blocks later segments nor
    is retried forever; its data stays on disk for inspection.
    :param on_ingested: optional callback(seq, header, counts) after each segment is stored
    :param on_rejected: optional callback(seq, path, error) for each segment set aside
    :return: number of segments processed
    """
    processed = 0
    for seq, path in sealed_segments(stream_dir):
        try:
            hdr, counts = ingest_segment(path, sqlite_path, channel_keys, batch_frames)
        except ValueError as e:
            os.replace(path, path + ".rejected")
            if on_rejected is not None:
                on_rejected(seq, path, e)
        else:
            if on_ingested is not None:
                on_ingested(seq, hdr, counts)
        processed += 1
    return processed


def ingest_loop(stream_dir: str, sqlite_path: str, channel_keys: List[str],
                batch_frames: int = 32, sleep_s: float = 0.2,
                metrics_proxy: Optional[dict] = None,
                data_mode_hint: Optional[str] = None,
                frame_shape_hint: Optional[Tuple[int, ...]] = None,
                dtype_hint: Optional[str] = None,
                precreate_sqlite: bool = True,
                stop_event=None, ready_event=None):
    """
    Ingester process: move sealed segments from stream_dir into SQLite until stop_event is set
    :param stop_event: when set, ingest every remaining sealed segment and return
    :param ready_event: set once the ingester is running
    """
    totals = {"segments": 0, "frames": 0, "bytes": 0, "batches": 0, "rejected": 0}
    rate = {"frames": 0, "t": time.monotonic()}

    def _on_ingested(seq, hdr, counts):
        totals["segments"] += 1
        totals["frames"] += int(counts["frames_ingested"])
        totals["bytes"] += int(counts["bytes_read"])
        totals["batches"] += int(counts["batches_flushed"])
        _safe_update(metrics_proxy, {
            "ingest_last_seq": int(seq),
            "ingest_last_header": {
                "data_mode": hdr.get('data_mode', 'line'),
                "frame_shape": tuple(hdr.get('frame_shape', [])),
                "dtype": hdr.get('dtype'),
            },
            "ingest_segments_ingested": totals["segments"],
            "ingest_frames_ingested": totals["frames"],
            "ingest_bytes_read": totals["bytes"],
            "ingest_batches_flushed": totals["batches"],
            "ingest_updated_unix": time.time(),
        })

    def _on_rejected(seq, path, error):
        totals["rejected"] += 1
        _safe_update(metrics_proxy, {
            "ingest_last_seq": int(seq),
            "ingest_segments_rejected": totals["rejected"],
            "ingest_last_error": f"{os.path.abspath(path)}: {error}",
        })

    def _publish_rate(force=False):
        now = time.monotonic()
        dt = now - rate["t"]
        if force or dt >= 1.0:
            _safe_update(metrics_proxy, {
                "ingest_fps_estimate": float((totals["frames"] - rate["frames"]) / max(1e-6, dt)),
                "ingest_pending_segments": len(sealed_segments(stream_dir)),
                "ingest_alive": True,
                "ingest_updated_unix": time.time(),
            })
            rate["frames"], rate["t"] = totals["frames"], now

    _safe_update(metrics_proxy, {
        "ingest_alive": True,
        "ingest_started": True,
        "ingest_stream_dir": os.path.abspath(stream_dir),
        "ingest_sqlite_path": os.path.abspath(sqlite_path),
        "ingest_segments_ingested": 0,
        "ingest_frames_ingested": 0,
        "ingest_bytes_read": 0,
        "ingest_batches_flushed": 0,
        "ingest_fps_estimate": 0.0,
        "ingest_updated_unix": time.time(),
    })
    parent = multiprocessing.parent_process()
    try:
        if precreate_sqlite:
            try:
                if (data_mode_hint or '').lower() == 'image':
                    if frame_shape_hint is not None and dtype_hint is not None:
                        _ensure_sqlite_keys_image(sqlite_path, tuple(frame_shape_hint), np.dtype(dtype_hint))
                else:
                    _ensure_sqlite_keys_line(sqlite_path, channel_keys, np.dtype(dtype_hint or np.float32))
            except Exception:
                pass
        if ready_event is not None:
            ready_event.set()

        while True:
            stopping = (stop_event is not None and stop_event.is_set()) or \
                       (parent is not None and not parent.is_alive())
            processed = ingest_pending_segments(stream_dir, sqlite_path, channel_keys, batch_frames,
                                                on_ingested=_on_ingested, on_rejected=_on_rejected)
            if processed:
                _publish_rate(force=True)
                continue
            if stopping:
                break  # a full pass after the stop request found nothing left to ingest
            _publish_rate()
            time.sleep(sleep_s)
    except Exception as e:
        _safe_update(metrics_proxy, {
            "ingest_alive": False,
            "ingest_last_error": f"{e.__class__.__name__}: {e}",
            "ingest_last_traceback": ''.join(traceback.format_exc())[-2000:],
            "ingest_updated_unix": time.time(),
        })
    finally:
        _safe_update(metrics_proxy, {"ingest_alive": False, "ingest_updated_unix": time.time()})
