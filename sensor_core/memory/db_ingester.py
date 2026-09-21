import os, json, struct, time, traceback
import multiprocessing
from typing import Optional, Tuple
import numpy as np
from .stream_logger import MAGIC, VERSION as STREAM_VERSION, sealed_segments, _safe_update
from . import strg_manager as storage

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

def _records(fh, record_bytes: int, counter: list):
    """(frame_index, t_ns, frame bytes) of each complete record; stops at a partial last record"""
    while True:
        rec = fh.read(REC_HEADER_SZ)
        if len(rec) < REC_HEADER_SZ:
            return
        t_ns, index = struct.unpack('<QQ', rec)
        data = fh.read(record_bytes)
        if len(data) < record_bytes:
            return  # left by a crash while the record was being written
        counter[0] += 1
        yield index, t_ns, data

def _describe(hdr: dict, path: str):
    """(data_mode, frame shape, dtype, channels, session) of a segment, checked for consistency"""
    try:
        mode = hdr.get('data_mode', 'line')
        shape = tuple(int(x) for x in hdr['frame_shape'])
        dtype = np.dtype(hdr['dtype'])
        session = hdr['session']
        channels = [str(c) for c in hdr['channels']]
    except (KeyError, TypeError) as e:
        raise ValueError(f"{path} has an incomplete header: {e}") from e
    if mode == 'line':
        _, window, n_channels = shape
        shape = (window, n_channels)
        if len(channels) != n_channels:
            raise ValueError(f"{path} names {len(channels)} channels for frames with {n_channels} channels")
    elif mode != 'image':
        raise ValueError(f"{path} has unknown data_mode {mode!r}")
    return mode, shape, dtype, channels, session

def ingest_segment(path: str, database) -> Tuple[dict, dict]:
    """
    Store one sealed segment in the database, then delete it
    All of a segment's frames are stored in one transaction, so a segment is stored completely or not at all;
    frames already stored (for example, from a segment ingested before a crash) are skipped.
    :param database: path of the SQLite database, or a connection from strg_manager.connect
    :return: (header, counts) -- the segment's header, and counts of frames read, frames stored, and bytes read
    :raises ValueError: if the segment was written in another format
    """
    conn = storage.connect(database) if isinstance(database, (str, os.PathLike)) else database
    try:
        with open(path, 'rb') as fh:
            ver, hdr, _, _, _ = _read_header(fh)
            if ver != STREAM_VERSION:
                raise ValueError(f"{path} uses stream format version {ver}; this version of sensor_core reads version {STREAM_VERSION}")
            mode, shape, dtype, channels, session = _describe(hdr, path)
            record_bytes = int(np.prod(shape)) * dtype.itemsize
            read = [0]
            with conn:
                session_id = storage.add_session(conn, session, mode, shape, dtype, channels)
                stored = storage.insert_frames(conn, session_id, _records(fh, record_bytes, read))
    finally:
        if conn is not database:
            conn.close()
    os.remove(path)
    counts = {"frames_read": read[0], "frames_stored": stored, "bytes_read": read[0] * (REC_HEADER_SZ + record_bytes)}
    return hdr, counts


def ingest_pending_segments(stream_dir: str, database, on_ingested=None, on_rejected=None) -> int:
    """
    Store every sealed segment in stream_dir, oldest first
    A segment that cannot be stored is renamed to *.rejected so it neither blocks later segments nor
    is retried forever; its data stays on disk for inspection.
    :param database: path of the SQLite database, or a connection from strg_manager.connect
    :param on_ingested: optional callback(seq, header, counts) after each segment is stored
    :param on_rejected: optional callback(seq, path, error) for each segment set aside
    :return: number of segments processed
    """
    conn = storage.connect(database) if isinstance(database, (str, os.PathLike)) else database
    processed = 0
    try:
        for seq, path in sealed_segments(stream_dir):
            try:
                hdr, counts = ingest_segment(path, conn)
            except ValueError as e:
                os.replace(path, path + ".rejected")
                if on_rejected is not None:
                    on_rejected(seq, path, e)
            else:
                if on_ingested is not None:
                    on_ingested(seq, hdr, counts)
            processed += 1
    finally:
        if conn is not database:
            conn.close()
    return processed


def ingest_loop(stream_dir: str, sqlite_path: str, sleep_s: float = 0.2,
                metrics_proxy: Optional[dict] = None, stop_event=None, ready_event=None):
    """
    Ingester process: move sealed segments from stream_dir into the SQLite database until stop_event is set
    :param stop_event: when set, store every remaining sealed segment and return
    :param ready_event: set once the database is open
    """
    totals = {"segments": 0, "frames": 0, "bytes": 0, "rejected": 0}
    rate = {"frames": 0, "t": time.monotonic()}

    def _on_ingested(seq, hdr, counts):
        totals["segments"] += 1
        totals["frames"] += int(counts["frames_stored"])
        totals["bytes"] += int(counts["bytes_read"])
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
        "ingest_fps_estimate": 0.0,
        "ingest_updated_unix": time.time(),
    })
    parent = multiprocessing.parent_process()
    conn = None
    try:
        conn = storage.connect(sqlite_path)
        if ready_event is not None:
            ready_event.set()

        while True:
            stopping = (stop_event is not None and stop_event.is_set()) or \
                       (parent is not None and not parent.is_alive())
            processed = ingest_pending_segments(stream_dir, conn,
                                                on_ingested=_on_ingested, on_rejected=_on_rejected)
            if processed:
                _publish_rate(force=True)
                continue
            if stopping:
                break  # a full pass after the stop request found nothing left to store
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
        if conn is not None:
            conn.close()
        _safe_update(metrics_proxy, {"ingest_alive": False, "ingest_updated_unix": time.time()})
