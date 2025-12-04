import os, json, struct, time, traceback
from typing import List, Optional, Tuple
import numpy as np
from sqlitedict import SqliteDict
from .strg_manager import StorageManager

MAGIC = b'SCBIN\x00\x00'
MAGIC_LEN = len(MAGIC)
REC_HEADER_SZ = 16

def _seal_path(p: str) -> str: return p + ".seal"

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
                      batch_frames: int, dtype: np.dtype, C: int, S: int,
                      metrics_accum: dict):
    _ensure_sqlite_keys_line(sqlite_path, channel_keys, dtype)
    sm = StorageManager(channel_key=channel_keys, filepath=sqlite_path, overwrite=False)
    acc = {k: [] for k in channel_keys}
    frames = 0; bytes_read = 0; batches = 0
    with open(path, 'rb') as fh:
        ver, hdr, ver_b, len_b, payload = _read_header(fh)
        while True:
            rec = fh.read(REC_HEADER_SZ)
            if not rec: break
            ts_ns, wi = struct.unpack('<QQ', rec)
            raw = fh.read(S * C * dtype.itemsize)
            if len(raw) < S * C * dtype.itemsize:
                break
            arr = np.frombuffer(raw, dtype=dtype, count=C*S).reshape(C, S)
            for ci, key in enumerate(channel_keys):
                acc[key].append(arr[ci])
            frames += 1
            bytes_read += REC_HEADER_SZ + len(raw)
            if sum(len(v) for v in acc.values()) >= batch_frames:
                for key in channel_keys:
                    if acc[key]:
                        block = np.concatenate(acc[key], axis=0)
                        sm.append_serial_channel(key, block)
                        acc[key].clear()
                batches += 1
    for key in channel_keys:
        if acc[key]:
            block = np.concatenate(acc[key], axis=0)
            sm.append_serial_channel(key, block)
            acc[key].clear()
    metrics_accum["frames_ingested"] = metrics_accum.get("frames_ingested", 0) + frames
    metrics_accum["bytes_read"] = metrics_accum.get("bytes_read", 0) + bytes_read
    metrics_accum["batches_flushed"] = metrics_accum.get("batches_flushed", 0) + batches
    return ver_b, len_b, payload

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
            if not rec: break
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

def ingest_loop(file_a: str, file_b: str, sqlite_path: str, channel_keys: List[str],
                batch_frames: int = 32, sleep_s: float = 0.2,
                metrics_proxy: Optional[dict] = None,
                data_mode_hint: Optional[str] = None,
                frame_shape_hint: Optional[Tuple[int, ...]] = None,
                dtype_hint: Optional[str] = None,
                precreate_sqlite: bool = True):

    try:
        # initialize metrics immediately
        if metrics_proxy is not None:
            metrics_proxy.update({
                "ingest_bins_ingested": 0,
                "ingest_frames_ingested": 0,
                "ingest_bytes_read": 0,
                "ingest_batches_flushed": 0,
                "ingest_fps_estimate": 0.0,
                "ingest_updated_unix": time.time(),
                "ingest_alive": True,
                "ingest_watch_paths": [os.path.abspath(file_a), os.path.abspath(file_b)],
                "ingest_sqlite_path": os.path.abspath(sqlite_path),
                "ingest_started": True,
            })
    except Exception:
        pass

    try:
        if metrics_proxy is not None:
            metrics_proxy.update({
                "ingest_alive": True,
                "ingest_starting": False
            })

        def _publish_scan(seal_a: str, seal_b: str):
            if metrics_proxy is None:
                return

            exists_a = os.path.exists(seal_a);
            exists_b = os.path.exists(seal_b)
            mt_a = os.path.getmtime(seal_a) if exists_a else None
            mt_b = os.path.getmtime(seal_b) if exists_b else None
            metrics_proxy.update({
                "ingest_scan_seals": [seal_a, seal_b],
                "ingest_scan_exists": [exists_a, exists_b],
                "ingest_scan_mtime": [mt_a, mt_b],
                "ingest_alive": True,
                "ingest_updated_unix": time.time(),
            })

        if precreate_sqlite:
            try:
                if (data_mode_hint or '').lower() == 'image':
                    if frame_shape_hint is not None and dtype_hint is not None:
                        _ensure_sqlite_keys_image(sqlite_path, tuple(frame_shape_hint), np.dtype(dtype_hint))
                else:
                    _ensure_sqlite_keys_line(sqlite_path, channel_keys, np.dtype(dtype_hint or np.float32))
            except Exception:
                pass

        last_frames_total = 0
        last_t = time.monotonic()

        def _metrics_flush(force=False):
            nonlocal last_frames_total, last_t
            if metrics_proxy is None:
                return
            now = time.monotonic()
            dt = now - last_t
            if force or dt >= 1.0:
                frames_total = int(metrics_proxy.get("ingest_frames_ingested", 0))
                fps = (frames_total - last_frames_total) / max(1e-6, dt)
                metrics_proxy.update({
                    "ingest_fps_estimate": float(fps),
                    "ingest_updated_unix": time.time(),
                    "ingest_alive": True,
                    "ingest_starting": False,
                })
                last_frames_total = frames_total
                last_t = now

        files = [file_a, file_b]
        while True:
            did_work = False
            seal_a = _seal_path(file_a)
            seal_b = _seal_path(file_b)
            _publish_scan(seal_a, seal_b)
            for path in files:
                seal = _seal_path(path)
                if os.path.exists(seal):
                    with open(path, 'rb') as fh:
                        try:
                            ver, hdr, ver_b, len_b, payload = _read_header(fh)
                            if metrics_proxy is not None:
                                metrics_proxy.update({
                                    "ingest_last_header": {
                                        "path": os.path.abspath(path),
                                        "data_mode": hdr.get('data_mode', 'line'),
                                        "frame_shape": tuple(hdr.get('frame_shape', [])),
                                        "dtype": hdr.get('dtype'),
                                    }
                                })
                        except ValueError as e:
                            if metrics_proxy is not None:
                                metrics_proxy.update({
                                    "ingest_last_error": f"HeaderError on {os.path.abspath(path)}: {e}",
                                })
                            continue
                    dtype = np.dtype(hdr['dtype'])
                    shape = tuple(hdr['frame_shape'])
                    mode = hdr.get('data_mode', 'line')

                    delta = {"frames_ingested": 0, "bytes_read": 0, "batches_flushed": 0}
                    if mode == 'line':
                        N, _, C = shape
                        _ = _ingest_file_line(path, sqlite_path, channel_keys, batch_frames, dtype, N, C, delta)
                    elif mode == 'image':
                        H, W, Cimg = shape
                        _ = _ingest_file_image(path, sqlite_path, (H, W, Cimg), batch_frames, dtype, delta)
                    else:
                        continue

                    if metrics_proxy is not None:
                        metrics_proxy.update({
                            "ingest_bins_ingested": int(metrics_proxy["ingest_bins_ingested"]) + 1,
                            "ingest_frames_ingested": int(metrics_proxy["ingest_frames_ingested"]) + int(delta["frames_ingested"]),
                            "ingest_bytes_read": int(metrics_proxy["ingest_bytes_read"]) + int(delta["bytes_read"]),
                            "ingest_batches_flushed": int(metrics_proxy["ingest_batches_flushed"]) + int(delta["batches_flushed"]),
                        })
                        _metrics_flush(force=True)

                    # truncate back to header
                    try: os.remove(seal)
                    except FileNotFoundError: pass
                    with open(path, 'wb') as out:
                        out.write(MAGIC); out.write(ver_b); out.write(len_b); out.write(payload)
                    did_work = True

            if not did_work:
                time.sleep(sleep_s)
            _metrics_flush(force=False)
    except Exception as e:
        if metrics_proxy is not None:
            metrics_proxy.update({
                "ingest_alive": False,
                "ingest_last_error": f"{e.__class__.__name__}: {e}",
                "ingest_last_traceback": ''.join(traceback.format_exc())[-2000:],
                "ingest_updated_unix": time.time(),
                "ingest_starting": False,
            })
