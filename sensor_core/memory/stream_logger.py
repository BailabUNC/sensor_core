import os, json, struct, time, traceback
from typing import Tuple, Optional
import numpy as np
from .ring_adapter import RingBuffer

MAGIC = b'SCBIN\x00\x00'
VERSION = 1

def _ensure_parent(path: str):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)

def _seal_path(path: str) -> str:
    return path + ".seal"

def _contiguous_bytes_view(mv: memoryview) -> memoryview:
    try:
        b = mv.cast('B')
        is_c = getattr(b, 'c_contiguous', None)
        if is_c is None:
            is_c = getattr(b, 'contiguous', True)
        if is_c:
            return b
    except TypeError:
        pass
    return memoryview(bytes(mv))

class BinaryStreamWriter:
    def __init__(self, file_a: str, file_b: str, ring_name: str, capacity_frames: int,
                 frame_shape: Tuple[int, ...], dtype, data_mode: str = 'line',
                 rotate_frames: int = 8192, rotate_seconds: Optional[float] = None,
                 overwrite: bool = False, metrics_proxy: Optional[dict] = None,
                 control_proxy: Optional[dict] = None):
        """
        Append-only binary logger to two alternating files with seal markers
        :param file_a: location of .bin file a
        :param file_b: location of .bin file b
        :param ring_name: location of ring buffer
        :param capacity_frames: max capacity of .bin files
        :param frame_shape: logical shape of frame
        :param dtype: data type
        :param data_mode: line or image data
        :param rotate_frames: how many frames to save before rotating to alternate .bin file
        :param rotate_seconds: how many seconds to wait before forcing switch to alternate .bin file
        :param overwrite: flag to overwrite existing bin and sqlite file
        :param metrics_proxy: metrics proxy for timing analysis
        :param control_proxy: contains flag to force switch between .bin files
        """
        self.files = [file_a, file_b]
        self.ring_name = ring_name
        self.capacity_frames = int(capacity_frames)
        self.frame_shape = tuple(frame_shape)
        self.dtype = np.dtype(dtype)
        self.data_mode = data_mode
        self.rotate_frames = int(rotate_frames)
        self.rotate_seconds = float(rotate_seconds) if rotate_seconds else None
        self._active = 0
        self._frames_written_in_active = 0
        self._fh = None

        # proxies
        self._metrics = metrics_proxy
        self._control = control_proxy

        # writer counters
        self._m_total_frames = 0
        self._m_total_bytes = 0
        self._m_rotations = 0

        # timers
        self._m_last_flush = time.monotonic()
        self._m_frames_since = 0
        self._last_rotation_wall = time.time()
        self._last_heartbeat = 0.0

        # Setup bin files
        _ensure_parent(file_a); _ensure_parent(file_b)
        if overwrite:
            for f in self.files:
                try: os.remove(_seal_path(f))
                except FileNotFoundError: pass
                with open(f, 'wb') as fh: self._write_header(fh)

        for f in self.files:
            if not os.path.exists(f) or os.path.getsize(f) == 0:
                with open(f, 'wb') as fh: self._write_header(fh)

        if os.path.exists(_seal_path(self.files[self._active])):
            self._active = 1 - self._active
        if os.path.exists(_seal_path(self.files[self._active])):
            os.remove(_seal_path(self.files[self._active]))

        self._fh = open(self.files[self._active], 'ab')
        self._frames_written_in_active = 0
        self._publish_heartbeat(force=True)

    def _write_header(self, fh):
        header = {
            'ring_name': self.ring_name,
            'frame_shape': self.frame_shape,
            'dtype': str(self.dtype),
            'data_mode': self.data_mode,
            'version': VERSION,
        }
        payload = json.dumps(header).encode('utf-8')
        fh.write(MAGIC)
        fh.write(struct.pack('<H', VERSION))
        fh.write(struct.pack('<I', len(payload)))
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())

    def _publish_heartbeat(self, force=False):
        if self._metrics is None:
            return
        now = time.time()
        if force or (now - self._last_heartbeat) >= 1.0:
            seals = [os.path.abspath(self.files[0]) + ".seal", os.path.abspath(self.files[1]) + ".seal"]
            seal_exists = [os.path.exists(seals[0]), os.path.exists(seals[1])]
            seal_mtime = [
                (os.path.getmtime(seals[0]) if seal_exists[0] else None),
                (os.path.getmtime(seals[1]) if seal_exists[1] else None),
            ]
            dt = max(1e-6, time.monotonic() - self._m_last_flush)
            fps = self._m_frames_since / dt
            self._metrics.update({
                "writer_active_bin": os.path.abspath(self.files[self._active]),
                "writer_total_frames": int(self._m_total_frames),
                "writer_total_bytes": int(self._m_total_bytes),
                "writer_rotations": int(self._m_rotations),
                "writer_fps_estimate": float(fps),
                "writer_last_rotation_unix": self._last_rotation_wall,
                "writer_updated_unix": now,
                "writer_alive": True,
                "writer_watch_seals": seals,
                "writer_seal_exists": seal_exists,
                "writer_seal_mtime": seal_mtime,
            })
            self._m_last_flush = time.monotonic()
            self._m_frames_since = 0
            self._last_heartbeat = now

    def _rotate(self):
        self._fh.flush(); os.fsync(self._fh.fileno()); self._fh.close()
        open(_seal_path(self.files[self._active]), 'wb').close()
        self._active = 1 - self._active
        try: os.remove(_seal_path(self.files[self._active]))
        except FileNotFoundError: pass
        with open(self.files[self._active], 'wb') as fh: self._write_header(fh)
        self._fh = open(self.files[self._active], 'ab')
        self._frames_written_in_active = 0
        self._m_rotations += 1
        self._last_rotation_wall = time.time()
        self._publish_heartbeat(force=True)

    def _maybe_time_rotate(self):
        if self.rotate_seconds is None:
            return
        if (time.time() - self._last_rotation_wall) >= self.rotate_seconds:
            self._fh.flush()
            self._rotate()

    def _maybe_force_rotate(self):
        if self._control is None:
            return
        if bool(self._control.get("force_rotate", False)):
            self._control.update({"force_rotate": False})
            self._fh.flush()
            self._rotate()

    def write_frames(self, buf: memoryview, frame_bytes: int, start_idx: int, nframes: int, ts_ns: int):
        if nframes <= 0:
            self._maybe_time_rotate()
            self._maybe_force_rotate()
            self._publish_heartbeat(force=False)
            return

        b = _contiguous_bytes_view(memoryview(buf))
        remaining = nframes
        idx = 0
        while remaining > 0:
            can_write = min(remaining, max(1, self.rotate_frames - self._frames_written_in_active))
            for i in range(can_write):
                off = (idx + i) * frame_bytes
                self._fh.write(struct.pack('<QQ', ts_ns, (start_idx + idx + i)))
                self._fh.write(b[off:off+frame_bytes])
            self._frames_written_in_active += can_write
            self._m_total_frames += can_write
            self._m_total_bytes += (can_write * (frame_bytes + 16))
            self._m_frames_since += can_write

            idx += can_write
            remaining -= can_write

            if self._frames_written_in_active >= self.rotate_frames:
                self._fh.flush()
                self._rotate()

        self._maybe_time_rotate()
        self._maybe_force_rotate()
        self._publish_heartbeat(force=False)

def dump_loop(file_a: str, file_b: str, shm_name: str, capacity_frames: int,
              frame_shape: Tuple[int, ...], dtype, data_mode: str = 'line',
              poll_hz: float = 400.0, overwrite: bool = False, rotate_frames: int = 8192,
              rotate_seconds: Optional[float] = None, metrics_proxy: Optional[dict] = None,
              control_proxy: Optional[dict] = None):
    if metrics_proxy is not None:
        metrics_proxy.update({
            "writer_alive": True,
            "writer_start_unix": time.time(),
            "writer_watch_bins": [os.path.abspath(file_a), os.path.abspath(file_b)],
            "writer_watch_seals": [os.path.abspath(file_a) + ".seal", os.path.abspath(file_b) + ".seal"],
            "writer_ring": {"name": shm_name, "capacity": int(capacity_frames), "shape": tuple(frame_shape), "dtype": str(np.dtype(dtype))},
        })
    try:
        ring = RingBuffer(shm_name, capacity_frames, frame_shape, dtype, create=False)
        writer = BinaryStreamWriter(file_a, file_b, shm_name, capacity_frames, frame_shape, dtype,
                                    data_mode=data_mode, rotate_frames=rotate_frames,
                                    rotate_seconds=rotate_seconds, overwrite=overwrite,
                                    metrics_proxy=metrics_proxy, control_proxy=control_proxy)
        last_idx = int(ring.write_idx)
        frame_bytes = ring.frame_bytes
        period = 1.0 / poll_hz
        cap = int(capacity_frames)

        def _write_window_wrapped(start: int, n: int, ts_ns: int):
            if n <= 0:
                return
            end = (start + n)
            if end <= cap:
                buf = ring.view_window_bytes(start, n)
                writer.write_frames(buf, frame_bytes, start, n, ts_ns)
            else:
                first = cap - start
                second = end - cap
                if first > 0:
                    buf1 = ring.view_window_bytes(start, first)
                    writer.write_frames(buf1, frame_bytes, start, first, ts_ns)
                if second > 0:
                    buf2 = ring.view_window_bytes(0, second)
                    writer.write_frames(buf2, frame_bytes, 0, second, ts_ns)

        while True:
            wi = int(ring.write_idx)
            if wi != last_idx:
                n = (wi - last_idx) % cap
                if n > 0:
                    start = (wi - n) % cap
                    ts_ns = time.time_ns()
                    _write_window_wrapped(start, n, ts_ns)
                    last_idx = wi
            else:
                writer.write_frames(memoryview(b""), frame_bytes, 0, 0, time.time_ns())
            time.sleep(period)
    except Exception as e:
        if metrics_proxy is not None:
            metrics_proxy.update({
                "writer_alive": False,
                "writer_last_error": f"{e.__class__.__name__}: {e}",
                "writer_last_traceback": ''.join(traceback.format_exc())[-2000:],  # tail to keep small
                "writer_updated_unix": time.time(),
            })
