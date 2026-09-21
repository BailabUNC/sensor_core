import os, re, json, struct, time, uuid, traceback
import multiprocessing
from typing import List, Tuple, Optional
import numpy as np
from .ring_adapter import RingBuffer
from sensor_core.utils.utils import clock_anchor

MAGIC = b'SCBIN\x00\x00'
# 2: one record per frame (a whole acquisition in line mode), global frame indices
# 3: records carry each frame's acquisition time; headers name the channels and the session
VERSION = 3

SEGMENT_RE = re.compile(r"^stream_(\d+)\.bin$")
PART_RE = re.compile(r"^stream_(\d+)\.bin\.part$")


def segment_path(stream_dir: str, seq: int) -> str:
    """Path of sealed segment number seq in stream_dir"""
    return os.path.join(stream_dir, f"stream_{seq:06d}.bin")


def sealed_segments(stream_dir: str) -> List[Tuple[int, str]]:
    """Sealed segments in stream_dir, oldest first, as (sequence number, path)"""
    try:
        names = os.listdir(stream_dir)
    except FileNotFoundError:
        return []
    found = [(int(m.group(1)), os.path.join(stream_dir, n)) for n in names if (m := SEGMENT_RE.match(n))]
    return sorted(found)


def new_session() -> dict:
    """Identity and clock anchor for a session that stores frames"""
    unix_ns, monotonic_ns = clock_anchor()
    return {"uuid": uuid.uuid4().hex, "started_unix_ns": unix_ns, "started_monotonic_ns": monotonic_ns}


def _has_records(path: str) -> bool:
    """True if the segment at path holds anything after its header"""
    prefix = len(MAGIC) + 6
    try:
        with open(path, 'rb') as fh:
            head = fh.read(prefix)
        if len(head) < prefix:
            return False
        (length,) = struct.unpack('<I', head[len(MAGIC) + 2:])
        return os.path.getsize(path) > prefix + length
    except OSError:
        return False


def _safe_update(proxy, values: dict):
    """Update a metrics proxy, ignoring failures (e.g., the metrics server has already shut down)"""
    if proxy is None:
        return
    try:
        proxy.update(values)
    except Exception:
        pass


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
    def __init__(self, stream_dir: str, ring_name: str, capacity_frames: int,
                 frame_shape: Tuple[int, ...], dtype, data_mode: str = 'line',
                 rotate_frames: int = 8192, rotate_seconds: Optional[float] = None,
                 metrics_proxy: Optional[dict] = None, channels: Optional[List[str]] = None,
                 session: Optional[dict] = None, rotate_bytes: Optional[int] = 256 * 2**20):
        """
        Append-only binary logger that writes numbered segment files
        Frames go to the active segment, stream_NNNNNN.bin.part. Sealing a segment renames it to
        stream_NNNNNN.bin, which the ingester reads and deletes; sealed segments are never modified.
        :param stream_dir: directory for segment files
        :param ring_name: name of the ring buffer being logged
        :param capacity_frames: capacity of the ring buffer in frames
        :param frame_shape: logical shape of frame
        :param dtype: data type
        :param data_mode: line or image data
        :param rotate_frames: seal the active segment after this many frames
        :param rotate_bytes: or before it grows past this many bytes, so large frames (images) still make
                             segments of a bounded size, each stored in one transaction
        :param rotate_seconds: seal the active segment after this many seconds
        :param metrics_proxy: metrics proxy for timing analysis
        :param channels: channel names, stored in each segment's header
        :param session: session identity and clock anchor (see new_session); a new one by default
        """
        self.stream_dir = os.path.abspath(stream_dir)
        self.ring_name = ring_name
        self.capacity_frames = int(capacity_frames)
        self.frame_shape = tuple(frame_shape)
        self.dtype = np.dtype(dtype)
        self.data_mode = data_mode
        self.rotate_frames = int(rotate_frames)
        self.rotate_bytes = int(rotate_bytes) if rotate_bytes else None
        self._bytes_written_in_active = 0
        self.rotate_seconds = float(rotate_seconds) if rotate_seconds else None
        self.session = dict(session) if session else new_session()
        if channels is None:
            channels = [f"ch{i}" for i in range(self.frame_shape[2])] if data_mode == 'line' else ['image']
        self.channels = [str(c) for c in channels]
        self._fh = None
        self._part = None
        self._frames_written_in_active = 0

        # proxies
        self._metrics = metrics_proxy

        # writer counters
        self._m_total_frames = 0
        self._m_total_bytes = 0
        self._m_rotations = 0
        self._m_dropped_frames = 0
        self.sealed_seq = 0  # sequence number of the most recently sealed segment

        # timers
        self._m_last_flush = time.monotonic()
        self._m_frames_since = 0
        self._last_rotation_wall = time.time()
        self._last_heartbeat = 0.0

        os.makedirs(self.stream_dir, exist_ok=True)
        self._seq = self._recover_interrupted_segments()
        self._open_segment()
        self._publish_heartbeat(force=True)

    def _recover_interrupted_segments(self) -> int:
        """Seal segments left active by an interrupted session; return the highest sequence number in use"""
        highest = 0
        for name in os.listdir(self.stream_dir):
            m = SEGMENT_RE.match(name) or PART_RE.match(name)
            if not m:
                continue
            seq = int(m.group(1))
            highest = max(highest, seq)
            if name.endswith(".part"):
                path = os.path.join(self.stream_dir, name)
                if _has_records(path):
                    os.replace(path, segment_path(self.stream_dir, seq))  # the ingester skips a partial last record
                else:
                    os.remove(path)
        return highest

    def _open_segment(self):
        self._seq += 1
        self._part = segment_path(self.stream_dir, self._seq) + ".part"
        self._fh = open(self._part, 'wb')
        self._write_header(self._fh)
        self._frames_written_in_active = 0
        self._bytes_written_in_active = 0

    def _seal_segment(self, fsync: bool = False):
        """Close the active segment; seal it if it holds frames, otherwise delete it

        Segments are only a buffer: the database is the durable copy, and after a crashed process the
        operating system still holds their data. So only the last segment, at close(), is synced to disk.
        """
        if self._fh is None:
            return
        self._fh.flush()
        if fsync:
            os.fsync(self._fh.fileno())
        self._fh.close()
        self._fh = None
        if self._frames_written_in_active > 0:
            os.replace(self._part, segment_path(self.stream_dir, self._seq))
            self.sealed_seq = self._seq
        else:
            os.remove(self._part)

    def _write_header(self, fh):
        header = {
            'ring_name': self.ring_name,
            'frame_shape': self.frame_shape,
            'dtype': str(self.dtype),
            'data_mode': self.data_mode,
            'version': VERSION,
            'channels': self.channels,
            'session': self.session,
        }
        payload = json.dumps(header).encode('utf-8')
        fh.write(MAGIC)
        fh.write(struct.pack('<H', VERSION))
        fh.write(struct.pack('<I', len(payload)))
        fh.write(payload)

    def _publish_heartbeat(self, force=False):
        if self._metrics is None:
            return
        now = time.time()
        if force or (now - self._last_heartbeat) >= 1.0:
            dt = max(1e-6, time.monotonic() - self._m_last_flush)
            fps = self._m_frames_since / dt
            _safe_update(self._metrics, {
                "writer_stream_dir": self.stream_dir,
                "writer_active_segment": self._part,
                "writer_sealed_seq": int(self.sealed_seq),
                "writer_total_frames": int(self._m_total_frames),
                "writer_total_bytes": int(self._m_total_bytes),
                "writer_rotations": int(self._m_rotations),
                "writer_dropped_frames": int(self._m_dropped_frames),
                "writer_fps_estimate": float(fps),
                "writer_last_rotation_unix": self._last_rotation_wall,
                "writer_updated_unix": now,
                "writer_alive": True,
            })
            self._m_last_flush = time.monotonic()
            self._m_frames_since = 0
            self._last_heartbeat = now

    def rotate(self):
        """Seal the active segment (if it holds frames) and start a new one"""
        self._seal_segment()
        self._open_segment()
        self._m_rotations += 1
        self._last_rotation_wall = time.time()
        self._publish_heartbeat(force=True)

    def _maybe_time_rotate(self):
        if self.rotate_seconds is None:
            return
        if (time.time() - self._last_rotation_wall) >= self.rotate_seconds:
            if self._frames_written_in_active > 0:
                self.rotate()
            else:
                self._last_rotation_wall = time.time()

    def room(self, frame_bytes: int) -> int:
        """Frames that still fit in the active segment (at least one)"""
        room = self.rotate_frames - self._frames_written_in_active
        if self.rotate_bytes is not None:
            room = min(room, (self.rotate_bytes - self._bytes_written_in_active) // (frame_bytes + 16))
        return max(1, room)

    def maybe_rotate(self, frame_bytes: int):
        """Seal the active segment if it is full or has been open for rotate_seconds"""
        full = self._frames_written_in_active >= self.rotate_frames or (
            self.rotate_bytes is not None and self._bytes_written_in_active + frame_bytes + 16 > self.rotate_bytes)
        if full:
            self.rotate()
        else:
            self._maybe_time_rotate()

    def mark(self):
        """Position in the active segment, for rollback()"""
        return self._fh.tell(), self._frames_written_in_active

    def rollback(self, mark):
        """Remove every record written to the active segment since mark()"""
        position, frames = mark
        removed = self._frames_written_in_active - frames
        self._fh.seek(position)
        self._fh.truncate()
        record_bytes = (self._bytes_written_in_active // self._frames_written_in_active) if self._frames_written_in_active else 0
        self._frames_written_in_active = frames
        self._bytes_written_in_active -= removed * record_bytes
        self._m_total_frames -= removed
        self._m_total_bytes -= removed * record_bytes
        self._m_frames_since = max(0, self._m_frames_since - removed)

    def note_dropped(self, nframes: int):
        """Record frames that were overwritten in the ring before they could be written"""
        self._m_dropped_frames += int(nframes)
        self._publish_heartbeat(force=True)

    def close(self):
        """Seal the active segment so it can be ingested; the writer cannot be used afterwards"""
        self._seal_segment(fsync=True)
        self._publish_heartbeat(force=True)

    def write_frames(self, buf: memoryview, frame_bytes: int, first_index: int, nframes: int, ts_ns,
                     rotate: bool = True):
        """
        Append frames to the active segment, one record per frame
        :param buf: bytes of nframes consecutive frames
        :param frame_bytes: size of one frame in bytes
        :param first_index: global index of the first frame (the ring's write index when it was published)
        :param nframes: number of frames in buf
        :param ts_ns: acquisition time of each frame (time.perf_counter_ns), or one time for all of them
        :param rotate: seal segments as they fill; if False, every frame goes to the active segment (see room())
                       and the caller rotates with maybe_rotate()
        """
        if nframes <= 0:
            self._maybe_time_rotate()
            self._publish_heartbeat(force=False)
            return

        b = _contiguous_bytes_view(memoryview(buf))
        stamps = np.broadcast_to(np.asarray(ts_ns, dtype=np.uint64), (nframes,))
        record = frame_bytes + 16
        remaining = nframes
        idx = 0
        while remaining > 0:
            room = self.rotate_frames - self._frames_written_in_active
            if self.rotate_bytes is not None:
                room = min(room, (self.rotate_bytes - self._bytes_written_in_active) // record)
            can_write = min(remaining, max(1, room)) if rotate else remaining  # an oversized frame gets its own segment
            for i in range(can_write):
                off = (idx + i) * frame_bytes
                self._fh.write(struct.pack('<QQ', int(stamps[idx + i]), first_index + idx + i))
                self._fh.write(b[off:off+frame_bytes])
            self._frames_written_in_active += can_write
            self._bytes_written_in_active += can_write * record
            self._m_total_frames += can_write
            self._m_total_bytes += (can_write * (frame_bytes + 16))
            self._m_frames_since += can_write

            idx += can_write
            remaining -= can_write

            full = self._frames_written_in_active >= self.rotate_frames or (
                self.rotate_bytes is not None and self._bytes_written_in_active + record > self.rotate_bytes)
            if full and rotate:
                self.rotate()

        if rotate:
            self._maybe_time_rotate()
        self._publish_heartbeat(force=False)


def drain_ring(ring: RingBuffer, writer: BinaryStreamWriter, last_idx: int) -> Tuple[int, int]:
    """
    Write every frame published since last_idx to the active segment

    Frames are copied straight from shared memory. If the producer overwrote a frame before or while it was
    copied (the writer fell almost a whole ring behind), that frame is dropped rather than stored: the copy is
    removed from the segment and counted, and writing resumes from the oldest frame that is still intact.
    :return: (next_idx, dropped) -- the index to continue from, and how many frames were dropped
             (also reported in metrics)
    """
    wi = int(ring.write_idx)
    if wi <= last_idx:
        return wi, 0  # nothing new (or the ring was recreated, so start over from its index)
    cap = ring.capacity
    frame_bytes = ring.frame_bytes
    margin = 1  # the producer publishes one frame at a time, so at most the oldest slot is being rewritten
    idx = max(last_idx, wi - cap + margin)  # anything older is overwritten, or about to be
    dropped = idx - last_idx
    while idx < wi:
        n = min(wi - idx, cap - idx % cap, writer.room(frame_bytes))  # stop at the ring's end or a full segment
        mark = writer.mark()
        writer.write_frames(ring.view_window_bytes(idx, n), frame_bytes, idx, n, ring.view_timestamps(idx, n),
                            rotate=False)
        intact_from = int(ring.write_idx) - cap + margin  # older frames may have changed during the copy
        if idx < intact_from:
            writer.rollback(mark)
            dropped += intact_from - idx
            idx = intact_from
            continue
        writer.maybe_rotate(frame_bytes)
        idx += n
    if dropped:
        writer.note_dropped(dropped)
    return max(wi, idx), dropped


def dump_loop(stream_dir: str, shm_name: str, capacity_frames: int,
              frame_shape: Tuple[int, ...], dtype, data_mode: str = 'line',
              poll_hz: float = 400.0, rotate_frames: int = 8192,
              rotate_seconds: Optional[float] = None, metrics_proxy: Optional[dict] = None,
              stop_event=None, seal_event=None, ready_event=None, start_idx: int = 0,
              channels: Optional[List[str]] = None, session: Optional[dict] = None,
              rotate_bytes: Optional[int] = 256 * 2**20):
    """
    Writer process: copy frames from the ring buffer into segment files until stop_event is set
    :param stop_event: when set, write everything published so far, seal the last segment, and return
    :param seal_event: when set, write everything published so far and seal the active segment
    :param ready_event: set once the writer is attached to the ring and can accept frames
    :param start_idx: index of the first frame to write; 0 writes everything the ring has held since it was
                      created, including frames published before this process started
    :param channels: channel names, stored with the data
    :param session: session identity and clock anchor (see new_session)
    :param rotate_bytes: seal the active segment before it grows past this many bytes
    """
    _safe_update(metrics_proxy, {
            "writer_alive": True,
            "writer_start_unix": time.time(),
            "writer_ring": {"name": shm_name, "capacity": int(capacity_frames), "shape": tuple(frame_shape), "dtype": str(np.dtype(dtype))},
        })
    parent = multiprocessing.parent_process()
    writer = None
    try:
        ring = RingBuffer(shm_name, capacity_frames, frame_shape, data_mode, dtype, create=False)
        writer = BinaryStreamWriter(stream_dir, shm_name, capacity_frames, frame_shape, dtype,
                                    data_mode=data_mode, rotate_frames=rotate_frames,
                                    rotate_seconds=rotate_seconds, metrics_proxy=metrics_proxy,
                                    channels=channels, session=session, rotate_bytes=rotate_bytes)
        last_idx = int(start_idx)
        period = 1.0 / poll_hz
        seals = 0
        if ready_event is not None:
            ready_event.set()
        next_parent_check = 0.0

        while stop_event is None or not stop_event.is_set():
            next_idx, _ = drain_ring(ring, writer, last_idx)
            if next_idx == last_idx:
                # idle: still check time-based rotation and publish a heartbeat
                writer.write_frames(memoryview(b""), ring.frame_bytes, next_idx, 0, 0)
            last_idx = next_idx

            if seal_event is not None and seal_event.is_set():
                seal_event.clear()
                last_idx, _ = drain_ring(ring, writer, last_idx)
                writer.rotate()
                seals += 1
                _safe_update(metrics_proxy, {"writer_seal_count": seals, "writer_seal_idx": int(last_idx),
                                             "writer_sealed_seq": int(writer.sealed_seq)})

            now = time.monotonic()
            if parent is not None and now >= next_parent_check:
                if not parent.is_alive():
                    break  # the process that owns the ring is gone; finish up and exit
                next_parent_check = now + 0.5
            time.sleep(period)

        drain_ring(ring, writer, last_idx)
    except Exception as e:
        _safe_update(metrics_proxy, {
            "writer_alive": False,
            "writer_last_error": f"{e.__class__.__name__}: {e}",
            "writer_last_traceback": ''.join(traceback.format_exc())[-2000:],  # tail to keep small
            "writer_updated_unix": time.time(),
        })
    finally:
        if writer is not None:
            writer.close()
        _safe_update(metrics_proxy, {"writer_alive": False, "writer_updated_unix": time.time()})
