"""
SQLite storage for acquired frames

A database holds one or more sessions (one per SensorManager run). The schema is plain SQLite, so any
language with an SQLite library can read it:

  sessions(id, uuid, started_unix_ns, started_monotonic_ns, data_mode, frame_shape, dtype, channels)
  frames(id, session_id, frame_index, t_ns, data)

Each row of `frames` is one frame, stored as raw bytes in row-major order: a whole acquisition shaped
(window_size, channels) in line mode, or one image shaped (height, width, channels) in image mode. The
session's `frame_shape` (JSON), `dtype` (a NumPy dtype string such as '<f4') and `channels` (JSON list of
names) describe how to decode it. `frame_index` counts frames from the start of the session, so a gap
means frames were dropped. `t_ns` is when the frame was acquired, in nanoseconds of the host's monotonic
clock; the wall-clock time of a frame is started_unix_ns + (t_ns - started_monotonic_ns).
"""
import json
import os
import pathlib
import sqlite3
import warnings
from typing import *

import numpy as np

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sensor_core (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    id                   INTEGER PRIMARY KEY,
    uuid                 TEXT NOT NULL UNIQUE,
    started_unix_ns      INTEGER NOT NULL,
    started_monotonic_ns INTEGER NOT NULL,
    data_mode            TEXT NOT NULL,
    frame_shape          TEXT NOT NULL,
    dtype                TEXT NOT NULL,
    channels             TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS frames (
    id          INTEGER PRIMARY KEY,
    session_id  INTEGER NOT NULL REFERENCES sessions(id),
    frame_index INTEGER NOT NULL,
    t_ns        INTEGER NOT NULL,
    data        BLOB NOT NULL,
    UNIQUE (session_id, frame_index)
);
"""

_warned_legacy = set()


def connect(filepath: str) -> sqlite3.Connection:
    """ Open a sensor_core database, creating it (and its tables) if needed
    """
    conn = sqlite3.connect(filepath, timeout=30)
    try:
        conn.execute("PRAGMA journal_mode=WAL")  # readers and the ingester do not block each other
        conn.executescript(_SCHEMA)
        with conn:
            conn.execute("INSERT OR IGNORE INTO sensor_core (key, value) VALUES ('schema_version', ?)",
                         (str(SCHEMA_VERSION),))
        version = int(conn.execute("SELECT value FROM sensor_core WHERE key = 'schema_version'").fetchone()[0])
        if version != SCHEMA_VERSION:
            raise ValueError(f"{filepath} uses sensor_core database schema {version}; "
                             f"this version of sensor_core reads schema {SCHEMA_VERSION}")
        legacy = conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'unnamed'").fetchone()
        if legacy and os.path.abspath(filepath) not in _warned_legacy:
            _warned_legacy.add(os.path.abspath(filepath))
            warnings.warn(f"{filepath} also holds data written by sensor_core 1.x, which this version does not "
                          f"read; only data stored since the upgrade is loaded", UserWarning, stacklevel=3)
    except BaseException:
        conn.close()
        raise
    return conn


def add_session(conn: sqlite3.Connection, session: dict, data_mode: str, frame_shape, dtype, channels) -> int:
    """ Record a session if it is not already stored, and return its id

    :param session: dict with uuid, started_unix_ns, and started_monotonic_ns
    :param frame_shape: shape of one stored frame
    """
    conn.execute("INSERT OR IGNORE INTO sessions (uuid, started_unix_ns, started_monotonic_ns, data_mode, "
                 "frame_shape, dtype, channels) VALUES (?, ?, ?, ?, ?, ?, ?)",
                 (str(session["uuid"]), int(session["started_unix_ns"]), int(session["started_monotonic_ns"]),
                  str(data_mode), json.dumps([int(x) for x in frame_shape]), np.dtype(dtype).str,
                  json.dumps([str(c) for c in channels])))
    return conn.execute("SELECT id FROM sessions WHERE uuid = ?", (str(session["uuid"]),)).fetchone()[0]


def insert_frames(conn: sqlite3.Connection, session_id: int, rows) -> int:
    """ Store frames; frames already stored (same session and frame_index) are skipped

    :param rows: iterable of (frame_index, t_ns, frame bytes)
    :return: number of frames stored
    """
    before = conn.total_changes
    conn.executemany("INSERT OR IGNORE INTO frames (session_id, frame_index, t_ns, data) VALUES (?, ?, ?, ?)",
                     ((session_id, int(index), int(t_ns), data) for index, t_ns, data in rows))
    return conn.total_changes - before


def _open_existing(filepath: str) -> sqlite3.Connection:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"no database at {filepath}")
    return connect(filepath)


def _sessions(conn: sqlite3.Connection, session=None) -> List[dict]:
    """ Stored sessions, oldest first: all of them (session=None), or one, given by id, negative index
    (-1 is the latest), or uuid
    """
    found = [{"id": r[0], "uuid": r[1], "started_unix_ns": r[2], "started_monotonic_ns": r[3],
              "data_mode": r[4], "frame_shape": tuple(json.loads(r[5])), "dtype": np.dtype(r[6]),
              "channels": json.loads(r[7])}
             for r in conn.execute("SELECT id, uuid, started_unix_ns, started_monotonic_ns, data_mode, "
                                   "frame_shape, dtype, channels FROM sessions ORDER BY id")]
    if session is None:
        return found
    if isinstance(session, str):
        match = [s for s in found if s["uuid"] == session]
    elif int(session) < 0:
        match = found[int(session):][:1] if -int(session) <= len(found) else []
    else:
        match = [s for s in found if s["id"] == int(session)]
    if not match:
        raise ValueError(f"no session {session!r} in this database")
    return match


def _frames(conn: sqlite3.Connection, s: dict):
    """(frame_index, t_ns, frames) of one session, ordered by frame_index"""
    index, t_ns, blobs = [], [], []
    for i, t, data in conn.execute("SELECT frame_index, t_ns, data FROM frames WHERE session_id = ? "
                                   "ORDER BY frame_index", (s["id"],)):
        index.append(i)
        t_ns.append(t)
        blobs.append(data)
    frames = np.frombuffer(b"".join(blobs), dtype=s["dtype"]).reshape((len(blobs), *s["frame_shape"])).copy()
    return np.array(index, dtype=np.int64), np.array(t_ns, dtype=np.int64), frames


def list_sessions(filepath: str) -> List[dict]:
    """ Sessions stored in a database, oldest first, with their settings and frame counts
    """
    conn = _open_existing(filepath)
    try:
        counts = dict(conn.execute("SELECT session_id, COUNT(*) FROM frames GROUP BY session_id").fetchall())
        sessions = _sessions(conn)
    finally:
        conn.close()
    for s in sessions:
        s["dtype"] = s["dtype"].str
        s["frames"] = counts.get(s["id"], 0)
    return sessions


def load_frames(filepath: str, session=-1):
    """ All frames of one session
    :param session: session id, negative index (-1 is the latest), or uuid
    :return: (frame_index, t_ns, frames) -- frames shaped (n, window_size, channels) or (n, height, width, channels)
    """
    conn = _open_existing(filepath)
    try:
        (s,) = _sessions(conn, session)
        return _frames(conn, s)
    finally:
        conn.close()


def load_channel(filepath: str, key: str, session=None) -> np.ndarray:
    """ Every sample of one line-mode channel, in acquisition order
    :param session: None for all sessions that recorded this channel, or one session (id, negative index, uuid)
    """
    conn = _open_existing(filepath)
    try:
        parts = []
        for s in _sessions(conn, session):
            if s["data_mode"] == "line" and key in s["channels"]:
                _, _, frames = _frames(conn, s)
                parts.append(frames[:, :, s["channels"].index(key)].reshape(-1))
    finally:
        conn.close()
    if not parts:
        raise ValueError(f"Given key {key} is not in sqlite3 file at {filepath}")
    return np.concatenate(parts)


def load_frame_times(filepath: str, session=None, clock: str = "unix") -> np.ndarray:
    """ Acquisition time of every frame, in nanoseconds
    :param session: None for all sessions, or one session (id, negative index, uuid)
    :param clock: "unix" for wall-clock time since the Unix epoch, or "monotonic" for the host's monotonic
                  clock, which is shared by all processes on the machine and never jumps
    """
    if clock not in ("unix", "monotonic"):
        raise ValueError("clock must be 'unix' or 'monotonic'")
    conn = _open_existing(filepath)
    try:
        parts = []
        for s in _sessions(conn, session):
            t_ns = np.fromiter((r[0] for r in conn.execute(
                "SELECT t_ns FROM frames WHERE session_id = ? ORDER BY frame_index", (s["id"],))), dtype=np.int64)
            if clock == "unix":
                t_ns = t_ns - s["started_monotonic_ns"] + s["started_unix_ns"]
            parts.append(t_ns)
    finally:
        conn.close()
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)


def load_images(filepath: str, session=None) -> np.ndarray:
    """ Every image-mode frame, shaped (n, height, width, channels)
    :param session: None for all image sessions, or one session (id, negative index, uuid)
    """
    conn = _open_existing(filepath)
    try:
        parts = [_frames(conn, s)[2] for s in _sessions(conn, session) if s["data_mode"] == "image"]
    finally:
        conn.close()
    if not parts:
        raise ValueError(f"no image data in sqlite3 file at {filepath}")
    return np.concatenate(parts)


class StorageManager:
    def __init__(self, channel_key: Union[np.ndarray, list, tuple, str],
                 filepath: str = './serial_db.sqlite3', overwrite: bool = False):
        """
        :param channel_key: list/array of channel key names (or single key)
        :param filepath: storage file path (.sqlite3)
        :param overwrite: kept for compatibility; stored data is never overwritten
        """
        self.channel_key = channel_key if isinstance(channel_key, (list, tuple, np.ndarray)) else [channel_key]
        self.filepath = filepath
        self.filetype = pathlib.Path(self.filepath).suffix
        self.overwrite = overwrite
        if self.filetype != ".sqlite3":
            raise ValueError(f"defined filetype {self.filetype} is unsupported. Must use .sqlite3")

    def create_serial_database(self, dtype_map: Dict[str, np.dtype] = None):
        """Create the database file and its tables, if they do not exist yet."""
        connect(self.filepath).close()

    @staticmethod
    def list_sessions(filepath: str = './serial_db.sqlite3') -> List[dict]:
        """Sessions stored in the database, oldest first (see list_sessions)."""
        return list_sessions(filepath)

    @classmethod
    def load_serial_channel(cls, key: str, filepath: str = './serial_db.sqlite3', session=None) -> np.ndarray:
        """Every sample of one line-mode channel (see load_channel)."""
        return load_channel(filepath, key, session)

    @staticmethod
    def load_frame_times(filepath: str = './serial_db.sqlite3', session=None, clock: str = "unix") -> np.ndarray:
        """Acquisition time of every frame, in nanoseconds (see load_frame_times)."""
        return load_frame_times(filepath, session, clock)

    @staticmethod
    def load_frames(filepath: str = './serial_db.sqlite3', session=-1):
        """(frame_index, t_ns, frames) of one session (see load_frames)."""
        return load_frames(filepath, session)

    @staticmethod
    def load_images(filepath: str = './serial_db.sqlite3', session=None) -> np.ndarray:
        """Every image-mode frame (see load_images)."""
        return load_images(filepath, session)
