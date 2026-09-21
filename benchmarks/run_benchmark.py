"""Measure sensor_core's throughput for line or image data.

A simulated sensor publishes frames at a fixed rate (or as fast as it can). The benchmark counts, over
time, how many frames were acquired, written to disk, and stored in SQLite, and optionally how many
frames of the live plot were drawn. Afterwards it checks that every stored frame arrived intact and in
order. Results, with the settings and a description of the machine, are saved as JSON; draw them with
plot_results.py.

    python benchmarks/run_benchmark.py line --rate 5000 --seconds 120 --out benchmarks/results/line.json
    python benchmarks/run_benchmark.py image --rate 200 --seconds 120 --plot --out benchmarks/results/image.json

Run `python benchmarks/run_benchmark.py line --help` for every option.
"""
import argparse
import datetime
import importlib.metadata
import itertools
import json
import os
import platform
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time

import numpy as np

SAMPLE_INTERVAL = 0.25  # seconds between readings of the counters
DISK_RESERVE = 2**30  # stop acquiring early rather than let the disk fill up completely


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("mode", choices=["line", "image"], help="kind of data to simulate")
    parser.add_argument("--seconds", type=float, default=120.0, help="how long to acquire")
    parser.add_argument("--rate", type=float, default=None,
                        help="acquisitions (line) or frames (image) per second; 0 means as fast as possible "
                             "(default: 5000 for line, 200 for image)")
    parser.add_argument("--channels", type=int, default=3, help="line: number of channels")
    parser.add_argument("--window", type=int, default=10, help="line: samples per acquisition")
    parser.add_argument("--points", type=int, default=1000, help="line: samples shown per channel in the plot")
    parser.add_argument("--height", type=int, default=480, help="image: frame height in pixels")
    parser.add_argument("--width", type=int, default=640, help="image: frame width in pixels")
    parser.add_argument("--plot", action="store_true",
                        help="also draw the live plot, offscreen (needs a GPU or a software renderer)")
    parser.add_argument("--rotate-seconds", type=float, default=5.0, help="seal a stream segment this often")
    parser.add_argument("--rotate-frames", type=int, default=8192, help="or after this many frames")
    parser.add_argument("--rotate-bytes", type=int, default=256 * 2**20, help="or before it grows past this size")
    parser.add_argument("--drain-timeout", type=float, default=600.0,
                        help="after acquisition stops, how long to wait for storage to catch up")
    parser.add_argument("--ring-capacity", type=int, default=4096, help="frames held in shared memory")
    parser.add_argument("--workdir", default=None,
                        help="directory for the database and stream files, i.e. the disk under test "
                             "(default: a new temporary directory, deleted afterwards)")
    parser.add_argument("--note", default="", help="free text stored with the results, e.g. the disk and GPU")
    parser.add_argument("--out", default=None, help="results file (default: <mode>.json in the current directory)")
    args = parser.parse_args(argv)
    if args.rate is None:
        args.rate = 5000.0 if args.mode == "line" else 200.0
    return args


def make_source(frame, rate):
    """Custom acquisition function for a simulated sensor

    Every frame is a copy of `frame` whose first four bytes hold the frame's number, so the stored data
    can be checked afterwards. Frames are paced to `rate` per second (not paced if rate is 0).
    """
    numbers = itertools.count()
    interval = 1.0 / rate if rate > 0 else 0.0
    due = [None]

    def acquire(ser, frame_shape):
        if interval:
            now = time.perf_counter()
            if due[0] is None:
                due[0] = now
            if due[0] > now:
                time.sleep(due[0] - now)
            due[0] += interval
        out = frame.copy()
        out.reshape(-1).view(np.uint8)[:4] = np.frombuffer(np.uint32(next(numbers)).tobytes(), np.uint8)
        return out

    return acquire


def describe_machine(args, workdir):
    """What the results were measured on"""
    def version(name):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    cpu = platform.processor() or None
    memory_gb = None
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/cpuinfo") as f:
                cpu = next((line.split(":", 1)[1].strip() for line in f if line.startswith("model name")), cpu)
            with open("/proc/meminfo") as f:
                memory_gb = round(int(next(line.split()[1] for line in f if line.startswith("MemTotal"))) / 2**20, 1)
        elif sys.platform == "darwin":
            cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                 text=True).stdout.strip() or cpu
            memory_gb = round(int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True,
                                                 text=True).stdout) / 2**30, 1)
    except (OSError, ValueError, StopIteration):
        pass
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True,
                                text=True, timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        commit = None
    return {
        "measured_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "note": args.note,
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu": cpu,
        "cpu_count": os.cpu_count(),
        "memory_gb": memory_gb,
        "disk_free_gb": round(shutil.disk_usage(workdir).free / 2**30, 1),
        "git_commit": commit,
        "packages": {name: version(name) for name in
                     ("sensor-pipeline", "numpy", "scipy", "fastplotlib", "pygfx", "wgpu")},
    }


def verify(db, session_uuid):
    """Check that the stored frames are numbered 0, 1, 2, ... and carry their own number"""
    stored = missing = corrupted = 0
    if not os.path.exists(db):
        return {"frames_stored": 0, "frames_missing": 0, "frames_corrupted": 0}
    conn = sqlite3.connect(db)
    try:
        row = conn.execute("SELECT id FROM sessions WHERE uuid = ?", (session_uuid,)).fetchone()
        if row is None:  # nothing was stored
            return {"frames_stored": 0, "frames_missing": 0, "frames_corrupted": 0}
        (session_id,) = row
        rows = conn.execute("SELECT frame_index, substr(data, 1, 4) FROM frames WHERE session_id = ? "
                            "ORDER BY frame_index", (session_id,))
        expected = 0
        for index, head in rows:
            missing += max(0, index - expected)
            corrupted += int.from_bytes(head, "little") != (index & 0xFFFFFFFF)
            expected = index + 1
            stored += 1
    finally:
        conn.close()
    return {"frames_stored": stored, "frames_missing": missing, "frames_corrupted": corrupted}


def main(argv=None):
    args = parse_args(argv)
    if args.plot:
        os.environ.setdefault("RENDERCANVAS_FORCE_OFFSCREEN", "1")  # before fastplotlib is imported

    if args.mode == "line":
        keys = [f"ch{i}" for i in range(args.channels)]
        frame = np.zeros((args.window, args.channels), dtype=np.float32)
        frame_shape = (args.points, args.window, args.channels)
        plot_bytes = args.points * args.channels * 4  # one plot update: every channel's trace, float32
        options = {"dtype": np.float32, "data_mode": "line"}
    else:
        keys = ["camera"]
        frame = np.fromfunction(lambda r, c: (r + c) % 256, (args.height, args.width), dtype=np.int64)
        frame = frame.astype(np.uint8)[:, :, None]
        frame_shape = (args.height, args.width, 1)
        plot_bytes = frame.nbytes  # one plot update: the newest image
        options = {"dtype": np.uint8, "data_mode": "image"}

    temporary = args.workdir is None
    workdir = tempfile.mkdtemp(prefix="sensor_core_benchmark_") if temporary else args.workdir
    os.makedirs(workdir, exist_ok=True)
    try:
        return run(args, workdir, keys, frame, frame_shape, plot_bytes, options)
    finally:
        if temporary:
            shutil.rmtree(workdir, ignore_errors=True)


def run(args, workdir, keys, frame, frame_shape, plot_bytes, options):
    from sensor_core import SensorManager

    free = shutil.disk_usage(workdir).free
    if args.rate > 0:
        needed = 1.2 * args.rate * args.seconds * (frame.nbytes + 16) + 2 * args.rotate_bytes + DISK_RESERVE
        if needed > free:
            print(f"This run needs about {needed / 1e9:.1f} GB free in {workdir}, but {free / 1e9:.1f} GB is "
                  f"free. Lower --rate or --seconds, or point --workdir at a larger disk.", file=sys.stderr)
            return 2
    else:
        print(f"--rate 0 writes as fast as this machine allows; it stops early if the {free / 1e9:.1f} GB "
              f"free in {workdir} runs low.")
    db = os.path.join(workdir, f"benchmark_{args.mode}.sqlite3")
    config = {key: getattr(args, key) for key in ("mode", "seconds", "rate", "plot", "rotate_seconds",
                                                  "rotate_frames", "rotate_bytes", "ring_capacity")}
    config.update({"frame_shape": list(frame.shape), "dtype": str(frame.dtype), "frame_bytes": frame.nbytes,
                   "plot_bytes_per_update": plot_bytes if args.plot else None})
    environment = describe_machine(args, workdir)

    samples = {"acquired": [], "written": [], "stored": [], "rendered": []}
    sm = SensorManager(ser_channel_key=keys, commport=None, frame_shape=frame_shape, start_stream_ingest=True,
                       sqlite_path=db, rotate_seconds=args.rotate_seconds, rotate_frames=args.rotate_frames,
                       rotate_bytes=args.rotate_bytes, ring_capacity=args.ring_capacity, **options)
    try:
        worker = sm.update_data_process(virtual_ser_port=True, func=make_source(frame, args.rate))
        sm.start_process(worker)
        figure = sm.create_plot() if args.plot else None  # after acquisition starts (see create_plot)
        if figure is not None:
            import wgpu
            adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
            environment["gpu"] = adapter.info.get("device") if adapter else None
            figure.canvas.draw()  # the first draw sets up the GPU; leave it out of the measurement

        # measure from here on, with acquisition (and the plot) running
        t0_wall, t0 = time.time(), time.perf_counter()
        acquired_at_start = sm.ring.write_idx

        rendered, next_sample, next_draw = 0, t0, t0
        last_written = last_stored = None
        stopped_early = None
        end = t0 + args.seconds
        while (now := time.perf_counter()) < end:
            if figure is not None and now >= next_draw:
                figure.canvas.draw()  # updates the plot from the ring buffer, then renders it
                rendered += 1
                next_draw = max(next_draw + 1 / 60, now)  # the plot's target frame rate
            if now >= next_sample:
                samples["acquired"].append([now - t0, sm.ring.write_idx])
                if figure is not None:
                    samples["rendered"].append([now - t0, rendered])
                metrics = sm.get_metrics()
                written = metrics["writer"].get("writer_total_frames")
                if written is not None and written != last_written:
                    samples["written"].append([metrics["writer"]["writer_updated_unix"] - t0_wall, written])
                    last_written = written
                stored = metrics["ingest"].get("ingest_frames_ingested")
                if stored is not None and stored != last_stored:
                    samples["stored"].append([metrics["ingest"]["ingest_updated_unix"] - t0_wall, stored])
                    last_stored = stored
                if shutil.disk_usage(workdir).free < DISK_RESERVE:
                    stopped_early = "the disk was nearly full"
                    break
                next_sample += SAMPLE_INTERVAL
                if next_sample < now:  # fell behind (for example, a slow draw): resume the regular spacing
                    next_sample = now + SAMPLE_INTERVAL
            time.sleep(max(0.0, min(next_sample, next_draw if figure is not None else next_sample)
                           - time.perf_counter()))

        stopped = time.perf_counter()
        sm.stop(timeout=args.drain_timeout)  # stop acquiring, then write and store everything still in flight
        drained = time.perf_counter()
        metrics = sm.get_metrics()
        acquired = sm.ring.write_idx
        session = sm.session["uuid"]
    finally:
        sm.close()

    check = verify(db, session)
    frames_stored = check["frames_stored"]
    seconds = stopped - t0
    errors = [metrics[stage].get(f"{stage}_last_error") for stage in ("writer", "ingest")]
    summary = {
        "seconds_acquiring": seconds,
        "stopped_early": stopped_early,
        "worker_errors": [error for error in errors if error],
        "frames_acquired": acquired,
        "frames_written": metrics["writer"].get("writer_total_frames"),
        "frames_stored": frames_stored,
        "frames_dropped": metrics["writer"].get("writer_dropped_frames", 0),
        "frames_missing": max(0, acquired - frames_stored),
        "frames_corrupted": check["frames_corrupted"],
        "all_frames_stored_intact": frames_stored == acquired and check["frames_corrupted"] == 0,
        "acquired_per_second": (acquired - acquired_at_start) / seconds,
        "acquired_MB_per_second": (acquired - acquired_at_start) * frame.nbytes / seconds / 1e6,
        "seconds_to_store_the_rest_after_stopping": drained - stopped,
        "plot_frames_per_second": rendered / seconds if args.plot else None,
    }
    results = {"benchmark": "sensor_core throughput", "config": config, "environment": environment,
               "samples": samples, "summary": summary}
    out = args.out or f"{args.mode}.json"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=1)

    if stopped_early:
        print(f"stopped after {seconds:.1f} s: {stopped_early}")
    for error in summary["worker_errors"]:
        print(f"worker error: {error}")
    print(f"{args.mode}: {acquired} frames acquired; measured over {seconds:.1f} s "
          f"({summary['acquired_per_second']:.0f}/s, {summary['acquired_MB_per_second']:.2f} MB/s)")
    print(f"stored intact: {frames_stored} | dropped: {summary['frames_dropped']} | "
          f"missing: {summary['frames_missing']} | corrupted: {summary['frames_corrupted']} | "
          f"storing the rest after stopping took {summary['seconds_to_store_the_rest_after_stopping']:.1f} s")
    if args.plot:
        print(f"plot: {summary['plot_frames_per_second']:.1f} frames/s")
    print(f"results: {os.path.abspath(out)}")
    return 0 if summary["all_frames_stored_intact"] else 1


if __name__ == "__main__":
    sys.exit(main())
