"""The benchmark and its figure script keep working."""
import json
import os
import pathlib
import subprocess
import sys

import pytest

BENCHMARKS = pathlib.Path(__file__).resolve().parent.parent / "benchmarks"
if not BENCHMARKS.is_dir():
    pytest.skip("the benchmarks directory is not available", allow_module_level=True)


def run_benchmark(tmp_path, *options):
    out = tmp_path / "results.json"
    result = subprocess.run([sys.executable, str(BENCHMARKS / "run_benchmark.py"), *options,
                             "--workdir", str(tmp_path / "work"), "--out", str(out)],
                            cwd=tmp_path, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    with open(out) as f:
        return json.load(f)


@pytest.mark.processes
@pytest.mark.parametrize("options", [("line", "--rate", "500"), ("image", "--rate", "50", "--height", "120",
                                                                 "--width", "160")])
def test_the_benchmark_stores_every_frame_intact(tmp_path, options):
    results = run_benchmark(tmp_path, *options, "--seconds", "2")
    summary = results["summary"]
    assert summary["all_frames_stored_intact"]
    assert summary["frames_dropped"] == 0 and summary["frames_acquired"] > 0
    assert results["samples"]["acquired"] and results["environment"]["python"]


@pytest.mark.skipif(os.environ.get("SENSOR_CORE_RENDER_TESTS") != "1",
                    reason="set SENSOR_CORE_RENDER_TESTS=1 to run tests that render figures")
@pytest.mark.processes
def test_the_benchmark_measures_the_live_plot(tmp_path):
    results = run_benchmark(tmp_path, "line", "--rate", "500", "--seconds", "2", "--plot")
    assert results["summary"]["plot_frames_per_second"] > 0
    assert results["summary"]["all_frames_stored_intact"]


def test_the_benchmark_refuses_a_run_that_would_fill_the_disk(tmp_path):
    result = subprocess.run([sys.executable, str(BENCHMARKS / "run_benchmark.py"), "image", "--rate", "1e9",
                             "--seconds", "1e6", "--workdir", str(tmp_path)],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 2
    assert "GB free" in result.stderr


def test_the_figure_script_draws_results(tmp_path):
    pytest.importorskip("matplotlib")
    sys.path.insert(0, str(BENCHMARKS))
    try:
        import plot_results
    finally:
        sys.path.remove(str(BENCHMARKS))
    results = {
        "config": {"mode": "line", "rate": 100.0, "frame_shape": [10, 3], "dtype": "float32", "frame_bytes": 120,
                   "rotate_seconds": 5.0, "rotate_frames": 8192, "rotate_bytes": 2**28,
                   "plot_bytes_per_update": 12000},
        "environment": {"cpu": "test CPU", "cpu_count": 1, "os": "test OS"},
        "samples": {"acquired": [[t / 4, 25 * t] for t in range(40)],
                    "written": [[t, 100 * t] for t in range(10)],
                    "stored": [[5.0, 500], [9.9, 990]],
                    "rendered": [[t / 4, 15 * t] for t in range(40)]},
        "summary": {"frames_acquired": 1000, "frames_stored": 1000, "frames_dropped": 0, "seconds_acquiring": 10.0},
    }
    path = tmp_path / "line.json"
    path.write_text(json.dumps(results))
    out = tmp_path / "figure.png"
    assert plot_results.main([str(path), str(path), "-o", str(out)]) == 0
    assert out.stat().st_size > 0
