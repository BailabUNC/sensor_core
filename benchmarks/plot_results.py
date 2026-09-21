"""Draw the benchmark figure from results files written by run_benchmark.py.

Each results file becomes one column (for example line data and image data, as in the paper):

    python benchmarks/plot_results.py benchmarks/results/line.json benchmarks/results/image.json -o paper/benchmark.png

Rows: frames per second acquired and written to disk (with MB/s on the right axis); frames acquired but
not yet stored in SQLite, which rises between stored segments and must not grow over time; and, if the
run drew the live plot, the plot's frame rate.
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def rates(samples):
    """Rate between consecutive [time, count] samples, plotted at the end of each interval"""
    samples = np.asarray(samples, dtype=float)
    if len(samples) < 2:
        return np.empty(0), np.empty(0)
    dt = np.diff(samples[:, 0])
    keep = dt > 0
    return samples[1:, 0][keep], (np.diff(samples[:, 1])[keep] / dt[keep])


def headroom(ax):
    """Start the y axis at zero and leave space above the highest point, so flat traces stay visible"""
    top = max((line.get_ydata().max() for line in ax.get_lines() if len(line.get_ydata())), default=1.0)
    ax.set_ylim(0, 1.15 * top if top > 0 else 1.0)


def storage_lag(acquired, stored):
    """Frames acquired but not yet stored, at each acquisition sample"""
    acquired = np.asarray(acquired, dtype=float)
    stored = np.asarray(stored, dtype=float) if len(stored) else np.zeros((1, 2))
    idx = np.searchsorted(stored[:, 0], acquired[:, 0], side="right") - 1
    stored_then = np.where(idx >= 0, stored[np.clip(idx, 0, None), 1], 0.0)
    return acquired[:, 0], acquired[:, 1] - stored_then


def describe(results):
    c, s = results["config"], results["summary"]
    shape = " x ".join(str(x) for x in c["frame_shape"])
    rate = f"{c['rate']:g}/s" if c["rate"] else "as fast as possible"
    kind = "Line data" if c["mode"] == "line" else "Image data"
    return (f"{kind}: frames of {shape} {c['dtype']}, {rate}\n"
            f"{s['frames_acquired']:,} acquired, {s['frames_stored']:,} stored intact, "
            f"{s['frames_dropped']:,} dropped")


def draw(results_list, out):
    has_plot = any(r["samples"]["rendered"] for r in results_list)
    rows = 3 if has_plot else 2
    fig, axes = plt.subplots(rows, len(results_list), figsize=(6.5 * len(results_list), 3.2 * rows),
                             squeeze=False, sharex="col")
    for col, results in enumerate(results_list):
        samples, config = results["samples"], results["config"]
        frame_mb = config["frame_bytes"] / 1e6

        ax = axes[0, col]
        t, r = rates(samples["acquired"])
        ax.plot(t, r, label="acquired", color="tab:blue")
        t, r = rates(samples["written"])
        ax.plot(t, r, label="written to disk", color="tab:orange", linestyle="--")
        ax.set_ylabel("frames per second")
        headroom(ax)
        ax.secondary_yaxis("right", functions=(lambda f, k=frame_mb: f * k, lambda m, k=frame_mb: m / k)
                           ).set_ylabel("MB per second")
        ax.set_title(describe(results), fontsize=9)
        ax.legend(fontsize=8, loc="lower right")

        ax = axes[1, col]
        t, lag = storage_lag(samples["acquired"], samples["stored"])
        ax.plot(t, lag, color="tab:red")
        ax.set_ylabel("frames not yet stored")
        headroom(ax)
        ax.set_title(f"stored in SQLite a segment at a time: every {config['rotate_seconds']:g} s, "
                     f"{config['rotate_frames']:,} frames, or {config['rotate_bytes'] / 2**20:g} MB, "
                     f"whichever comes first", fontsize=8)

        if has_plot:
            ax = axes[2, col]
            if samples["rendered"]:
                plot_mb = (config["plot_bytes_per_update"] or 0) / 1e6
                t, r = rates(samples["rendered"])
                ax.plot(t, r, color="tab:purple")
                ax.secondary_yaxis("right", functions=(lambda f, k=plot_mb: f * k, lambda m, k=plot_mb: m / k)
                                   ).set_ylabel("MB per second")
            else:
                ax.text(0.5, 0.5, "plot not measured", ha="center", va="center", transform=ax.transAxes)
            ax.set_ylabel("plot frames per second")
            headroom(ax)
        axes[-1, col].set_xlabel("time (s)")
        axes[-1, col].set_xlim(0, results["summary"]["seconds_acquiring"])

    env = results_list[0]["environment"]
    cpus = f"{env.get('cpu_count')} CPU{'s' if env.get('cpu_count') != 1 else ''}"
    machine = ", ".join(str(x) for x in (env.get("cpu"), cpus, env.get("os"),
                                          env.get("gpu"), env.get("note")) if x)
    fig.text(0.01, 0.005, f"Measured on {machine}", fontsize=7, color="0.4")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out, dpi=200)
    print(f"figure: {out}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("results", nargs="+", help="results files from run_benchmark.py, one column each")
    parser.add_argument("-o", "--out", default="benchmark.png", help="image file to write")
    args = parser.parse_args(argv)
    results_list = []
    for path in args.results:
        with open(path) as f:
            results_list.append(json.load(f))
    draw(results_list, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
