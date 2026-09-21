# Benchmarks

`run_benchmark.py` measures how sensor_core handles a stream of line or image data, and `plot_results.py`
draws the results. Together they produce the performance figure in the paper.

## What is measured

A simulated sensor publishes frames at a fixed rate, or as fast as the machine allows. Each frame carries
its own number. While it runs, the benchmark counts:

- frames **acquired** (published to the shared-memory ring buffer),
- frames **written** to disk by the stream writer,
- frames **stored** in SQLite by the ingester, and
- with `--plot`, frames of the live plot **drawn**.

Every count is a running total read from the pipeline itself, not an estimate, so a stage that stalls
shows up as a drop in its rate. When the run ends, the benchmark checks that every stored frame is
present, in order, and holds its own content, and exits with an error if any frame is missing or
corrupted. The results, with the settings and a description of the machine (CPU, memory, operating
system, package versions, git commit, and GPU), are saved as JSON.

## Setup

```
pip install -e ".[benchmark]"
```

`--plot` also needs a GPU or a software renderer (on Linux: `sudo apt-get install mesa-vulkan-drivers`).
The plot is drawn offscreen, so the benchmark measures drawing, not presenting to a window.

## Reproducing the paper's figure

```
python benchmarks/run_benchmark.py line --rate 5000 --seconds 120 --plot --out benchmarks/results/line.json
python benchmarks/run_benchmark.py image --rate 200 --seconds 120 --plot --out benchmarks/results/image.json
python benchmarks/plot_results.py benchmarks/results/line.json benchmarks/results/image.json -o paper/benchmark.png
```

- **Line data:** 3 channels, 10 samples per acquisition (float32), 5,000 acquisitions per second, which is
  50,000 samples per second on each channel. The plot shows the latest 1,000 samples of each channel.
- **Image data:** 640 × 480 grayscale frames (uint8), 200 frames per second, or 61 MB per second.

Each run takes about two minutes. The image run needs about 10 GB of free disk; the benchmark checks
before it starts and deletes its data afterwards. The disk it measures is the one holding `--workdir`
(a temporary directory by default).

The files in `results/` were measured on the machine described in each file's `environment` section.
Your numbers will depend on your CPU, disk, and GPU; what should hold on any machine that sustains
these rates is that nothing is dropped, every frame is stored intact, and the storage lag stays bounded.

## Reading the results

The figure has one column per results file:

1. **Frames per second** acquired and written to disk, with MB per second on the right axis.
2. **Frames not yet stored** in SQLite. Data is stored a segment at a time (every `--rotate-seconds`,
   `--rotate-frames`, or `--rotate-bytes`, whichever comes first), so this rises while a segment fills
   and drops when it is stored. If storage keeps up, it stays bounded instead of growing.
3. **Plot frames per second**, when the run used `--plot`. The plotter's target is 60.

Each results file's `summary` also reports:

- `all_frames_stored_intact`: every acquired frame was stored, in order, with its own content
- `frames_dropped`: frames the writer could not read before the ring buffer overwrote them; they are
  counted and never stored, not stored damaged
- `acquired_per_second` and `acquired_MB_per_second`
- `seconds_to_store_the_rest_after_stopping`: how long storage took to catch up after acquisition stopped
- `plot_frames_per_second`

## Finding a machine's limits

`--rate 0` acquires as fast as the machine allows. Frames that the stream writer cannot keep up with are
reported as dropped. Because the data rate is unknown in advance, the run stops early if the disk runs
low. Run `python benchmarks/run_benchmark.py line --help` for every option, including frame sizes,
channels, segment sizes, and the ring buffer's capacity.
