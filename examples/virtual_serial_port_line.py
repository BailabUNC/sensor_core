"""Acquire from a simulated three-channel sensor, store the data, and check what was stored.

This is the virtual_serial_port_line notebook without the live plot, so it runs anywhere,
including machines without a display:

    python virtual_serial_port_line.py --seconds 5 --database example.sqlite3
"""
import argparse
import itertools
import time

import numpy as np

from sensor_core import SensorManager
from sensor_core.memory.strg_manager import StorageManager

CHANNELS = ["red", "infrared", "violet"]
FREQUENCIES = [1.0, 2.0, 3.0]  # Hz, one sine wave per channel
WINDOW = 10  # samples delivered per acquisition
SAMPLE_RATE = 500  # samples per second on each channel

next_sample = itertools.count(step=WINDOW)


def simulated_signal(first_sample, count):
    """`count` samples of every channel's sine wave, starting at sample number `first_sample`"""
    t = (first_sample + np.arange(count)) / SAMPLE_RATE
    return np.stack([100*np.sin(2 * np.pi * f * t) for f in FREQUENCIES], axis=1).astype(np.float32)


def acquire_data(ser, frame_shape):
    """Simulated sensor: the next WINDOW samples of every channel, delivered at SAMPLE_RATE

    sensor_core calls this over and over with the serial port (None for a virtual port) and
    frame_shape = (num_points, window_size, channels). It returns one acquisition shaped
    (window_size, channels), or None if nothing new is available.
    """
    time.sleep(WINDOW / SAMPLE_RATE)  # as long as the sensor takes to produce WINDOW samples
    return simulated_signal(next(next_sample), WINDOW)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seconds", type=float, default=5.0, help="how long to acquire")
    parser.add_argument("--database", default="virtual_serial_port_line.sqlite3",
                        help="SQLite file to store the data in")
    args = parser.parse_args()

    # Leaving the with block closes the SensorManager: acquisition stops and everything is stored
    with SensorManager(ser_channel_key=CHANNELS, commport=None, frame_shape=(1000, WINDOW, len(CHANNELS)),
                       start_stream_ingest=True, sqlite_path=args.database) as sm:
        sm.start_process(sm.update_data_process(virtual_ser_port=True, func=acquire_data))
        time.sleep(args.seconds)

    session = StorageManager.list_sessions(args.database)[-1]
    red = StorageManager.load_serial_channel("red", args.database, session=-1)
    times = StorageManager.load_frame_times(args.database, session=-1)
    complete = np.array_equal(red, simulated_signal(0, len(red))[:, 0])
    print(f"Stored {session['frames']} acquisitions ({len(red)} samples per channel) in {args.database}")
    print(f"Every sample stored, in order: {complete}")
    print(f"Acquisition times span {(times[-1] - times[0]) / 1e9:.2f} s")
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
