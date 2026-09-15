#!/usr/bin/env python3
"""Read audio data from an AMS24 (ZOOM AMS-24) USB-C audio interface and
live-plot it.

The AMS24 enumerates as a standard USB Audio Class 2.0 (UAC 2.0) device, so
no special driver is needed on macOS -- it shows up like any other audio
interface. This captures audio from the AMS24's input, plots a rolling
window of the waveform in real time, and plays that same audio out through
your Mac's default speaker.

Install dependencies:
    pip install sounddevice numpy matplotlib

Run:
    python3 ams24_driver.py                    # auto-detect AMS24 input, live plot
    python3 ams24_driver.py --list-devices      # list available audio device names
    python3 ams24_driver.py --input "..."       # override the input device
    python3 ams24_driver.py --window 2          # seconds of audio shown (default 5)
    python3 ams24_driver.py --print             # also print each raw block as an array
"""

import argparse
import sys
import threading
import time

import numpy as np
import sounddevice as sd

DEVICE_NAME_HINT = "ams24"


def _normalize(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def list_devices() -> None:
    for i, device in enumerate(sd.query_devices()):
        kind = []
        if device["max_input_channels"] > 0:
            kind.append(f"in:{device['max_input_channels']}")
        if device["max_output_channels"] > 0:
            kind.append(f"out:{device['max_output_channels']}")
        print(f"  [{i}] {device['name']}  ({' '.join(kind)})")


def find_input_device(hint: str) -> int | None:
    normalized_hint = _normalize(hint)
    for i, device in enumerate(sd.query_devices()):
        if device["max_input_channels"] > 0 and normalized_hint in _normalize(device["name"]):
            return i
    return None


def resolve_input_device(explicit: str | None) -> int:
    if explicit:
        try:
            return int(explicit)
        except ValueError:
            pass
        normalized = _normalize(explicit)
        for i, device in enumerate(sd.query_devices()):
            if normalized in _normalize(device["name"]):
                return i
        print(f"No input device matching '{explicit}'.", file=sys.stderr)
        sys.exit(1)

    found = find_input_device(DEVICE_NAME_HINT)
    if found is not None:
        return found

    print(f"Could not find an input device matching '{DEVICE_NAME_HINT}'.", file=sys.stderr)
    print("Available devices:", file=sys.stderr)
    list_devices()
    print("Pass --input \"<device name or index>\" to pick one explicitly.", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", dest="input_device", default=None,
                         help="Name or index of the input device (default: auto-detect AMS24).")
    parser.add_argument("--blocksize", type=int, default=1024,
                         help="Samples per audio callback (default: 1024).")
    parser.add_argument("--sr", type=int, default=None,
                             help="Sample rate (default: 48000).")
    parser.add_argument("--window", type=float, default=5.0,
                         help="Seconds of audio shown in the plot (default: 5.0).")
    parser.add_argument("--print", dest="print_blocks", action="store_true",
                         help="Also print each raw block as an array.")
    parser.add_argument("--list-devices", action="store_true",
                         help="List available audio devices and exit.")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    device_index = resolve_input_device(args.input_device)
    device_info = sd.query_devices(device_index)
    channels = device_info["max_input_channels"]
    samplerate = device_info["default_samplerate"] if args.sr is None else args.sr

    output_info = sd.query_devices(kind="output")
    output_channels = min(channels, output_info["max_output_channels"])

    print(f"Input device:  [{device_index}] {device_info['name']}")
    print(f"Output device: {output_info['name']} (default speaker)")
    print(f"Channels: {channels}  Sample rate: {samplerate}")
    print("Live plotting + playback. Close the plot window to stop.\n")

    window_samples = max(int(args.window * samplerate), args.blocksize)
    buffer = np.zeros((window_samples, channels), dtype="float32")
    buffer_lock = threading.Lock()

    def callback(indata, outdata, frames, time_info, status):
        nonlocal buffer
        t0 = time.perf_counter()
        # if status:
        #     print(f"status: {status}", file=sys.stderr)
        # if args.print_blocks:
        #     print(indata)
        with buffer_lock:
            buffer = np.roll(buffer, -frames, axis=0)
            buffer[-frames:] = indata
        outdata[:] = indata[:, :output_channels]
        peak = np.abs(indata).max()
        if peak >= 0.99:  # float32, so 1.0 = full scale = clipping
            print(f"CLIPPING, peak={peak:.3f}")
        # elapsed = time.perf_counter() - t0
        # budget = frames / samplerate
        # if elapsed > budget * 0.5:
        # print(f"callback took {elapsed * 1000:.2f}ms, budget was {budget * 1000:.2f}ms",
        #         file=sys.stderr)

    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    x = np.arange(window_samples) / samplerate
    lines = [ax.plot(x, buffer[:, ch], label=f"ch{ch}")[0] for ch in range(channels)]
    ax.set_ylim(-1, 1)
    ax.set_xlim(0, x[-1])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(device_info["name"])
    ax.legend(loc="upper right")

    def update(_frame):
        with buffer_lock:
            data = buffer.copy()
        for ch, line in enumerate(lines):
            line.set_ydata(data[:, ch])
        return lines

    with sd.Stream(
        device=(device_index, None),
        channels=(channels, output_channels),
        samplerate=samplerate,
        blocksize=args.blocksize,
        latency="low",
        callback=callback,
    ):
        # ani = animation.FuncAnimation(fig, update, interval=30, blit=True, cache_frame_data=False)
        # plt.show()
        input()


if __name__ == "__main__":
    main()
