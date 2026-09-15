#!/usr/bin/env python3
"""AMS24 -> Reverb (Muse arousal-controlled) -> speaker.

Combines the AMS24 USB audio interface driver (ams24_driver.py) with the
Muse EEG OSC control loop (muse_audio.py). Audio flows from the AMS24 input,
through a Pedalboard Reverb effect, out to the default speaker. The Reverb's
wet/dry mix and room size are driven in real time by the Muse arousal index:

    arousal = theta_absolute / (alpha_absolute + beta_absolute)

received via /muse/elements/{theta,alpha,beta}_absolute.

All audio I/O and DSP run inside pedalboard's compiled AudioStream (like
muse_audio.py), so there is no Python in the hot audio path -- only OSC
message handling (on its own thread) touches the Reverb's parameters. That
attribute assignment is a single GIL-protected op, so no lock is needed
between the OSC thread and the audio thread. A small buffer_size, set at
stream construction (like ams24_driver.py's low-latency sd.Stream), keeps
end-to-end latency down.

Install dependencies:
    pip install pedalboard python-osc

Run:
    python3 main.py                     # auto-detect AMS24 input, default speaker output
    python3 main.py --list-devices
    python3 main.py --input "..." --output "..."
    python3 main.py --osc-port 5500
"""

import argparse
import sys
import threading
import time

from pedalboard import Pedalboard, Reverb
from pedalboard.io import AudioStream
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer

DEVICE_NAME_HINT = "ams24"

# theta/(alpha+beta) typically falls somewhere around 0-2; map that range
# onto a musically useful reverb wet-mix / room-size range.
AROUSAL_MIN = 0.0
AROUSAL_MAX = 2.0
WET_MIN = 0.1
WET_MAX = 0.8
ROOM_SIZE_MIN = 0.2
ROOM_SIZE_MAX = 0.95


def _normalize(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def list_devices() -> None:
    print("Input devices:")
    for name in AudioStream.input_device_names:
        print(f"  - {name}")
    print("Output devices:")
    for name in AudioStream.output_device_names:
        print(f"  - {name}")


def resolve_input_device(explicit: str | None) -> str:
    if explicit:
        normalized = _normalize(explicit)
        for name in AudioStream.input_device_names:
            if normalized in _normalize(name):
                return name
        print(f"No input device matching '{explicit}'.", file=sys.stderr)
        sys.exit(1)

    normalized_hint = _normalize(DEVICE_NAME_HINT)
    for name in AudioStream.input_device_names:
        if normalized_hint in _normalize(name):
            return name

    print(f"Could not find an input device matching '{DEVICE_NAME_HINT}'.", file=sys.stderr)
    print("Available input devices:", file=sys.stderr)
    for name in AudioStream.input_device_names:
        print(f"  - {name}", file=sys.stderr)
    print('Pass --input "<device name>" to pick one explicitly.', file=sys.stderr)
    sys.exit(1)


def clamp01(fraction: float) -> float:
    return max(0.0, min(1.0, fraction))


class ArousalState:
    """Tracks theta/alpha/beta and maps arousal = theta / (alpha + beta) onto Reverb params."""

    def __init__(self, reverb: Reverb):
        self._reverb = reverb
        self._lock = threading.Lock()
        self._theta = 0.0
        self._alpha = 0.0
        self._beta = 0.0
        self._arousal = 0.0

    def update_theta(self, value: float) -> None:
        with self._lock:
            self._theta = value
            self._apply()

    def update_alpha(self, value: float) -> None:
        with self._lock:
            self._alpha = value
            self._apply()

    def update_beta(self, value: float) -> None:
        with self._lock:
            self._beta = value
            self._apply()

    def _apply(self) -> None:
        denom = self._alpha + self._beta
        # Mind Monitor's *_absolute band powers are log-scale (Bels) and are
        # commonly negative, so denom is often negative too -- guard against
        # it being near zero in magnitude, not just near-zero-and-positive.
        arousal = self._theta / denom if abs(denom) > 1e-6 else 0.0
        self._arousal = arousal
        fraction = clamp01((arousal - AROUSAL_MIN) / (AROUSAL_MAX - AROUSAL_MIN))
        wet = WET_MIN + fraction * (WET_MAX - WET_MIN)
        self._reverb.wet_level = wet
        self._reverb.dry_level = 1.0 - wet
        self._reverb.room_size = ROOM_SIZE_MIN + fraction * (ROOM_SIZE_MAX - ROOM_SIZE_MIN)

    def get_arousal(self) -> float:
        with self._lock:
            return self._arousal


def build_effects_chain() -> tuple[Pedalboard, Reverb]:
    reverb = Reverb(
        room_size=ROOM_SIZE_MIN,
        damping=0.5,
        wet_level=WET_MIN,
        dry_level=1.0 - WET_MIN,
        width=1.0,
        freeze_mode=0.0,
    )
    board = Pedalboard([reverb])
    return board, reverb


def _mean(values: tuple[float, ...]) -> float:
    return sum(values) / len(values) if values else 0.0


def make_dispatcher(state: ArousalState) -> Dispatcher:
    # Mind Monitor / muse-lsl send one value per EEG channel (e.g. TP9, AF7,
    # AF8, TP10) on each /muse/elements/*_absolute address, not a single
    # scalar -- average across channels.
    dispatcher = Dispatcher()
    dispatcher.map("/muse/elements/theta_absolute", lambda addr, *values: state.update_theta(_mean(values)))
    dispatcher.map("/muse/elements/alpha_absolute", lambda addr, *values: state.update_alpha(_mean(values)))
    dispatcher.map("/muse/elements/beta_absolute", lambda addr, *values: state.update_beta(_mean(values)))
    return dispatcher


def start_osc_server(ip: str, port: int, state: ArousalState) -> BlockingOSCUDPServer:
    server = BlockingOSCUDPServer((ip, port), make_dispatcher(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def start_arousal_printer(state: ArousalState, interval: float) -> threading.Event:
    """Print the current arousal index on its own thread, concurrently with
    OSC handling and audio streaming, independent of how often OSC updates
    arrive."""
    stop_event = threading.Event()

    def loop() -> None:
        while not stop_event.wait(interval):
            print(f"arousal={state.get_arousal():.3f}", flush=True)

    threading.Thread(target=loop, daemon=True).start()
    return stop_event


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", dest="input_device", default=None,
                         help="Name of the input device (default: auto-detect AMS24).")
    parser.add_argument("--output", dest="output_device", default=None,
                         help="Name of the output device (default: system default speaker).")
    parser.add_argument("--list-devices", action="store_true",
                         help="List available audio devices and exit.")
    parser.add_argument("--osc-ip", default="0.0.0.0",
                         help="Local interface to bind for Muse OSC (default: 0.0.0.0)")
    parser.add_argument("--osc-port", type=int, default=5500,
                         help="UDP port to listen for Muse OSC data (default: 5500)")
    parser.add_argument("--buffer-size", type=int, default=128,
                         help="Audio buffer size in samples; smaller = lower latency (default: 128).")
    parser.add_argument("--print-interval", type=float, default=0.5,
                         help="Seconds between concurrent arousal-index prints (default: 0.5).")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    input_device = resolve_input_device(args.input_device)
    output_device = args.output_device or AudioStream.default_output_device_name

    print(f"Input device:  {input_device}")
    print(f"Output device: {output_device}")
    print(f"Buffer size:   {args.buffer_size} samples")

    board, reverb = build_effects_chain()
    state = ArousalState(reverb)

    osc_server = start_osc_server(args.osc_ip, args.osc_port, state)
    print(f"Listening for Muse OSC on {args.osc_ip}:{args.osc_port}")

    printer_stop = start_arousal_printer(state, args.print_interval)

    try:
        with AudioStream(
            input_device_name=input_device,
            output_device_name=output_device,
            buffer_size=args.buffer_size,
            plugins=board,
            allow_feedback=True,
        ) as stream:
            print(
                "Streaming live from AMS24. Reverb is controlled by "
                "theta / (alpha + beta) arousal index. Press Enter to stop...",
            )
            input()
    finally:
        printer_stop.set()
        osc_server.shutdown()


if __name__ == "__main__":
    main()
