#!/usr/bin/env python3
"""Muse EEG-controlled audio effects.

Combines the OSC listener from muse.py with the live microphone -> speaker
Pedalboard stream from audio.py. The mic/speaker signal path is unchanged;
what's new is that the Distortion() effect's drive is driven in real time by
the Muse headband's alpha_absolute + beta_absolute band powers (received via
/muse/elements/alpha_absolute and /muse/elements/beta_absolute).

Install dependencies:
    pip install pedalboard python-osc

Run:
    python3 muse_audio.py                          # OSC on 0.0.0.0:5500, default audio devices
    python3 muse_audio.py --list-devices
    python3 muse_audio.py --input "MacBook Pro Microphone" --output "MacBook Pro Speakers"
    python3 muse_audio.py --osc-port 5500
"""

import argparse
import threading

from pedalboard import Distortion, Pedalboard, Reverb
from pedalboard.io import AudioStream
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer

# alpha_absolute + beta_absolute typically fall somewhere around 0-2 in the
# Muse stream; map that range onto a musically useful distortion drive range.
DRIVE_DB_MIN = 0.0
DRIVE_DB_MAX = 60.0
BAND_SUM_MIN = 0.0
BAND_SUM_MAX = 2.0


def list_devices() -> None:
    print("Input devices:")
    for name in AudioStream.input_device_names:
        print(f"  - {name}")
    print("Output devices:")
    for name in AudioStream.output_device_names:
        print(f"  - {name}")


def band_sum_to_drive_db(total: float) -> float:
    clamped = max(BAND_SUM_MIN, min(BAND_SUM_MAX, total))
    span = BAND_SUM_MAX - BAND_SUM_MIN
    fraction = (clamped - BAND_SUM_MIN) / span if span else 0.0
    return DRIVE_DB_MIN + fraction * (DRIVE_DB_MAX - DRIVE_DB_MIN)


class BandState:
    """Tracks the latest alpha/beta readings and updates the Distortion plugin."""

    def __init__(self, distortion: Distortion):
        self._distortion = distortion
        self._lock = threading.Lock()
        self._alpha = 0.0
        self._beta = 0.0

    def update_alpha(self, value: float) -> None:
        with self._lock:
            self._alpha = value
            self._apply()

    def update_beta(self, value: float) -> None:
        with self._lock:
            self._beta = value
            self._apply()

    def _apply(self) -> None:
        total = self._alpha + self._beta
        drive_db = band_sum_to_drive_db(total)
        self._distortion.drive_db = drive_db
        print(
            f"alpha={self._alpha:.3f} beta={self._beta:.3f} "
            f"sum={total:.3f} -> drive_db={drive_db:.1f}",
            flush=True,
        )


def build_effects_chain() -> tuple[Pedalboard, Distortion]:
    distortion = Distortion(drive_db=DRIVE_DB_MIN)
    board = Pedalboard([
        distortion,
        # Reverb(
        #     room_size=0.75,
        #     damping=0.5,
        #     wet_level=0.33,
        #     dry_level=0.6,
        #     width=1.0,
        #     freeze_mode=0.0,
        # ),
    ])
    return board, distortion


def make_dispatcher(state: BandState) -> Dispatcher:
    dispatcher = Dispatcher()
    dispatcher.map("/muse/elements/alpha_absolute", lambda addr, value: state.update_alpha(value))
    dispatcher.map("/muse/elements/beta_absolute", lambda addr, value: state.update_beta(value))
    return dispatcher


def start_osc_server(ip: str, port: int, state: BandState) -> BlockingOSCUDPServer:
    server = BlockingOSCUDPServer((ip, port), make_dispatcher(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", dest="input_device", default=None,
                         help="Name of the input (microphone) device to use.")
    parser.add_argument("--output", dest="output_device", default=None,
                         help="Name of the output (speaker) device to use.")
    parser.add_argument("--list-devices", action="store_true",
                         help="List available audio devices and exit.")
    parser.add_argument("--osc-ip", default="0.0.0.0",
                         help="local interface to bind for Muse OSC (default: 0.0.0.0)")
    parser.add_argument("--osc-port", type=int, default=5500,
                         help="UDP port to listen for Muse OSC data (default: 5500)")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    input_device = args.input_device or AudioStream.default_input_device_name
    output_device = args.output_device or AudioStream.default_output_device_name

    print(f"Input device:  {input_device}")
    print(f"Output device: {output_device}")

    board, distortion = build_effects_chain()
    state = BandState(distortion)

    osc_server = start_osc_server(args.osc_ip, args.osc_port, state)
    print(f"Listening for Muse OSC on {args.osc_ip}:{args.osc_port}")

    try:
        with AudioStream(input_device_name=input_device, output_device_name=output_device, allow_feedback=True) as stream:
            stream.plugins = board
            print("Streaming live. Distortion drive is controlled by alpha+beta band power. Press Enter to stop...")
            input()
    finally:
        osc_server.shutdown()


if __name__ == "__main__":
    main()
