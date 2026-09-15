"""
Real-time microphone -> speaker audio streaming with Pedalboard effects.

Captures audio from the Mac's microphone, runs it through a Pedalboard
effects chain (starting with Reverb), and plays the processed audio back
out in real time.

Install dependencies:
    pip install pedalboard

Run:
    python audio.py                 # use default input/output devices
    python audio.py --list-devices  # list available audio device names
    python audio.py --input "MacBook Pro Microphone" --output "MacBook Pro Speakers"
"""

import argparse

from pedalboard import Pedalboard, Reverb
from pedalboard.io import AudioStream


def list_devices() -> None:
    print("Input devices:")
    for name in AudioStream.input_device_names:
        print(f"  - {name}")
    print("Output devices:")
    for name in AudioStream.output_device_names:
        print(f"  - {name}")


def build_effects_chain() -> Pedalboard:
    return Pedalboard([
        Reverb(
            room_size=0.75,
            damping=0.5,
            wet_level=0.33,
            dry_level=0.6,
            width=1.0,
            freeze_mode=0.0,
        ),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", dest="input_device", default=None,
                         help="Name of the input (microphone) device to use.")
    parser.add_argument("--output", dest="output_device", default=None,
                         help="Name of the output (speaker) device to use.")
    parser.add_argument("--list-devices", action="store_true",
                         help="List available audio devices and exit.")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    input_device = args.input_device or AudioStream.default_input_device_name
    output_device = args.output_device or AudioStream.default_output_device_name

    print(f"Input device:  {input_device}")
    print(f"Output device: {output_device}")

    with AudioStream(input_device_name=input_device, output_device_name=output_device, allow_feedback=True) as stream:
        stream.plugins = build_effects_chain()
        print("Streaming live with reverb applied. Press Enter to stop...")
        input()


if __name__ == "__main__":
    main()
