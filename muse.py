#!/usr/bin/env python3
"""Receive Muse OSC streaming over UDP and print every message.

Point your Muse app (Mind Monitor, muse-lsl, muselsl stream --osc, etc.) at
this machine's IP and the port below, then run:

    python3 muse.py                    # listen on all interfaces, port 5000
    python3 muse.py --port 5001
    python3 muse.py --ip 192.168.1.10  # bind one specific local interface
    python3 muse.py --filter /muse/eeg # only print addresses starting with this
"""

import argparse
import datetime
import socket
import sys

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer


def make_handler(address_filter):
    def handler(address, *args):
        if address_filter and not address.startswith(address_filter):
            return
        ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        values = " ".join(
            f"{a:9.3f}" if isinstance(a, float) else str(a) for a in args
        )
        print(f"{ts}  {address:<28} {values}", flush=True)

    return handler


def main():
    parser = argparse.ArgumentParser(description="Print incoming Muse OSC data.")
    parser.add_argument(
        "--ip",
        default="0.0.0.0",
        help="local interface to bind (default: 0.0.0.0 = all interfaces)",
    )
    parser.add_argument("--port", type=int, default=5500, help="UDP port (default: 5000)")
    parser.add_argument(
        "--filter",
        dest="address_filter",
        default=None,
        help="only print OSC addresses starting with this prefix",
    )
    args = parser.parse_args()

    dispatcher = Dispatcher()
    dispatcher.set_default_handler(make_handler(args.address_filter))

    try:
        server = BlockingOSCUDPServer((args.ip, args.port), dispatcher)
    except OSError as exc:
        print(f"Could not bind {args.ip}:{args.port} -> {exc}", file=sys.stderr)
        print(
            "The bind address must be an IP on THIS machine "
            f"(this host is {socket.gethostbyname(socket.gethostname())}). "
            "Use the default 0.0.0.0 to listen on every interface.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Listening for OSC on {args.ip}:{args.port}  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
