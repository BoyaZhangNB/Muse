#!/usr/bin/env python3
"""Turn the Muse raw EEG stream into a melody.

Receives Muse OSC over UDP exactly like muse.py, treats the /muse/eeg samples
as a signal and transcribes it into notes:

  1. Frames   - channels are averaged to mono and cut into overlapping frames.
  2. Onsets   - spectral flux between consecutive frames, an adaptive
                (median-based) threshold and peak-picking decide where notes start.
  3. Pitch    - at each onset the fundamental f0 of the frame is estimated with
                YIN (autocorrelation-style periodicity search).
  4. Playback - f0 is shifted up a fixed number of octaves into the audible
                range, quantised to the nearest semitone and played on a small
                additive synth.

EEG is sampled at 256 Hz and its rhythms live at ~4-40 Hz, so the frame/hop
defaults are longer than the 20-50 ms / 5-10 ms used for audio: a frame must
hold a couple of periods of the slowest rhythm we want to detect.

    python3 melody.py                     # listen on all interfaces, port 5500
    python3 melody.py --port 5001
    python3 melody.py --octaves 6         # transpose higher
    python3 melody.py --fmin 8 --fmax 13  # only follow the alpha band
"""

import argparse
import collections
import datetime
import math
import socket
import sys
import threading

import numpy as np
import sounddevice as sd
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def freq_to_midi(freq: float) -> int:
    return int(round(69 + 12 * math.log2(freq / 440.0)))


def midi_to_freq(midi: int) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12)


def midi_name(midi: int) -> str:
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def yin_pitch(x: np.ndarray, fs: float, fmin: float, fmax: float, threshold: float):
    """Return (f0, aperiodicity) for frame x, or (None, aperiodicity) if unvoiced."""
    tau_min = max(2, int(fs / fmax))
    tau_max = min(len(x) // 2, int(math.ceil(fs / fmin)))
    if tau_max <= tau_min:
        return None, 1.0
    w = len(x) - tau_max

    # Difference function d(tau) and its cumulative-mean-normalised version d'(tau).
    d = np.array([np.sum((x[:w] - x[tau:tau + w]) ** 2) for tau in range(tau_max + 1)])
    cmnd = np.ones_like(d)
    running = np.cumsum(d[1:])
    cmnd[1:] = d[1:] * np.arange(1, tau_max + 1) / np.where(running > 0, running, 1)

    # First dip below the threshold, then walk down to its local minimum.
    below = np.nonzero(cmnd[tau_min:tau_max] < threshold)[0]
    if below.size == 0:
        return None, float(cmnd[tau_min:tau_max].min())
    tau = tau_min + below[0]
    while tau + 1 < tau_max and cmnd[tau + 1] < cmnd[tau]:
        tau += 1

    # Parabolic interpolation for sub-sample period accuracy.
    a, b, c = cmnd[tau - 1], cmnd[tau], cmnd[tau + 1]
    denom = a - 2 * b + c
    shift = 0.5 * (a - c) / denom if denom else 0.0
    return fs / (tau + shift), float(b)


class Synth:
    """Monophonic additive synth: each new note releases the previous one."""

    ATTACK = 0.01
    RELEASE = 0.12
    HARMONICS = (1.0, 0.5, 0.25)

    def __init__(self, max_note: float, volume: float, samplerate: int = 44100):
        self.sr = samplerate
        self.max_note = max_note
        self.volume = volume
        self._voices = []
        self._lock = threading.Lock()
        self._stream = sd.OutputStream(
            samplerate=samplerate, channels=1, dtype="float32", callback=self._callback
        )

    def __enter__(self):
        self._stream.start()
        return self

    def __exit__(self, *exc):
        self._stream.stop()
        self._stream.close()

    def play(self, freq: float) -> None:
        with self._lock:
            for voice in self._voices:
                voice["release_at"] = min(voice["release_at"], voice["n"] / self.sr)
            self._voices.append({"freq": freq, "n": 0, "release_at": self.max_note})

    def _render(self, voice, frames: int) -> np.ndarray:
        t = (voice["n"] + np.arange(frames)) / self.sr
        voice["n"] += frames
        wave = sum(
            amp * np.sin(2 * np.pi * voice["freq"] * (k + 1) * t)
            for k, amp in enumerate(self.HARMONICS)
        ) / sum(self.HARMONICS)
        env = np.clip(t / self.ATTACK, 0, 1) * np.exp(-3 * t / self.max_note)
        env *= np.clip(1 - (t - voice["release_at"]) / self.RELEASE, 0, 1)
        return wave * env

    def _callback(self, outdata, frames, time, status):
        buf = np.zeros(frames)
        with self._lock:
            for voice in self._voices:
                buf += self._render(voice, frames)
            self._voices = [
                v for v in self._voices if v["n"] / self.sr < v["release_at"] + self.RELEASE
            ]
        outdata[:, 0] = np.clip(buf * self.volume, -1, 1)


class NoteTracker:
    """Frame-by-frame onset detection + pitch estimation on a mono sample stream."""

    def __init__(self, args, on_note):
        self.fs = args.fs
        self.fmin = args.fmin
        self.fmax = args.fmax
        self.frame_len = int(round(args.frame * self.fs))
        self.hop = max(1, int(round(args.hop * self.fs)))
        self.yin_threshold = args.yin_threshold
        self.k = args.k
        self.delta = args.delta
        self.min_gap = args.min_gap
        self.on_note = on_note

        self._samples = collections.deque(maxlen=self.frame_len)
        self._since_hop = 0
        self._window = np.hanning(self.frame_len)
        freqs = np.fft.rfftfreq(self.frame_len, 1 / self.fs)
        self._band = (freqs >= self.fmin) & (freqs <= self.fmax)
        self._ramp = np.arange(self.frame_len) - (self.frame_len - 1) / 2

        self._prev_spec = None
        history_frames = max(3, int(round(args.threshold_window / args.hop)))
        self._flux_history = collections.deque(maxlen=history_frames)
        self._flux = collections.deque([0.0, 0.0], maxlen=2)  # flux at t-2, t-1
        self._frame_index = 0
        self._last_onset = -math.inf

    def push(self, sample: float) -> None:
        self._samples.append(sample)
        self._since_hop += 1
        if len(self._samples) == self.frame_len and self._since_hop >= self.hop:
            self._since_hop = 0
            self._process_frame(np.asarray(self._samples))

    def _process_frame(self, frame: np.ndarray) -> None:
        # Remove DC and linear drift (raw Muse EEG sits on a large offset).
        x = frame - frame.mean()
        x -= self._ramp * (np.dot(self._ramp, x) / np.dot(self._ramp, self._ramp))

        # Spectral flux: summed positive increase of log-magnitude across bins.
        spec = np.log1p(np.abs(np.fft.rfft(x * self._window))[self._band])
        if self._prev_spec is None:
            self._prev_spec = spec
            return
        flux = float(np.sum(np.maximum(spec - self._prev_spec, 0)))
        self._prev_spec = spec
        self._frame_index += 1

        # Peak-picking with one frame of look-ahead: frame t-1 is an onset if it
        # is a local maximum above the adaptive (median-based) threshold.
        before, candidate = self._flux
        threshold = self.k * float(np.median(self._flux_history)) + self.delta if self._flux_history else math.inf
        onset_time = (self._frame_index - 1) * self.hop / self.fs
        if (
            candidate > before
            and candidate >= flux
            and candidate > threshold
            and onset_time - self._last_onset >= self.min_gap
        ):
            self._last_onset = onset_time
            f0, aperiodicity = yin_pitch(x, self.fs, self.fmin, self.fmax, self.yin_threshold)
            self.on_note(f0, aperiodicity, candidate, threshold)

        self._flux_history.append(flux)
        self._flux.append(flux)


def make_eeg_handler(tracker: NoteTracker):
    def handler(address, *args):
        values = np.array([a for a in args if isinstance(a, (int, float))], dtype=float)
        if values.size == 0 or np.all(np.isnan(values)):
            return
        tracker.push(float(np.nanmean(values)))  # sum channels to mono

    return handler


def make_note_handler(synth: Synth, octaves: int, semitones: float = 0.0):
    shift = octaves + semitones / 12

    def on_note(f0, aperiodicity, flux, threshold):
        ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        if f0 is None:
            print(f"{ts}  onset  flux={flux:7.3f} thr={threshold:7.3f}  unvoiced "
                  f"(aperiodicity {aperiodicity:.2f})", flush=True)
            return
        midi = freq_to_midi(f0 * 2 ** shift)
        synth.play(midi_to_freq(midi))
        print(f"{ts}  onset  flux={flux:7.3f} thr={threshold:7.3f}  "
              f"f0={f0:6.2f} Hz -> {midi_name(midi):<4} (aperiodicity {aperiodicity:.2f})",
              flush=True)

    return on_note


def main():
    parser = argparse.ArgumentParser(description="Play Muse EEG as a melody.")
    parser.add_argument(
        "--ip",
        default="0.0.0.0",
        help="local interface to bind (default: 0.0.0.0 = all interfaces)",
    )
    parser.add_argument("--port", type=int, default=5500, help="UDP port (default: 5500)")
    parser.add_argument("--fs", type=float, default=256.0, help="EEG sample rate in Hz (default: 256)")
    parser.add_argument("--frame", type=float, default=0.2, help="frame length in s (default: 0.5)")
    parser.add_argument("--hop", type=float, default=0.0625, help="hop size in s (default: 0.0625)")
    parser.add_argument("--fmin", type=float, default=4.0, help="lowest EEG f0 in Hz (default: 4)")
    parser.add_argument("--fmax", type=float, default=60.0, help="highest EEG f0 in Hz (default: 60)")
    parser.add_argument("--k", type=float, default=1.5,
                        help="onset threshold = k * median(recent flux) + delta (default: 1.5)")
    parser.add_argument("--delta", type=float, default=0.1, help="onset threshold offset (default: 0.1)")
    parser.add_argument("--threshold-window", type=float, default=1.0,
                        help="seconds of flux history for the median threshold (default: 1.0)")
    parser.add_argument("--min-gap", type=float, default=0.2,
                        help="minimum time between onsets in s (default: 0.2)")
    parser.add_argument("--yin-threshold", type=float, default=0.25,
                        help="YIN aperiodicity threshold; higher accepts noisier pitch (default: 0.25)")
    parser.add_argument("--octaves", type=int, default=5,
                        help="octaves to shift EEG f0 up into audio (default: 5, i.e. x32)")
    parser.add_argument("--semitones", type=float, default=0.0,
                        help="extra transposition in semitones, added to --octaves (default: 0)")
    parser.add_argument("--max-note", type=float, default=1.0, help="max note length in s (default: 1.0)")
    parser.add_argument("--volume", type=float, default=0.3, help="output volume 0-1 (default: 0.3)")
    args = parser.parse_args()

    if args.frame * args.fmin < 2:
        print(f"Warning: a {args.frame}s frame holds fewer than two periods of "
              f"{args.fmin} Hz; YIN may miss the lowest pitches.", file=sys.stderr)

    with Synth(args.max_note, args.volume) as synth:
        tracker = NoteTracker(args, make_note_handler(synth, args.octaves, args.semitones))
        dispatcher = Dispatcher()
        dispatcher.map("/muse/eeg", make_eeg_handler(tracker))

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

        print(f"Listening for Muse EEG on {args.ip}:{args.port}  (Ctrl-C to stop)")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
