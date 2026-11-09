"""Stdlib JSON state server. Hardware is read only by one sampling thread."""

import argparse
from dataclasses import asdict
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit

import numpy as np

from audio_fft import AUD4_BASE_ADDRESS, AUD4_ADDRESS_RANGE, AUD4_REGISTERS, AudioFFT, FFTRegisterMap
from audio_music import (MusicConfig, MusicRecognizer, midi_frequency,
                         recognize_chord, spectrum_from_samples, CHORD_TYPES)
from music_tracking import MusicTracker
from av_fusion import fuse, read_visual, FretMappingTracker, visual_transition


class MusicState:
    def __init__(self, source, recognizer=None, poll_interval=0.1,
                 source_kind="hardware", unavailable_error=None, visual_path=None):
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        self.source = source
        self.recognizer = recognizer or MusicRecognizer()
        self.poll_interval = poll_interval
        self.source_kind = source_kind
        self.unavailable_error = unavailable_error
        self.visual_path = visual_path
        self.fret_mapping = FretMappingTracker()
        self.tracker = MusicTracker()
        self._lock = threading.Lock()
        self._last_sequence = None
        self._last_success = None
        self._state = self._empty("hardware_unavailable" if source is None else "starting",
                                  unavailable_error)

    def _empty(self, status, error=None):
        return {"schema_version": 1, "status": status, "signal_active": False,
                "source": self.source_kind, "hardware_available": False,
                "confidence": 0.0, "notes": [], "chord": recognize_chord([]),
                "sequence": self._last_sequence, "stale": True,
                "updated_at": None, "error": error, "fft": None,
                "spectrum": None, "scale_evidence": None}

    def update(self):
        if self.source is None:
            return
        try:
            update_started = time.monotonic()
            if hasattr(self.source, "read_snapshot_if_new"):
                snapshot = self.source.read_snapshot_if_new(self._last_sequence)
            else:
                snapshot = self.source.read_snapshot()
            if snapshot is None:
                return
            observed_at = time.monotonic()
            snapshot_read_ms = (observed_at - update_started) * 1000
            if snapshot.sample_rate != self.recognizer.config.sample_rate or snapshot.fft_size != self.recognizer.config.fft_size:
                raise ValueError("FFT source and recognizer configurations do not match")
            if snapshot.sequence == self._last_sequence:
                return
            if type(snapshot.sequence) is not int or not 0 <= snapshot.sequence < 2**32:
                raise ValueError('Invalid FFT sequence')
            if (self._last_sequence is not None and
                    (snapshot.sequence-self._last_sequence) & 0xffffffff >= 0x80000000):
                # A delayed snapshot cannot refresh age or cast confirmation votes.
                return
            inference_started = time.monotonic()
            state = self.recognizer.analyze(snapshot.power, snapshot.sequence)
            visual = read_visual(self.visual_path) if self.visual_path else None
            if visual:
                state['visual_transition'] = visual_transition(visual, time.monotonic(),
                                                              audio_observed_at=observed_at)
            inference_seconds = time.monotonic() - inference_started
            metadata = snapshot.metadata or {"fft_size": snapshot.fft_size,
                                             "sample_rate_hz": snapshot.sample_rate,
                                             "bin_count": len(snapshot.power),
                                             "bin_spacing_hz": snapshot.sample_rate / snapshot.fft_size,
                                             "window": self.recognizer.config.window,
                                             "power_format": "normalized_float_power"}
            if metadata.get("window", self.recognizer.config.window) != self.recognizer.config.window:
                raise ValueError("FFT source window and recognition template window do not match")
            state = self.tracker.update(state, now=time.monotonic(),
                                        context=(self.source_kind, metadata.get("analysis_test_mode", 0)))
            state.update(schema_version=1, source=self.source_kind,
                         hardware_available=self.source_kind == "hardware",
                         updated_at=time.time(), stale=False, error=None,
                         observed_monotonic=observed_at,
                         config=asdict(self.recognizer.config), fft=metadata,
                         inference_ms=round(inference_seconds * 1000, 3))
            state["timing"] = {
                "snapshot_read_ms": round(snapshot_read_ms, 3),
                "inference_ms": round(inference_seconds * 1000, 3),
                "processing_ms": None,
                "poll_interval_ms": self.poll_interval * 1000,
                "fft_window_ms": snapshot.fft_size / snapshot.sample_rate * 1000,
                "observation_to_result_ms": None,
                "evidence": "PS monotonic durations; excludes capture age and browser display"}
            state["algorithm_version"] = "harmonic_tracking_v4_av_integrity"
            state['capabilities'] = dict(chord_qualities=[quality for quality,_,_ in CHORD_TYPES],
                midi_range=[self.recognizer.config.min_midi,self.recognizer.config.max_midi],
                guitar_fret_range=[0,24], complex_voicings_require_observed_tones=True,
                chord_names_register_invariant=True)
            state["compute_backend"] = "arm_native" if getattr(self.recognizer, "_native", None) is not None else "numpy"
            # Spectrum peaks visualize spectral energy; they are never used as note labels.
            power = np.asarray(snapshot.power)
            state["spectrum"]["bins"] = power.tolist()
            peaks = np.flatnonzero((power[1:-1] > power[:-2]) & (power[1:-1] >= power[2:])) + 1
            peaks = sorted(peaks, key=lambda index: float(power[index]), reverse=True)[:24]
            state["spectrum"]["peaks"] = [{"bin": int(index),
                                            "frequency_hz": round(index * snapshot.sample_rate / snapshot.fft_size, 3),
                                            "power": float(power[index])} for index in peaks]
            if self.visual_path:
                state['fusion'] = self.fret_mapping.update(state, visual, time.monotonic())
            state["timing"]["processing_ms"] = round((time.monotonic() - update_started) * 1000, 3)
            state["timing"]["observation_to_result_ms"] = round((time.monotonic()-observed_at)*1000, 3)
            with self._lock:
                self._state = state
                self._last_success = time.monotonic()
            self._last_sequence = snapshot.sequence
        except Exception as error:
            self.tracker.reset()
            with self._lock:
                self._state = self._empty("hardware_unavailable" if isinstance(error, PermissionError) else "error",
                                          "{}: {}".format(type(error).__name__, error))
            # A recovered frame must republish even if its sequence equals the last good one.
            self._last_sequence = None

    def snapshot(self):
        with self._lock:
            state = dict(self._state)
            age = None if self._last_success is None else time.monotonic() - self._last_success
            state["age_seconds"] = None if age is None else round(age, 4)
            state["stale"] = bool(state.get("stale") or age is None or
                                  age > max(1.0, 5 * self.poll_interval))
        if self.visual_path:
            now = time.monotonic()
            visual = read_visual(self.visual_path)
            state['fusion'] = self.fret_mapping.apply_anchor(fuse(state,visual,now),visual,now)
        return state

    def run(self, stop):
        while not stop.is_set():
            started = time.monotonic()
            self.update()
            stop.wait(max(0.001, self.poll_interval - (time.monotonic() - started)))


def make_server(state, host="127.0.0.1", port=8081, cors_origin=None, web_root=None):
    web_root = Path(web_root) if web_root is not None else Path(__file__).resolve().parent.parent / "web"

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(web_root), **kwargs)

        def translate_path(self, path):
            translated = Path(super().translate_path(path)).resolve()
            root = web_root.resolve()
            if translated != root and root not in translated.parents:
                return str(root / ".outside_static_root_denied")
            return str(translated)

        def list_directory(self, path):
            self._reply(404, {"error": "static_index_unavailable"})
            return None
        def _reply(self, code, payload):
            body = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            if cors_origin is not None:
                self.send_header("Access-Control-Allow-Origin", cors_origin)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path in ("/api/audio", "/api/music", "/api/state"):
                self._reply(200, state.snapshot())
            elif path == "/healthz":
                current = state.snapshot()
                healthy = not current["stale"] and current["status"] not in ("starting", "error", "hardware_unavailable")
                self._reply(200 if healthy else 503, {"healthy": healthy,
                                                     "status": current["status"],
                                                     "stale": current["stale"]})
            elif path.startswith("/api/"):
                self._reply(404, {"error": "unknown_endpoint"})
            else:
                super().do_GET()

        def log_message(self, format_string, *args):
            pass

    return ThreadingHTTPServer((host, port), Handler)


class DemoSource:
    """Synthetic local source; this class never imports or touches PYNQ."""

    def __init__(self):
        self.sequence = 0
        self.time = np.arange(4096) / 48000.0

    def read_snapshot(self):
        from audio_fft import FFTSnapshot
        voicings = ((48, 52, 55), (50, 53, 57, 60), (40, 47), ())
        notes = voicings[(self.sequence // 30) % len(voicings)]
        samples = np.zeros(4096)
        for midi in notes:
            for harmonic in range(1, 9):
                samples += 0.12 / harmonic * np.sin(2 * np.pi * midi_frequency(midi) * harmonic * self.time)
        self.sequence = (self.sequence + 1) & 0xFFFFFFFF
        return FFTSnapshot(self.sequence, spectrum_from_samples(samples))


def settle_analysis_source(source, timeout=5.0):
    """Discard spectra still in flight when the analysis source was selected."""
    previous = source.read_snapshot().sequence
    observed = 0
    deadline = time.monotonic() + timeout
    while observed < 3:
        if time.monotonic() >= deadline:
            raise TimeoutError("No fresh FFT frames after analysis source selection")
        time.sleep(.09)
        sequence = source.read_snapshot().sequence
        if sequence != previous:
            delta = (sequence - previous) & 0xFFFFFFFF
            if delta >= 0x80000000:
                raise RuntimeError("FFT sequence moved backward while settling")
            previous = sequence
            observed += 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="Synthetic source, no board access")
    parser.add_argument("--base-address", type=lambda value: int(value, 0), default=AUD4_BASE_ADDRESS)
    parser.add_argument("--address-range", type=lambda value: int(value, 0), default=AUD4_ADDRESS_RANGE)
    parser.add_argument("--dev-mem", action="store_true",
                        help="Use Linux /dev/mem mmap directly, bypassing PYNQ/XRT device discovery")
    parser.add_argument("--test-mode", type=int, choices=range(4), default=0,
                        help="AUD4 analysis input: 0 live (default), 1 A4, 2 C major, 3 A minor")
    parser.add_argument("--register-map", help="JSON map overriding the verified AUD4 register map")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--poll-interval", type=float, default=0.1)
    parser.add_argument("--silence-power", type=float, default=1e-10)
    parser.add_argument("--minimum-rms", type=float, default=0.001,
                        help="Full-scale normalized RMS gate; 0.001 is about 33 PCM16 counts")
    parser.add_argument("--window", choices=("hann", "rectangular"),
                        help="Match the actual PL window; defaults to map metadata")
    parser.add_argument("--scales", action="store_true")
    parser.add_argument("--warmup-templates", action="store_true")
    parser.add_argument("--cors-origin", help="Explicit browser origin, only needed across origins")
    parser.add_argument("--web-root", default=str(Path(__file__).resolve().parent.parent / "web"))
    args = parser.parse_args(argv)
    error = None
    if args.demo:
        source = DemoSource()
        config = MusicConfig(silence_power=args.silence_power, minimum_rms=args.minimum_rms, infer_scales=args.scales,
                             window=args.window or "hann")
        if config.window != "hann":
            parser.error("The demo source uses a Hann window")
    else:
        source = None
        registers = AUD4_REGISTERS
        try:
            if args.register_map:
                with open(args.register_map, encoding="utf-8") as handle:
                    registers = FFTRegisterMap(**json.load(handle))
            source = AudioFFT(register_map=registers, base_address=args.base_address,
                              address_range=args.address_range, dev_mem=args.dev_mem)
            source.set_test_mode(args.test_mode)
            settle_analysis_source(source)
        except Exception as failure:
            error = "{}: {}".format(type(failure).__name__, failure)
            print("Hardware unavailable: " + error)
            if source is not None:
                source.close()
                source = None
        config = MusicConfig(silence_power=args.silence_power, minimum_rms=args.minimum_rms,
                             power_scale=float(1 << registers.power_shift), infer_scales=args.scales,
                             window=args.window or registers.window)
    recognizer = MusicRecognizer(config)
    if args.warmup_templates:
        recognizer._build_templates()
    state = MusicState(source, recognizer, args.poll_interval,
                       source_kind="demo" if args.demo else "hardware", unavailable_error=error,
                       visual_path=os.environ.get('FRETLENS_VISUAL_STATE'))
    server = make_server(state, args.host, args.port, args.cors_origin, args.web_root)
    stop = threading.Event()
    sampler = threading.Thread(target=state.run, args=(stop,), name="music-sampler", daemon=True)
    sampler.start()
    print("Music JSON API: http://{}:{}/api/audio".format(args.host, server.server_port))
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        server.server_close()
        sampler.join(timeout=2.0)
        if source is not None and hasattr(source, "close") and not sampler.is_alive():
            source.close()


if __name__ == "__main__":
    main()
