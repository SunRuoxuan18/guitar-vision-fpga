"""Board golden checks and measured OpenCV/DMA timings, with JSON evidence.

sudo env XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 \
    pynq/verify_vision_pl.py --load-overlay --report vision_pl_report.json
Optional --frame accepts a uint8 BGR .npy or OpenCV-decodable JPEG at 640x480.
Without --load-overlay the matching vision overlay must already be active.
"""

import argparse
import importlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

try:
    from .vision_pl import (DEFAULT_BITFILE, HEIGHT, WIDTH, VisionPL,
                            reference_cv2_words, reference_words, validate_frame)
except ImportError:
    from vision_pl import (DEFAULT_BITFILE, HEIGHT, WIDTH, VisionPL,
                           reference_cv2_words, reference_words, validate_frame)


def synthetic_frames():
    """Deterministic patterns including exact threshold and causal boundaries."""
    yield "black", np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    yield "white", np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
    colors = np.array([[0, 0, 255], [0, 255, 0], [255, 0, 0],
                       [0, 255, 255], [255, 0, 255], [255, 255, 0]], dtype=np.uint8)
    yield "color_bars", np.tile(colors[(np.arange(WIDTH) // 31) % 6], (HEIGHT, 1, 1))
    y, x = np.indices((HEIGHT, WIDTH))
    ramp = ((x * 8) % 256).astype(np.uint8)
    yield "threshold64_ramp", np.repeat(ramp[:, :, None], 3, axis=2)
    checker = (((x // 7 + y // 5) % 2) * 255).astype(np.uint8)
    yield "checkerboard", np.repeat(checker[:, :, None], 3, axis=2)
    impulse = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    impulse[[0, 1, 2, HEIGHT - 1], [0, 1, 2, WIDTH - 1]] = 255
    impulse[HEIGHT // 2, WIDTH // 2] = [1, 127, 253]
    yield "boundary_impulses", impulse
    yield "seeded_noise", np.random.RandomState(7020).randint(
        0, 256, size=(HEIGHT, WIDTH, 3)).astype(np.uint8)


def load_frame(path):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        frame = np.load(str(path), allow_pickle=False)
    else:
        cv2 = importlib.import_module("cv2")
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Cannot decode frame: {}".format(path))
    validate_frame(frame)
    return frame


def compare_words(expected, actual):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return {"exact": False, "error": "output shape/dtype differs from uint32 golden"}
    mismatch = np.flatnonzero(expected.reshape(-1) != actual.reshape(-1))
    report = {"exact": mismatch.size == 0, "mismatched_words": int(mismatch.size),
              "compared_words": int(expected.size)}
    if mismatch.size:
        index = int(mismatch[0])
        report["first_mismatch"] = {"x": index % WIDTH, "y": index // WIDTH,
                                    "expected": int(expected.reshape(-1)[index]),
                                    "actual": int(actual.reshape(-1)[index])}
    return report


def distribution(samples):
    return {"samples": len(samples), "mean": float(np.mean(samples)),
            "median": float(np.median(samples)), "p95": float(np.percentile(samples, 95)),
            "min": float(np.min(samples)), "max": float(np.max(samples))}


def verify(args):
    report = {"schema": "vision-pl-verification-v1", "status": "error",
              "backend": "pl", "cpu_fallback": False,
              "runtime": {"python": platform.python_version(), "platform": platform.platform()},
              "frames": [], "iterations": args.iterations, "warmup": args.warmup,
              "timing_scope": {
                  "opencv_ms": "exact integer grayscale, causal Sobel L1/clip, edge and output packing",
                  "pack_ms": "BGR camera frame to reusable contiguous uint32 BGR0",
                  "dma_ms": "register submission and both DMA completions; excludes packing and cache maintenance",
                  "total_ms": "driver call including lock, packing, caches, DMA and detached result copy",
                  "excluded": "camera capture, JPEG decode, HTTP, rendering and overlay initialization"}}
    driver = None
    try:
        # Fail before overlay download if the benchmark dependency is unavailable.
        cv2 = importlib.import_module("cv2")
        report["runtime"]["opencv"] = cv2.__version__
        supplied = [(str(path.resolve()), load_frame(path)) for path in args.frame]
        driver = VisionPL(args.bitfile, download=args.load_overlay, timeout=args.timeout)
        report["hardware_fingerprint"] = driver.fingerprint
        for name, frame in _frames(supplied):
            expected = reference_words(frame)
            cv_result = compare_words(expected, reference_cv2_words(frame))
            result = driver.process(frame)
            entry = {"name": name, "opencv_golden": cv_result,
                     "pl_golden": compare_words(expected, result["words"])}
            report["frames"].append(entry)
            if not cv_result["exact"] or not entry["pl_golden"]["exact"]:
                report["status"] = "mismatch"
                return report
            for _ in range(args.warmup):
                reference_cv2_words(frame)
                warm = driver.process(frame)
                if not compare_words(expected, warm["words"])["exact"]:
                    entry["warmup_failure"] = compare_words(expected, warm["words"])
                    report["status"] = "mismatch"
                    return report
            cpu_times, dma_times = [], []
            for _ in range(args.iterations):
                started = time.perf_counter()
                cpu_words = reference_cv2_words(frame)
                cpu_times.append((time.perf_counter() - started) * 1000)
                result = driver.process(frame)
                dma_times.append(result["timings_ms"])
                cpu_check = compare_words(expected, cpu_words)
                pl_check = compare_words(expected, result["words"])
                if not cpu_check["exact"] or not pl_check["exact"]:
                    entry["benchmark_failure"] = {"opencv": cpu_check, "pl": pl_check}
                    report["status"] = "mismatch"
                    return report
            entry["timings_ms"] = {"opencv_ms": distribution(cpu_times)}
            for key in dma_times[0]:
                entry["timings_ms"][key] = distribution([sample[key] for sample in dma_times])
            entry["provenance"] = result["provenance"]
        report["status"] = "passed"
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc),
                           "diagnostics": getattr(exc, "diagnostics", {})}
    finally:
        if driver is not None:
            try:
                driver.close()
            except Exception as exc:
                report["status"] = "error"
                report["close_error"] = {"type": type(exc).__name__, "message": str(exc)}
            report["driver_status"] = driver.status()
    return report


def _frames(supplied):
    yield from synthetic_frames()
    yield from supplied


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bitfile", type=Path, default=DEFAULT_BITFILE)
    parser.add_argument("--load-overlay", action="store_true",
                        help="Explicitly program the isolated vision overlay (replaces current PL)")
    parser.add_argument("--frame", action="append", type=Path, default=[],
                        help="Additional 640x480 uint8 BGR .npy or JPEG; may be repeated")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=1.0)
    parser.add_argument("--report", type=Path, help="Also write the JSON report to this path")
    args = parser.parse_args(argv)
    if args.iterations < 1 or args.warmup < 0:
        parser.error("iterations must be positive and warmup must be nonnegative")
    if not np.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be finite and positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    report = verify(args)
    text = json.dumps(report, indent=2, allow_nan=False)
    if args.report:
        args.report.write_text(text + "\n", encoding="utf-8")
    print(text)
    return {"passed": 0, "mismatch": 1}.get(report["status"], 2)


if __name__ == "__main__":
    sys.exit(main())
