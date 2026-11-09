"""Offline YOLO readiness, image inference, and CPU latency measurements."""

import argparse
import json
from pathlib import Path
import platform
import statistics
import sys
import time

try:
    from .yolo_detector import YoloDetector
except ImportError:
    from yolo_detector import YoloDetector


def _count(text):
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if not 1 <= value <= 10000:
        raise argparse.ArgumentTypeError("must be in [1, 10000]")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("status", "image", "benchmark"))
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--class", dest="class_filter", action="append",
                        help="Exact metadata class name; repeat to select multiple classes")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--iterations", type=_count, default=20)
    parser.add_argument("--warmup", type=_count, default=3)
    args = parser.parse_args(argv)
    try:
        detector = YoloDetector(args.weights, args.metadata, args.confidence, args.iou, args.class_filter)
        result = detector.status()
        if result["status"] == "ready" and args.command != "status":
            if args.image is None or not args.image.is_file():
                raise ValueError("image/benchmark requires an existing --image file")
            import cv2
            image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("OpenCV could not decode --image")
            if args.command == "image":
                result = detector.detect(image)
            else:
                for _ in range(args.warmup):
                    result = detector.detect(image)
                    if result["status"] in ("error", "not_ready"):
                        break
                else:
                    samples = []
                    for _ in range(args.iterations):
                        started = time.perf_counter()
                        result = detector.detect(image)
                        if result["status"] in ("error", "not_ready"):
                            break
                        samples.append((time.perf_counter() - started) * 1000)
                    else:
                        ordered = sorted(samples)
                        result["benchmark"] = {
                            "iterations": len(samples), "warmup": args.warmup,
                            "median_ms": statistics.median(samples),
                            "p95_ms": ordered[max(0, (95 * len(samples) + 99) // 100 - 1)],
                            "min_ms": min(samples), "max_ms": max(samples),
                            "mean_ms": statistics.mean(samples),
                            "throughput_fps": 1000.0 / statistics.mean(samples),
                            "scope": "letterbox_blob_forward_decode_nms_serial_same_image",
                            "host": platform.node(), "machine": platform.machine(),
                            "python": platform.python_version(),
                            "source_image": str(args.image.resolve()), "image_io_included": False,
                            "accuracy_measured": False, "pl_acceleration": False}
        code = 2 if result["status"] in ("not_ready", "error") else 0
    except (ValueError, OSError, ImportError) as exc:
        result, code = {"status": "error", "reason": str(exc)}, 2
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return code


if __name__ == "__main__":
    sys.exit(main())
