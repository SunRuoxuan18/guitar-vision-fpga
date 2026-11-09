from __future__ import annotations

import json
import time
from pathlib import Path

import ncnn
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
IMAGE_ROOT = ROOT.parents[1] / "datasets" / "fingertip_boost_20261030" / "images"
OUTPUT_ROOT = ROOT / "output"
DETECTOR_PARAM = ROOT / "weights" / "handpose.param"
DETECTOR_MODEL = ROOT / "weights" / "handpose.bin"
HANDPOSE_PARAM = ROOT / "weights" / "resnet50.param"
HANDPOSE_MODEL = ROOT / "weights" / "resnet50.bin"

FILENAMES = (
    "v01_0001.jpg",
    "v01_0112.jpg",
    "v01_0182.jpg",
    "v01_0223.jpg",
    "v01_0233.jpg",
)

CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)

COLORS = (
    (235, 64, 52),
    (239, 101, 72),
    (240, 163, 59),
    (135, 191, 61),
    (48, 176, 190),
)


def load_net(param: Path, model: Path) -> ncnn.Net:
    net = ncnn.Net()
    net.opt.num_threads = 4
    if net.load_param(str(param)) != 0:
        raise RuntimeError(f"Cannot load NCNN param: {param}")
    if net.load_model(str(model)) != 0:
        raise RuntimeError(f"Cannot load NCNN model: {model}")
    return net


def letterbox(image: Image.Image, size: int = 256) -> tuple[Image.Image, float, int, int]:
    scale = size / max(image.width, image.height)
    width = int(image.width * scale)
    height = int(image.height * scale)
    resized = image.resize((width, height), Image.Resampling.BILINEAR)
    left = (size - width) // 2
    top = (size - height) // 2
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(resized, (left, top))
    return canvas, scale, left, top


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> list[int]:
    order = scores.argsort()[::-1]
    areas = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0.0, boxes[:, 3] - boxes[:, 1]
    )
    picked = []
    while order.size:
        index = int(order[0])
        picked.append(index)
        if order.size == 1:
            break
        rest = order[1:]
        x0 = np.maximum(boxes[index, 0], boxes[rest, 0])
        y0 = np.maximum(boxes[index, 1], boxes[rest, 1])
        x1 = np.minimum(boxes[index, 2], boxes[rest, 2])
        y1 = np.minimum(boxes[index, 3], boxes[rest, 3])
        intersection = np.maximum(0.0, x1 - x0) * np.maximum(0.0, y1 - y0)
        union = areas[index] + areas[rest] - intersection
        keep = intersection / np.maximum(union, 1e-6) < threshold
        order = rest[keep]
    return picked


def detect_hands(net: ncnn.Net, image: Image.Image) -> tuple[np.ndarray, list[dict], float]:
    painted, scale, left, top = letterbox(image)
    rgb = np.asarray(painted, dtype=np.uint8)
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    mat = ncnn.Mat.from_pixels_resize(
        bgr.tobytes(),
        ncnn.Mat.PixelType.PIXEL_RGB,
        painted.width,
        painted.height,
        256,
        256,
    )
    mat.substract_mean_normalize([0.0, 0.0, 0.0], [1.0 / 255.0] * 3)

    extractor = net.create_extractor()
    extractor.input("in0", mat)
    started = time.perf_counter()
    code, output = extractor.extract("out0")
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if code != 0:
        raise RuntimeError(f"Detector extraction failed: {code}")

    raw = np.asarray(output, dtype=np.float32)
    if raw.size % 65 != 0:
        raise RuntimeError(f"Unexpected detector output shape: {raw.shape}")
    predictions = raw.reshape(-1, 65)

    proposals = []
    row_offset = 0
    for stride in (8, 16, 32):
        grid = 256 // stride
        count = grid * grid
        block = predictions[row_offset:row_offset + count]
        row_offset += count
        scores = sigmoid(block[:, 64])
        candidates = np.flatnonzero(scores >= 0.35)
        for flat_index in candidates:
            distances = block[flat_index, :64].reshape(4, 16)
            distances = distances - distances.max(axis=1, keepdims=True)
            probabilities = np.exp(distances)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            ltrb = (probabilities * np.arange(16)).sum(axis=1) * stride

            cell_y, cell_x = divmod(int(flat_index), grid)
            center_x = (cell_x + 0.5) * stride
            center_y = (cell_y + 0.5) * stride
            box = np.array([
                (center_x - ltrb[0] - left) / scale,
                (center_y - ltrb[1] - top) / scale,
                (center_x + ltrb[2] - left) / scale,
                (center_y + ltrb[3] - top) / scale,
            ])
            box[[0, 2]] = np.clip(box[[0, 2]], 0, image.width - 1)
            box[[1, 3]] = np.clip(box[[1, 3]], 0, image.height - 1)
            proposals.append((box, float(scores[flat_index])))

    if row_offset != predictions.shape[0]:
        raise RuntimeError(
            f"Detector output has {predictions.shape[0]} rows, consumed {row_offset}"
        )

    detections = []
    if proposals:
        boxes = np.stack([item[0] for item in proposals])
        scores = np.array([item[1] for item in proposals], dtype=np.float32)
        for index in nms(boxes, scores):
            detections.append({"box": boxes[index], "score": float(scores[index])})
    return painted, detections, elapsed_ms


def expand_roi(box: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    center_x = (box[0] + box[2]) / 2.0
    center_y = (box[1] + box[3]) / 2.0
    roi_width = int(np.clip((box[2] - box[0]) * 1.3, 128, 320))
    roi_height = int(np.clip((box[3] - box[1]) * 1.3, 128, 320))
    x = int(np.clip(center_x - roi_width / 2.0, 0, width - roi_width))
    y = int(np.clip(center_y - roi_height / 2.0, 0, height - roi_height))
    return x, y, x + roi_width, y + roi_height


def infer_pose(net: ncnn.Net, crop: Image.Image) -> tuple[np.ndarray, float]:
    rgb = np.asarray(crop.convert("RGB"), dtype=np.uint8)
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    mat = ncnn.Mat.from_pixels_resize(
        bgr.tobytes(),
        ncnn.Mat.PixelType.PIXEL_BGR,
        crop.width,
        crop.height,
        256,
        256,
    )
    mat.substract_mean_normalize(
        [128.0, 128.0, 128.0],
        [1.0 / 256.0, 1.0 / 256.0, 1.0 / 256.0],
    )

    extractor = net.create_extractor()
    extractor.input("in0", mat)
    started = time.perf_counter()
    code, output = extractor.extract("out0")
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if code != 0:
        raise RuntimeError(f"Handpose extraction failed: {code}")

    raw = np.asarray(output, dtype=np.float32)
    if raw.size != 42:
        raise RuntimeError(f"Unexpected handpose output shape: {raw.shape}")
    points = raw.reshape(21, 2)
    points[:, 0] *= crop.width
    points[:, 1] *= crop.height
    return points, elapsed_ms


def draw_result(
    image: Image.Image,
    detections: list[dict],
    title: str,
) -> Image.Image:
    canvas_width = min(1000, image.width)
    scale = canvas_width / image.width
    canvas = image.resize(
        (canvas_width, int(image.height * scale)),
        Image.Resampling.LANCZOS,
    )
    draw = ImageDraw.Draw(canvas)
    for detection in detections:
        box = detection["box"] * scale
        point_sets = detection["points"] * scale
        draw.rectangle(tuple(box), outline=(34, 220, 90), width=3)
        draw.text(
            (float(box[0]) + 4, float(box[1]) + 4),
            f"hand {detection['score']:.2f}",
            fill=(255, 255, 255),
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )
        for index, (start, end) in enumerate(CONNECTIONS):
            draw.line(
                (*point_sets[start], *point_sets[end]),
                fill=COLORS[index // 4],
                width=3,
            )
        for index, (x, y) in enumerate(point_sets):
            radius = 6 if index in (4, 8, 12, 16, 20) else 4
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=(255, 255, 255),
                outline=(20, 20, 20),
                width=2,
            )
            if index in (4, 8, 12, 16, 20):
                draw.text(
                    (x + 7, y - 16),
                    str(index),
                    fill=(255, 255, 255),
                    stroke_width=2,
                    stroke_fill=(0, 0, 0),
                )
    draw.rectangle((0, 0, canvas.width, 34), fill=(20, 20, 20))
    draw.text((10, 9), title, fill=(255, 255, 255))
    return canvas


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    detector = load_net(DETECTOR_PARAM, DETECTOR_MODEL)
    handpose = load_net(HANDPOSE_PARAM, HANDPOSE_MODEL)
    results = {}
    overlays = []

    for filename in FILENAMES:
        source = Image.open(IMAGE_ROOT / filename).convert("RGB")
        _, detections, detector_ms = detect_hands(detector, source)
        rendered = []
        for detection in detections:
            roi_box = expand_roi(detection["box"], source.width, source.height)
            crop = source.crop(roi_box)
            points, handpose_ms = infer_pose(handpose, crop)
            points[:, 0] += roi_box[0]
            points[:, 1] += roi_box[1]
            detection["roi_box"] = roi_box
            detection["points"] = points
            detection["handpose_ms"] = handpose_ms

        name = Path(filename).stem
        title = (
            f"{filename}  det {detector_ms:.1f} ms  "
            f"pose {sum(d['handpose_ms'] for d in detections):.1f} ms  "
            f"hands {len(detections)}"
        )
        overlay = draw_result(source, detections, title)
        output_path = OUTPUT_ROOT / f"{name}_two_stage.jpg"
        overlay.save(output_path, quality=94)
        overlays.append(overlay)
        results[name] = {
            "source": str(IMAGE_ROOT / filename),
            "detector_ms": round(detector_ms, 3),
            "hands": [
                {
                    "score": round(detection["score"], 6),
                    "box": np.round(detection["box"], 3).tolist(),
                    "roi_box": list(detection["roi_box"]),
                    "handpose_ms": round(detection["handpose_ms"], 3),
                    "points": np.round(detection["points"], 3).tolist(),
                }
                for detection in detections
            ],
            "output": str(output_path),
        }
        print(title, flush=True)

    sheet_width = max(overlay.width for overlay in overlays)
    sheet_height = sum(overlay.height for overlay in overlays)
    sheet = Image.new("RGB", (sheet_width, sheet_height), (245, 245, 245))
    y = 0
    for overlay in overlays:
        sheet.paste(overlay, (0, y))
        y += overlay.height
    sheet_path = OUTPUT_ROOT / "two_stage_sheet.jpg"
    sheet.save(sheet_path, quality=94)
    (OUTPUT_ROOT / "two_stage_results.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    print(sheet_path)


if __name__ == "__main__":
    main()
