from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2 as cv
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
PROJECT = Path(__file__).resolve().parents[2]
IMAGE_ROOT = PROJECT / "datasets" / "fingertip_boost_20261030" / "images"
OUTPUT_ROOT = ROOT / "output"
MODEL_ROOT = PROJECT / "models" / "hand"

sys.path.insert(0, str(PROJECT / "diagnostics"))
sys.path.insert(0, str(PROJECT / "pynq"))

from upstream_mp_palmdet import MPPalmDet  # noqa: E402
from handpose_model import MPHandPose  # noqa: E402


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


def draw_result(image: Image.Image, hands: list[dict], title: str) -> Image.Image:
    canvas_width = min(1000, image.width)
    scale = canvas_width / image.width
    canvas = image.resize(
        (canvas_width, int(image.height * scale)),
        Image.Resampling.LANCZOS,
    )
    draw = ImageDraw.Draw(canvas)
    for hand in hands:
        bbox = hand["bbox"] * scale
        points = hand["points"] * scale
        draw.rectangle(
            (bbox[0, 0], bbox[0, 1], bbox[1, 0], bbox[1, 1]),
            outline=(42, 130, 255),
            width=3,
        )
        draw.text(
            (float(bbox[0, 0]) + 4, float(bbox[0, 1]) + 4),
            f"mp {hand['score']:.2f}",
            fill=(255, 255, 255),
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )
        for index, (start, end) in enumerate(CONNECTIONS):
            draw.line(
                (*points[start], *points[end]),
                fill=COLORS[index // 4],
                width=3,
            )
        for index, (x, y) in enumerate(points):
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
    palm_detector = MPPalmDet(
        str(MODEL_ROOT / "palm_detection_mediapipe_2023feb.onnx")
    )
    handpose = MPHandPose(
        str(MODEL_ROOT / "handpose_estimation_mediapipe_2023feb.onnx"),
        confThreshold=0.2,
    )
    results = {}
    overlays = []

    for filename in FILENAMES:
        bgr = cv.imread(str(IMAGE_ROOT / filename))
        if bgr is None:
            raise RuntimeError(f"Cannot read image: {filename}")
        palms = palm_detector.infer(bgr)
        hands = []
        for palm in palms:
            pose = handpose.infer(bgr, palm)
            if pose is None:
                continue
            hands.append(
                {
                    "bbox": pose[:4].reshape(2, 2),
                    "points": pose[4:67].reshape(21, 3)[:, :2],
                    "score": float(pose[-1]),
                }
            )

        source = Image.fromarray(cv.cvtColor(bgr, cv.COLOR_BGR2RGB))
        title = f"{filename}  palms {len(palms)}  accepted {len(hands)}"
        overlay = draw_result(source, hands, title)
        output_path = OUTPUT_ROOT / f"{Path(filename).stem}_mediapipe.jpg"
        overlay.save(output_path, quality=94)
        overlays.append(overlay)
        results[Path(filename).stem] = {
            "source": str(IMAGE_ROOT / filename),
            "palm_count": len(palms),
            "hands": [
                {
                    "score": hand["score"],
                    "bbox": hand["bbox"].round(3).tolist(),
                    "points": hand["points"].round(3).tolist(),
                }
                for hand in hands
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
    sheet_path = OUTPUT_ROOT / "mediapipe_sheet.jpg"
    sheet.save(sheet_path, quality=94)
    (OUTPUT_ROOT / "mediapipe_results.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    print(sheet_path)


if __name__ == "__main__":
    main()
