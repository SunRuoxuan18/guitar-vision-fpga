from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
from PIL import Image, ImageDraw

from run_detector_trial import (
    HANDPOSE_MODEL,
    HANDPOSE_PARAM,
    infer_pose,
    load_net,
)


ROOT = Path(__file__).resolve().parent
PROJECT = Path(__file__).resolve().parents[2]
IMAGE_ROOT = PROJECT / "datasets" / "fingertip_boost_20261030" / "images"
OUTPUT_ROOT = ROOT / "output"
MODEL_ROOT = PROJECT / "models" / "hand"

sys.path.insert(0, str(PROJECT / "diagnostics"))
sys.path.insert(0, str(PROJECT / "pynq"))

from handpose_model import MPHandPose  # noqa: E402
from upstream_mp_palmdet import MPPalmDet  # noqa: E402


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


def render_crop(
    crop: Image.Image,
    points: np.ndarray | None,
    title: str,
    width: int = 420,
) -> Image.Image:
    scale = width / crop.width
    rendered = crop.resize(
        (width, int(crop.height * scale)),
        Image.Resampling.LANCZOS,
    )
    draw = ImageDraw.Draw(rendered)
    if points is not None:
        scaled = points * scale
        for index, (start, end) in enumerate(CONNECTIONS):
            draw.line(
                (*scaled[start], *scaled[end]),
                fill=COLORS[index // 4],
                width=3,
            )
        for index, (x, y) in enumerate(scaled):
            radius = 5 if index in (4, 8, 12, 16, 20) else 3
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=(255, 255, 255),
                outline=(20, 20, 20),
                width=2,
            )
            if index in (4, 8, 12, 16, 20):
                draw.text(
                    (x + 6, y - 15),
                    str(index),
                    fill=(255, 255, 255),
                    stroke_width=2,
                    stroke_fill=(0, 0, 0),
                )
    else:
        draw.text((12, 40), "no accepted pose", fill=(255, 255, 255))
    draw.rectangle((0, 0, rendered.width, 28), fill=(20, 20, 20))
    draw.text((8, 7), title, fill=(255, 255, 255))
    return rendered


def original_handpose_x_roi(
    box_xyxy: np.ndarray,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    side = max(
        abs(float(box_xyxy[2] - box_xyxy[0])),
        abs(float(box_xyxy[3] - box_xyxy[1])),
    ) * 1.1
    center_x = (box_xyxy[0] + box_xyxy[2]) / 2.0
    center_y = (box_xyxy[1] + box_xyxy[3]) / 2.0
    x0 = int(np.clip(center_x - side / 2.0, 0, image_width - 1))
    x1 = int(np.clip(center_x + side / 2.0, 1, image_width))
    y0 = int(np.clip(center_y - side / 2.0, 0, image_height - 1))
    y1 = int(np.clip(center_y + side / 2.0, 1, image_height))
    return x0, y0, x1, y1


def infer_upright_handpose_x(
    handpose_x,
    bgr: np.ndarray,
    palm: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    palm_box = palm[0:4].reshape(2, 2)
    palm_landmarks = palm[4:18].reshape(7, 2)
    palm_base = palm_landmarks[0]
    middle_base = palm_landmarks[2]
    radians = np.pi / 2 - np.arctan2(
        -(middle_base[1] - palm_base[1]),
        middle_base[0] - palm_base[0],
    )
    angle = float(np.rad2deg(radians))
    center = tuple(float(value) for value in palm_box.mean(axis=0))
    rotation = cv.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv.warpAffine(
        bgr,
        rotation,
        (bgr.shape[1], bgr.shape[0]),
        flags=cv.INTER_LINEAR,
        borderMode=cv.BORDER_CONSTANT,
    )

    rotated_palm_landmarks = palm_landmarks @ rotation[:, :2].T + rotation[:, 2]
    rotated_min = rotated_palm_landmarks.min(axis=0)
    rotated_max = rotated_palm_landmarks.max(axis=0)
    roi = original_handpose_x_roi(
        np.array([
            rotated_min[0],
            rotated_min[1],
            rotated_max[0],
            rotated_max[1],
        ]),
        bgr.shape[1],
        bgr.shape[0],
    )
    rotated_rgb = cv.cvtColor(rotated, cv.COLOR_BGR2RGB)
    crop = Image.fromarray(rotated_rgb).crop(roi)
    points_rotated, elapsed_ms = infer_pose(handpose_x, crop)
    points_rotated += np.array([roi[0], roi[1]], dtype=np.float32)

    inverse = cv.invertAffineTransform(rotation)
    points_original = points_rotated @ inverse[:, :2].T + inverse[:, 2]
    return points_original.astype(np.float32), elapsed_ms, angle


def comparison_roi(
    seed_roi: tuple[int, int, int, int],
    point_sets: tuple[np.ndarray | None, ...],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    xs = [seed_roi[0], seed_roi[2]]
    ys = [seed_roi[1], seed_roi[3]]
    for points in point_sets:
        if points is None:
            continue
        xs.extend(float(value) for value in points[:, 0])
        ys.extend(float(value) for value in points[:, 1])
    side = max(max(xs) - min(xs), max(ys) - min(ys)) * 1.2
    center_x = (min(xs) + max(xs)) / 2.0
    center_y = (min(ys) + max(ys)) / 2.0
    x0 = int(np.clip(center_x - side / 2.0, 0, image_width - 1))
    x1 = int(np.clip(center_x + side / 2.0, 1, image_width))
    y0 = int(np.clip(center_y - side / 2.0, 0, image_height - 1))
    y1 = int(np.clip(center_y + side / 2.0, 1, image_height))
    return x0, y0, x1, y1


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    palm_detector = MPPalmDet(
        str(MODEL_ROOT / "palm_detection_mediapipe_2023feb.onnx")
    )
    mediapipe = MPHandPose(
        str(MODEL_ROOT / "handpose_estimation_mediapipe_2023feb.onnx"),
        confThreshold=0.2,
    )
    handpose_x = load_net(HANDPOSE_PARAM, HANDPOSE_MODEL)
    results = {}
    rows = []

    for filename in FILENAMES:
        bgr = cv.imread(str(IMAGE_ROOT / filename))
        if bgr is None:
            raise RuntimeError(f"Cannot read image: {filename}")
        palms = palm_detector.infer(bgr)
        source = Image.fromarray(cv.cvtColor(bgr, cv.COLOR_BGR2RGB))
        frame_rows = []

        for index, palm in enumerate(palms[:2]):
            palm_box = palm[0:4].reshape(2, 2)
            box_xyxy = np.array(
                [palm_box[0, 0], palm_box[0, 1], palm_box[1, 0], palm_box[1, 1]]
            )
            roi = original_handpose_x_roi(box_xyxy, source.width, source.height)

            x_points, handpose_ms = infer_pose(handpose_x, source.crop(roi))
            x_points += np.array([roi[0], roi[1]], dtype=np.float32)
            upright_points, upright_ms, upright_angle = infer_upright_handpose_x(
                handpose_x,
                bgr,
                palm,
            )
            mediapipe_pose = mediapipe.infer(bgr, palm)
            if mediapipe_pose is None:
                mediapipe_points = None
                mediapipe_score = None
            else:
                mediapipe_points = mediapipe_pose[4:67].reshape(21, 3)[:, :2]
                mediapipe_score = float(mediapipe_pose[-1])

            display_roi = comparison_roi(
                roi,
                (x_points, upright_points, mediapipe_points),
                source.width,
                source.height,
            )
            crop = source.crop(display_roi)
            display_origin = np.array(
                [display_roi[0], display_roi[1]],
                dtype=np.float32,
            )
            left = render_crop(
                crop,
                x_points - display_origin,
                f"{filename} #{index} handpose_x {handpose_ms:.1f} ms",
            )
            middle = render_crop(
                crop,
                upright_points - display_origin,
                (
                    f"{filename} #{index} upright "
                    f"{upright_angle:.0f}deg {upright_ms:.1f} ms"
                ),
            )
            right = render_crop(
                crop,
                (
                    mediapipe_points - display_origin
                    if mediapipe_points is not None
                    else None
                ),
                (
                    f"{filename} #{index} mediapipe "
                    f"{mediapipe_score:.2f}"
                    if mediapipe_score is not None
                    else f"{filename} #{index} mediapipe rejected"
                ),
            )
            row = Image.new(
                "RGB",
                (left.width + middle.width + right.width, left.height),
                (245, 245, 245),
            )
            row.paste(left, (0, 0))
            row.paste(middle, (left.width, 0))
            row.paste(right, (left.width + middle.width, 0))
            frame_rows.append(row)

            results.setdefault(Path(filename).stem, []).append(
                {
                    "hand_index": index,
                    "roi_box": list(roi),
                    "display_roi": list(display_roi),
                    "palm_score": float(palm[-1]),
                    "handpose_x_ms": round(handpose_ms, 3),
                    "upright_angle": round(upright_angle, 3),
                    "upright_ms": round(upright_ms, 3),
                    "mediapipe_score": mediapipe_score,
                    "handpose_x_points": np.round(x_points, 3).tolist(),
                    "upright_points": np.round(upright_points, 3).tolist(),
                    "mediapipe_points": (
                        None
                        if mediapipe_points is None
                        else np.round(mediapipe_points, 3).tolist()
                    ),
                }
            )

        if not frame_rows:
            placeholder = Image.new("RGB", (1260, 100), (245, 245, 245))
            ImageDraw.Draw(placeholder).text(
                (12, 40),
                f"{filename}: MediaPipe palm detector found no hand",
                fill=(20, 20, 20),
            )
            frame_rows.append(placeholder)
        rows.extend(frame_rows)

    sheet_width = max(row.width for row in rows)
    sheet_height = sum(row.height for row in rows)
    sheet = Image.new("RGB", (sheet_width, sheet_height), (245, 245, 245))
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height
    sheet_path = OUTPUT_ROOT / "three_way_comparison_sheet.jpg"
    sheet.save(sheet_path, quality=94)
    (OUTPUT_ROOT / "three_way_results.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    print(sheet_path)


if __name__ == "__main__":
    main()
