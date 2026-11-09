from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import ncnn


ROOT = Path(__file__).resolve().parent
IMAGE_ROOT = ROOT.parents[1] / "datasets" / "fingertip_boost_20261030" / "images"
OUTPUT_ROOT = ROOT / "output"
PARAM = ROOT / "weights" / "resnet50.param"
MODEL = ROOT / "weights" / "resnet50.bin"

CROPS = {
    "v01_0001_fret": ("v01_0001.jpg", (1190, 345, 1725, 880)),
    "v01_0112_fret": ("v01_0112.jpg", (1435, 315, 1875, 755)),
    "v01_0182_fret": ("v01_0182.jpg", (1335, 275, 1795, 735)),
    "v01_0223_shift": ("v01_0223.jpg", (1165, 370, 1605, 810)),
    "v01_0233_fret": ("v01_0233.jpg", (1080, 365, 1525, 810)),
}

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


def infer(crop: Image.Image) -> tuple[np.ndarray, float]:
    array = np.asarray(crop.convert("RGB"), dtype=np.uint8)
    bgr = np.ascontiguousarray(array[:, :, ::-1])
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

    net = ncnn.Net()
    net.opt.num_threads = 4
    if net.load_param(str(PARAM)) != 0:
        raise RuntimeError(f"Cannot load NCNN param: {PARAM}")
    if net.load_model(str(MODEL)) != 0:
        raise RuntimeError(f"Cannot load NCNN model: {MODEL}")

    extractor = net.create_extractor()
    extractor.input("in0", mat)
    started = time.perf_counter()
    code, output = extractor.extract("out0")
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if code != 0:
        raise RuntimeError(f"NCNN extraction failed: {code}")

    raw = np.asarray(output, dtype=np.float32)
    print(
        f"NCNN output: dims={output.dims} w={output.w} h={output.h} c={output.c} "
        f"shape={raw.shape}",
        flush=True,
    )
    if raw.size == 42:
        raw = raw.reshape(21, 2)
    elif raw.size == 21 * 2 * 5 * 5:
        raw = raw.reshape(21, 2, 5, 5).mean(axis=(2, 3)).T
    else:
        raise RuntimeError(f"Unsupported output shape: {raw.shape}")
    points = raw
    points[:, 0] *= crop.width
    points[:, 1] *= crop.height
    return points, elapsed_ms


def render(crop: Image.Image, points: np.ndarray, title: str) -> Image.Image:
    scale = 2
    canvas = crop.resize((crop.width * scale, crop.height * scale), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(canvas)
    scaled = points * scale

    for index, (start, end) in enumerate(CONNECTIONS):
        color = COLORS[index // 4]
        draw.line((*scaled[start], *scaled[end]), fill=color, width=4)
    for index, (x, y) in enumerate(scaled):
        radius = 7 if index in (4, 8, 12, 16, 20) else 5
        draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                     fill=(255, 255, 255), outline=(20, 20, 20), width=2)
        if index in (4, 8, 12, 16, 20):
            draw.text((x + 8, y - 18), str(index), fill=(255, 255, 255), stroke_width=2,
                      stroke_fill=(0, 0, 0))
    draw.rectangle((0, 0, canvas.width, 40), fill=(20, 20, 20))
    draw.text((12, 10), title, fill=(255, 255, 255))
    return canvas


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = {}
    overlays = []
    for name, (filename, box) in CROPS.items():
        source = Image.open(IMAGE_ROOT / filename).convert("RGB")
        crop = source.crop(box)
        points, elapsed_ms = infer(crop)
        overlay = render(crop, points, f"{name}  {elapsed_ms:.1f} ms")
        output_path = OUTPUT_ROOT / f"{name}.jpg"
        overlay.save(output_path, quality=94)
        overlays.append(overlay)
        results[name] = {
            "source": str(IMAGE_ROOT / filename),
            "crop_box": box,
            "inference_ms": round(elapsed_ms, 3),
            "crop_points": np.round(points, 3).tolist(),
            "output": str(output_path),
        }
        print(f"{name}: {elapsed_ms:.1f} ms")

    sheet = Image.new("RGB", (1024, 1024), (245, 245, 245))
    placements = ((0, 0), (512, 0), (0, 512), (512, 512), (0, 830))
    for overlay, placement in zip(overlays, placements):
        thumb = overlay.resize((512, 512), Image.Resampling.LANCZOS)
        sheet.paste(thumb, placement)
    sheet_path = OUTPUT_ROOT / "comparison_sheet.jpg"
    sheet.save(sheet_path, quality=94)
    (OUTPUT_ROOT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(sheet_path)


if __name__ == "__main__":
    main()
