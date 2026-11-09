#!/usr/bin/env python3
"""Extract an evenly spaced, reproducible video frame dataset."""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


DEFAULT_FFMPEG_CANDIDATES = (
    r"C:\Program Files (x86)\Lenovo\LegionZone\2.0.29.9292\SEGamingAI\services\editor\ffmpeg.exe",
)


def find_ffmpeg(explicit_path):
    if explicit_path:
        path = Path(explicit_path)
        if path.is_file():
            return str(path)
        raise FileNotFoundError("ffmpeg not found: {}".format(path))

    path_from_env = shutil.which("ffmpeg")
    if path_from_env:
        return path_from_env

    for candidate in DEFAULT_FFMPEG_CANDIDATES:
        if Path(candidate).is_file():
            return candidate
    raise FileNotFoundError("ffmpeg was not found; pass it with --ffmpeg")


def run_ffmpeg(command):
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(
            "ffmpeg failed with exit code {}:\n{}".format(
                completed.returncode, stderr[-8000:]
            )
        )
    return completed


def probe_video(ffmpeg, source):
    command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-f",
        "null",
        os.devnull,
    ]
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    log = completed.stderr.decode("utf-8", errors="replace")
    if completed.returncode != 0:
        raise RuntimeError("Could not decode {}:\n{}".format(source, log[-8000:]))

    duration_match = re.search(
        r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", log
    )
    if not duration_match:
        raise RuntimeError("Could not read duration for {}".format(source))
    hours, minutes, seconds = duration_match.groups()
    duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)

    video_line_match = re.search(r"Stream #\d+:\d+.*?Video:.*", log)
    if not video_line_match:
        raise RuntimeError("Could not read a video stream for {}".format(source))
    video_line = video_line_match.group(0)
    size_match = re.search(r",\s*(\d+)x(\d+)(?:\s|\[|,)", video_line)
    fps_match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s+fps", video_line)
    if not size_match or not fps_match:
        raise RuntimeError("Could not read resolution or fps for {}".format(source))

    frame_counts = [
        int(value) for value in re.findall(r"frame=\s*(\d+)", log)
    ]
    if not frame_counts:
        raise RuntimeError("Could not count decoded frames for {}".format(source))

    width = int(size_match.group(1))
    height = int(size_match.group(2))
    fps = float(fps_match.group(1))
    return {
        "path": str(source.resolve()),
        "file_name": source.name,
        "size_bytes": source.stat().st_size,
        "duration_seconds": round(duration, 6),
        "fps": round(fps, 6),
        "frame_count": frame_counts[-1],
        "width": width,
        "height": height,
    }


def allocate_counts(probes, total):
    durations = np.asarray(
        [probe["duration_seconds"] for probe in probes], dtype=np.float64
    )
    exact = durations * float(total) / float(durations.sum())
    counts = np.floor(exact).astype(np.int64)
    remainder = total - int(counts.sum())
    order = np.argsort(-(exact - counts), kind="stable")
    for index in order[:remainder]:
        counts[index] += 1
    return [int(value) for value in counts], exact.tolist()


def extract_segment(ffmpeg, source, output_dir, count, segment_number):
    metadata = probe_video(ffmpeg, source)
    sample_fps = float(count) / metadata["duration_seconds"]
    output_pattern = output_dir / "v{:02d}_%04d.jpg".format(segment_number)
    command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-vf",
        "fps={:.12f}".format(sample_fps),
        "-frames:v",
        str(count),
        "-q:v",
        "2",
        "-start_number",
        "1",
        str(output_pattern),
    ]
    run_ffmpeg(command)

    extracted = sorted(output_dir.glob("v{:02d}_*.jpg".format(segment_number)))
    if len(extracted) != count:
        raise RuntimeError(
            "Expected {} frames from {}, found {}".format(
                count, source.name, len(extracted)
            )
        )
    return metadata, extracted


def validate_images(image_paths):
    records = []
    blank_files = []
    dimensions = set()
    for index, path in enumerate(image_paths, start=1):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("Could not decode {}".format(path))
        height, width = image.shape[:2]
        dimensions.add((width, height))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean = float(gray.mean())
        stddev = float(gray.std())
        if stddev < 2.0 or mean < 3.0 or mean > 252.0:
            blank_files.append(path.name)
        records.append(
            {
                "file": path.name,
                "width": width,
                "height": height,
                "mean_luma": round(mean, 3),
                "std_luma": round(stddev, 3),
                "size_bytes": path.stat().st_size,
            }
        )
        if index % 100 == 0:
            print("Validated {}/{} images".format(index, len(image_paths)))
    return records, sorted(dimensions), blank_files


def create_contact_sheets(image_paths, contact_dir):
    contact_dir.mkdir(parents=True, exist_ok=True)
    sample_count = min(100, len(image_paths))
    indices = np.linspace(
        0, len(image_paths) - 1, num=sample_count, dtype=np.int64
    )
    tile_width = 320
    tile_height = 180
    columns = 5
    rows = 5
    sheets = []

    for sheet_index, start in enumerate(range(0, sample_count, columns * rows), start=1):
        sheet_indices = indices[start : start + columns * rows]
        canvas = np.full(
            (rows * tile_height, columns * tile_width, 3),
            24,
            dtype=np.uint8,
        )
        for tile_index, dataset_index in enumerate(sheet_indices):
            image = cv2.imread(str(image_paths[int(dataset_index)]), cv2.IMREAD_COLOR)
            image = cv2.resize(
                image, (tile_width, tile_height), interpolation=cv2.INTER_AREA
            )
            row = tile_index // columns
            column = tile_index % columns
            y0 = row * tile_height
            x0 = column * tile_width
            canvas[y0 : y0 + tile_height, x0 : x0 + tile_width] = image
            label = "{}  {}".format(
                int(dataset_index) + 1, image_paths[int(dataset_index)].name
            )
            cv2.rectangle(
                canvas, (x0, y0 + tile_height - 24), (x0 + tile_width, y0 + tile_height),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                canvas,
                label,
                (x0 + 5, y0 + tile_height - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        sheet_path = contact_dir / "contact_{:02d}.jpg".format(sheet_index)
        if not cv2.imwrite(str(sheet_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise RuntimeError("Could not write {}".format(sheet_path))
        sheets.append(str(sheet_path.resolve()))
    return sheets


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ffmpeg", default=None)
    parser.add_argument("--total", type=int, default=1000)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(r"D:\FPGA\ZynqAudioLab\datasets\fingertip_boost_20261030"),
    )
    parser.add_argument("sources", nargs="+", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.total <= 0:
        raise ValueError("--total must be positive")
    if len(args.sources) == 0:
        raise ValueError("At least one source video is required")

    ffmpeg = find_ffmpeg(args.ffmpeg)
    sources = [source.resolve() for source in args.sources]
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(str(source))

    output_root = args.output_root.resolve()
    image_dir = output_root / "images"
    contact_dir = output_root / "contact_sheets"
    image_dir.mkdir(parents=True, exist_ok=True)
    existing = list(image_dir.glob("v*.jpg"))
    if existing:
        raise FileExistsError(
            "{} already contains {} image files; choose another --output-root".format(
                image_dir, len(existing)
            )
        )

    print("Probing {} source videos".format(len(sources)))
    probes = [probe_video(ffmpeg, source) for source in sources]
    counts, exact_counts = allocate_counts(probes, args.total)

    print("Allocated frame counts: {}".format(counts))
    all_metadata = []
    all_images = []
    for segment_index, (source, probe, count, exact) in enumerate(
        zip(sources, probes, counts, exact_counts), start=1
    ):
        print(
            "Extracting {} frames from {} ({:.3f} exact share)".format(
                count, source.name, exact
            )
        )
        metadata, images = extract_segment(
            ffmpeg, source, image_dir, count, segment_index
        )
        all_metadata.append(metadata)
        all_images.extend(images)

    if len(all_images) != args.total:
        raise RuntimeError(
            "Expected {} images, found {}".format(args.total, len(all_images))
        )

    print("Validating images")
    records, dimensions, blank_files = validate_images(all_images)
    if len(dimensions) != 1:
        raise RuntimeError("Mixed image dimensions: {}".format(dimensions))
    if blank_files:
        raise RuntimeError(
            "{} sampled images appear blank: {}".format(
                len(blank_files), ", ".join(blank_files[:20])
            )
        )

    print("Creating contact sheets")
    sheets = create_contact_sheets(all_images, contact_dir)

    segment_records = []
    offset = 0
    for index, (metadata, count) in enumerate(zip(all_metadata, counts), start=1):
        segment_files = records[offset : offset + count]
        sample_interval = metadata["duration_seconds"] / float(count)
        for local_index, record in enumerate(segment_files, start=1):
            sample_time = (float(local_index) - 0.5) * sample_interval
            record["segment"] = index
            record["segment_index"] = local_index
            record["sample_time_seconds"] = round(sample_time, 6)
            record["estimated_source_frame"] = int(
                round(sample_time * metadata["fps"])
            )
        metadata["requested_frames"] = count
        metadata["extracted_frames"] = len(segment_files)
        segment_records.append(metadata)
        offset += count

    manifest = {
        "dataset": "fingertip_boost_20261030",
        "created_local_time": datetime.now().astimezone().isoformat(),
        "generator": str(Path(__file__).resolve()),
        "ffmpeg": ffmpeg,
        "output_root": str(output_root),
        "image_directory": str(image_dir),
        "requested_images": args.total,
        "extracted_images": len(records),
        "sampling": "uniform in time within each source; original resolution retained",
        "jpeg_quality_argument": 2,
        "image_dimensions": {
            "width": dimensions[0][0],
            "height": dimensions[0][1],
        },
        "segments": segment_records,
        "contact_sheets": sheets,
        "images": records,
    }
    manifest_path = output_root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    summary = {
        "output_root": str(output_root),
        "images": len(records),
        "dimensions": manifest["image_dimensions"],
        "counts_per_video": counts,
        "contact_sheets": len(sheets),
        "manifest": str(manifest_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("ERROR: {}".format(error), file=sys.stderr)
        sys.exit(1)
