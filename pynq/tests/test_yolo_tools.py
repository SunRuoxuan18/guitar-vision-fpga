"""Dataset validation and export sidecar tests using temporary synthetic data."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "vision"))
from yolo_dataset import validate_dataset
from yolo_export import inspect_onnx
from test_yolo_detector import HAS_CV, HAS_ONNX, build_tiny_onnx


def _cli_has_onnx():
    """The metadata CLI runs in a clean child, not this suite's sys.path."""
    try:
        process = subprocess.run([sys.executable, "-B", "-c", "import onnx"],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                 timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return process.returncode == 0


HAS_CLI_ONNX = _cli_has_onnx()


if HAS_CV:
    import cv2
    import numpy as np


@unittest.skipUnless(HAS_CV, "OpenCV/numpy required")
class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = self.root / "dataset.json"
        self.manifest.write_text(json.dumps({"path": ".", "names": ["fretboard"],
                                             "train": "images/train", "val": "images/val"}), encoding="utf-8")
        for index, split in enumerate(("train", "val")):
            (self.root / "images" / split).mkdir(parents=True)
            (self.root / "labels" / split).mkdir(parents=True)
            image = np.full((32, 64, 3), index * 25, np.uint8)
            self.assertTrue(cv2.imwrite(str(self.root / "images" / split / "a.png"), image))
            (self.root / "labels" / split / "a.txt").write_text("0 0.5 0.5 0.5 0.25\n", encoding="utf-8")

    def test_valid_manifest_counts_and_explicit_negatives(self):
        result = validate_dataset(self.manifest, ["fretboard"])
        self.assertEqual(result["status"], "valid", result)
        self.assertEqual(result["splits"]["train"]["boxes"], 1)
        image = np.zeros((32, 64, 3), np.uint8)
        cv2.imwrite(str(self.root / "images/train/negative.png"), image)
        (self.root / "labels/train/negative.txt").write_text("", encoding="utf-8")
        result = validate_dataset(self.manifest)
        self.assertEqual(result["status"], "valid", result)
        self.assertEqual(result["splits"]["train"]["negative_images"], 1)
        self.assertFalse(result["semantic_labels_verified"])

    def test_bad_class_nonfinite_box_and_outside_bounds(self):
        label = self.root / "labels/train/a.txt"
        for line in ("1 0.5 0.5 0.2 0.2", "0 nan 0.5 0.2 0.2", "0 0.5 0.5 inf 0.2",
                     "0 0.5 0.5 0 0.2", "0 0.95 0.5 0.2 0.2", "0.0 0.5 0.5 0.2 0.2",
                     "0 0.5 0.5 0.2", "0 0.5 0.5 0.2 0.2 0.7"):
            label.write_text(line, encoding="utf-8")
            with self.subTest(line=line):
                self.assertEqual(validate_dataset(self.manifest)["status"], "invalid")

    def test_duplicate_content_missing_orphan_labels_and_unreadable_image(self):
        original = self.root / "images/train/a.png"
        (self.root / "images/val/a.png").write_bytes(original.read_bytes())
        result = validate_dataset(self.manifest)
        self.assertTrue(any("leaks" in error for error in result["errors"]), result)
        (self.root / "labels/train/a.txt").unlink()
        (self.root / "labels/train/orphan.txt").write_text("", encoding="utf-8")
        original.write_bytes(b"not an image")
        result = validate_dataset(self.manifest)
        self.assertTrue(any("missing label" in error for error in result["errors"]), result)
        self.assertTrue(any("without matching image" in error for error in result["errors"]), result)
        self.assertTrue(any("unreadable image" in error for error in result["errors"]), result)

    def test_empty_image_and_malformed_json_are_invalid_results(self):
        (self.root / "images/train/a.png").write_bytes(b"")
        result = validate_dataset(self.manifest)
        self.assertEqual(result["status"], "invalid")
        self.assertTrue(any("unreadable image" in error for error in result["errors"]), result)
        self.manifest.write_text("{broken", encoding="utf-8")
        self.assertEqual(validate_dataset(self.manifest)["status"], "invalid")

    def test_names_required_class_coverage_and_manifest_validation(self):
        self.assertEqual(validate_dataset(self.manifest, ["guitar"])["status"], "invalid")
        for data in ({}, [], {"names": ["x", "x"]}, {"names": ["x"], "nc": 2},
                     {"names": ["x"], "path": "absent"},
                     {"names": ["x"], "train": "../images", "val": "images/val"}):
            self.manifest.write_text(json.dumps(data), encoding="utf-8")
            self.assertEqual(validate_dataset(self.manifest)["status"], "invalid")


@unittest.skipUnless(HAS_CV and HAS_ONNX, "ONNX authoring dependency required")
class ExportMetadataTests(unittest.TestCase):
    def test_corrupt_graph_returns_useful_validation_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corrupt.onnx"
            path.write_bytes(b"not a protobuf graph")
            with self.assertRaisesRegex(ValueError, "invalid ONNX graph"):
                inspect_onnx(path, ["fretboard"], "yolov8")

    @unittest.skipUnless(HAS_CLI_ONNX, "ONNX must be importable by metadata CLI interpreter")
    def test_real_graph_layout_hash_and_metadata_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            classes = ["fretboard", "fingertip"]
            for layout in ("BCN", "BNC"):
                path = directory / (layout + ".onnx")
                build_tiny_onnx(path, layout)
                result = inspect_onnx(path, classes, "yolo11")
                self.assertEqual(result["output_layout"], layout)
                self.assertEqual(result["input_size"], [32, 32])
                self.assertEqual(len(result["onnx_sha256"]), 64)
                self.assertFalse(result["validation"]["target_accuracy_measured"])
                self.assertFalse(result["validation"]["board_opencv_4_2_verified"])
                with self.assertRaises(ValueError):
                    inspect_onnx(path, ["wrong_count"], "yolo11")
            classes_path = directory / "classes.json"
            classes_path.write_text(json.dumps(classes), encoding="utf-8")
            metadata_path = directory / "metadata.json"
            cli = Path(__file__).resolve().parents[2] / "vision/yolo_export.py"
            args = [sys.executable, "-B", str(cli), "metadata", "--onnx", str(path),
                    "--classes", str(classes_path), "--family", "yolo11", "--metadata", str(metadata_path),
                    "--weights-license", "synthetic-test-only"]
            process = subprocess.run(args, capture_output=True, text=True, timeout=30)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            self.assertEqual(json.loads(metadata_path.read_text())["classes"], classes)
            self.assertEqual(json.loads(metadata_path.read_text())["weights_license"], "synthetic-test-only")
            process = subprocess.run(args, capture_output=True, text=True, timeout=30)
            self.assertEqual(process.returncode, 2)
            self.assertIn("already exists", json.loads(process.stdout)["reason"])

    def test_export_missing_checkpoint_never_downloads(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            classes = directory / "classes.json"
            classes.write_text('["fretboard"]', encoding="utf-8")
            cli = Path(__file__).resolve().parents[2] / "vision/yolo_export.py"
            process = subprocess.run([sys.executable, "-B", str(cli), "export", "--checkpoint",
                                      str(directory / "missing.pt"), "--classes", str(classes),
                                      "--family", "yolov8", "--metadata", str(directory / "model.json"),
                                      "--weights-license", "synthetic-test-only"],
                                     capture_output=True, text=True, timeout=30)
            self.assertEqual(process.returncode, 2, process.stderr)
            self.assertIn("existing local", json.loads(process.stdout)["reason"])


if __name__ == "__main__":
    unittest.main()
