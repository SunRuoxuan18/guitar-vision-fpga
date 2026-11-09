"""Host-only differential checks against the pre-optimization tracker step."""

import math
from pathlib import Path
import sys
import unittest
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pynq"))
from fingertip_heatmap import POINT_NAMES
from fingertip_tracking import FourTipTracker


class OriginalTracker(FourTipTracker):
    """Frozen original patch extraction and step, including native cutoffs."""

    @classmethod
    def _patch(cls, gray, point):
        x, y = (float(v) for v in point)
        radius = cls.PATCH_SIZE // 2
        if (not math.isfinite(x) or not math.isfinite(y)
                or not radius <= x < gray.shape[1] - radius
                or not radius <= y < gray.shape[0] - radius):
            return None
        return cv2.getRectSubPix(gray, (cls.PATCH_SIZE, cls.PATCH_SIZE), (x, y)).astype(np.float32)

    def _step(self, previous, current):
        names = list(self._tips)
        points = np.array([self._tips[name]["point"] for name in names], np.float32).reshape(-1, 1, 2)
        try:
            moved, forward, _ = self._flow(previous, current, points)
            moved = np.asarray(moved, np.float32).reshape(-1, 1, 2).copy()
            forward = np.asarray(forward).reshape(-1)
            if len(moved) != len(names) or len(forward) != len(names):
                raise ValueError("Invalid forward LK output")
            patches = {i: self._patch(current, moved[i, 0]) for i in range(len(names))
                       if forward[i] == 1}
            valid = [i for i, patch in patches.items() if patch is not None]
            for i, name in enumerate(names):
                if i not in valid:
                    self._losses[name] = "forward_lk"
                    self._tips.pop(name, None)
            if not valid:
                return
            reverse_points = moved[valid].copy()
            returned, backward, _ = self._flow(current, previous, reverse_points)
            returned = np.asarray(returned, np.float32).reshape(-1, 2).copy()
            backward = np.asarray(backward).reshape(-1)
            if len(returned) != len(valid) or len(backward) != len(valid):
                raise ValueError("Invalid reverse LK output")
            old_patches = [self._patch(previous, points[i, 0]) for i in valid]
            metrics = None
            if self._patch_quality is not None:
                metrics = self._patch_quality.evaluate(
                    np.stack(old_patches), np.stack([patches[i] for i in valid]),
                    np.stack([self._tips[names[i]]["anchor"] for i in valid]),
                    points[valid, 0].copy(), returned)
            for j, i in enumerate(valid):
                name = names[i]
                tip = self._tips[name]
                old_patch, patch = old_patches[j], patches[i]
                if metrics is not None:
                    fb, corr, anchor_corr, texture, old_texture = (float(v) for v in metrics[j])
                if metrics is None or any(abs(value - boundary) <= 1e-5 for value, boundary in
                        zip((fb, corr, anchor_corr, texture, old_texture), (1.5, .88, .88, 5., 5.))):
                    fb = float(np.linalg.norm(returned[j] - points[i, 0]))
                    texture, old_texture = float(patch.std()), float(old_patch.std())
                    corr = self._correlation(old_patch, patch)
                    anchor_corr = self._correlation(tip["anchor"], patch)
                quality = {"fb_error_px": fb, "patch_correlation": corr,
                           "anchor_correlation": anchor_corr, "texture": texture}
                reason = None
                if backward[j] != 1 or not math.isfinite(fb) or fb >= 1.5:
                    reason = "forward_backward"
                elif texture < 5 or old_texture < 5:
                    reason = "texture"
                elif corr < .88 or anchor_corr < .88:
                    reason = "patch_correlation"
                if reason:
                    self._losses[name] = reason
                    self._tips.pop(name, None)
                else:
                    tip["point"] = moved[i, 0].copy()
                    self._quality[name] = quality
        except Exception as exc:
            self._lose_all("flow_backend:" + type(exc).__name__)
            if getattr(self._flow_backend, "fault", False):
                if isinstance(exc, (OSError, RuntimeError)):
                    raise
                raise RuntimeError("Fingertip flow backend is faulted") from exc


class Clock:
    value = 100.0

    def __call__(self):
        return self.value


class IdentityFlow:
    fault = False
    backend_name = "host_identity"

    def __call__(self, previous, current, points):
        return points.copy(), np.ones((len(points), 1), np.uint8), None


class BatchQuality:
    """Check native input contracts and use float64 statistics on real patches."""

    def __init__(self, metrics=None):
        self.metrics = metrics
        self.calls = []

    def evaluate(self, previous, current, anchors, original, returned):
        arrays = previous, current, anchors, original, returned
        count = len(previous)
        for array, shape in zip(arrays, [(count, 11, 11)] * 3 + [(count, 2)] * 2):
            if (array.dtype != np.float32 or array.shape != shape
                    or not array.flags.c_contiguous or not array.flags.aligned):
                raise ValueError("Invalid native batch layout")
        self.calls.append(tuple(array.copy() for array in arrays))
        if self.metrics is not None:
            return np.tile(self.metrics, (count, 1))
        previous, current, anchors, original, returned = (a.astype(np.float64) for a in arrays)
        return np.array([[np.linalg.norm(q - p), FourTipTracker._correlation(old, new),
                          FourTipTracker._correlation(anchor, new), new.std(), old.std()]
                         for old, new, anchor, p, q in
                         zip(previous, current, anchors, original, returned)])


class FastTrackerTests(unittest.TestCase):
    def setUp(self):
        threads = cv2.getNumThreads()
        cv2.setNumThreads(1)
        self.addCleanup(cv2.setNumThreads, threads)
        timer = mock.patch("fingertip_tracking.time.perf_counter", return_value=0.)
        timer.start()
        self.addCleanup(timer.stop)
        self.clock = Clock()
        self.centers = [(100, 120), (240, 120), (380, 300), (520, 300)]
        rng = np.random.default_rng(14)
        self.patches = [rng.integers(20, 235, (64, 64), np.uint8) for _ in POINT_NAMES]
        self.frame = self.image()

    def image(self, offsets=None, hidden=()):
        frame = np.full((480, 640), 100, np.uint8)
        for i, ((x, y), (dx, dy)) in enumerate(zip(self.centers, offsets or [(0, 0)] * 4)):
            if i not in hidden:
                frame[y + dy - 32:y + dy + 32, x + dx - 32:x + dx + 32] = self.patches[i]
        return frame

    def source(self, sequence=0):
        return {"stream_id": "camera", "generation": 1, "sequence": sequence,
                "capture_monotonic": self.clock(), "roi": (40, 60, 550, 330),
                "points": {name: {"x": x, "y": y, "raw_score": -10. + i}
                           for i, (name, (x, y)) in enumerate(zip(POINT_NAMES, self.centers))}}

    def update(self, tracker, sequence, frame=None, source=None):
        return tracker.update(self.frame if frame is None else frame, source,
                              {"stream": "camera", "generation": 1, "sequence": sequence})

    def assert_state_equal(self, original, optimized):
        self.assertEqual(original._losses, optimized._losses)
        self.assertEqual(original._quality, optimized._quality)
        self.assertEqual(set(original._tips), set(optimized._tips))
        for name, tip in original._tips.items():
            self.assertEqual(tip["raw_score"], optimized._tips[name]["raw_score"])
            for key in ("point", "anchor"):
                np.testing.assert_array_equal(tip[key], optimized._tips[name][key])

    def test_real_lk_independent_motion_and_partial_occlusion_match_exactly(self):
        for batched in (False, True):
            for hidden in ((), (1,), (0, 2), (0, 1, 2, 3)):
                with self.subTest(batched=batched, hidden=hidden):
                    pair = [cls(clock=self.clock, patch_quality=BatchQuality() if batched else None)
                            for cls in (OriginalTracker, FourTipTracker)]
                    source = self.source()
                    self.assertEqual(*(self.update(t, 0, source=source) for t in pair))
                    self.clock.value += .03
                    frame = self.image([(4, 0), (0, 0), (0, 12), (-6, 0)], hidden)
                    outputs = [self.update(t, 1, frame) for t in pair]
                    self.assertEqual(*outputs)
                    self.assertEqual(set(outputs[1]["points"]),
                                     set(POINT_NAMES) - {POINT_NAMES[i] for i in hidden})
                    self.assert_state_equal(*pair)
                    self.clock.value += .03
                    outputs = [self.update(t, 2, source=source) for t in pair]
                    self.assertEqual(*outputs)
                    for i in hidden:
                        self.assertNotIn(POINT_NAMES[i], outputs[1]["points"])

    def test_native_cutoff_guard_matches_on_both_sides_and_endpoints(self):
        for column, boundary in enumerate((1.5, .88, .88, 5., 5.)):
            values = [boundary, np.nextafter(boundary, -np.inf), np.nextafter(boundary, np.inf)]
            for sign in (-1, 1):
                edge = boundary + sign * 1e-5
                values.extend((np.nextafter(edge, -np.inf), edge, np.nextafter(edge, np.inf)))
            for value in values:
                with self.subTest(column=column, value=value):
                    metrics = [0., 1., 1., 50., 50.]
                    metrics[column] = value
                    pair = [cls(IdentityFlow(), self.clock, patch_quality=BatchQuality(metrics))
                            for cls in (OriginalTracker, FourTipTracker)]
                    for tracker in pair:
                        self.update(tracker, 0, source=self.source())
                    outputs = []
                    expected = 8 if abs(float(value) - boundary) <= 1e-5 else 0
                    for tracker in pair:
                        with mock.patch.object(tracker, "_correlation", wraps=tracker._correlation) as corr:
                            outputs.append(self.update(tracker, 1))
                            self.assertEqual(corr.call_count, expected)
                        self.assertEqual(len(tracker._patch_quality.calls), 1)
                    self.assertEqual(*outputs)
                    self.assert_state_equal(*pair)

    def test_numpy_fb_cutoff_uses_same_float32_rounding(self):
        small = cv2.resize(self.frame, (320, 200), interpolation=cv2.INTER_AREA)
        for offset in (np.float32(1.49999), np.float32(1.5), np.float32(1.50001)):
            class OffsetFlow(IdentityFlow):
                calls = 0

                def __call__(self, previous, current, points):
                    result, status, error = super().__call__(previous, current, points)
                    self.calls += 1
                    if self.calls == 2:
                        result[0, 0, 0] += offset
                    return result, status, error

            pair = [cls(OffsetFlow(), self.clock) for cls in (OriginalTracker, FourTipTracker)]
            for tracker in pair:
                self.update(tracker, 0, source=self.source())
                tracker._step(small, small)
            self.assert_state_equal(*pair)
            self.assertEqual("index" in pair[1]._tips, offset < 1.5)
            self.assertEqual(set(pair[1]._tips) - {"index"}, set(POINT_NAMES) - {"index"})

    def test_partial_batch_is_detached_from_reused_backend_storage(self):
        class ReusedFlow(IdentityFlow):
            def __init__(self):
                self.calls = 0
                self.storage = np.empty((4, 1, 2), np.float32)

            def __call__(self, previous, current, points):
                self.calls += 1
                if self.calls == 1:
                    self.original = points.copy()
                    self.storage[:] = points
                    return self.storage, np.array([1, 0, 1, 1], np.uint8), None
                if np.shares_memory(points, self.storage):
                    raise AssertionError("Reverse points alias backend storage")
                points.fill(-999.)
                self.storage[:3] = self.original[[0, 2, 3]]
                return self.storage[:3], np.array([1, 0, 1], np.uint8), None

        small = cv2.resize(self.frame, (320, 200), interpolation=cv2.INTER_AREA)
        pair = [cls(ReusedFlow(), self.clock, patch_quality=BatchQuality())
                for cls in (OriginalTracker, FourTipTracker)]
        for tracker in pair:
            self.update(tracker, 0, source=self.source())
            tracker._step(small, small)
            tracker._flow_backend.storage.fill(12345.)
            self.assertEqual(set(tracker._tips), {"index", "little"})
            self.assertEqual(tracker._losses, {"middle": "forward_lk", "ring": "forward_backward"})
        self.assert_state_equal(*pair)
        for first, second in zip(pair[0]._patch_quality.calls[0], pair[1]._patch_quality.calls[0]):
            np.testing.assert_array_equal(first, second)

    def test_fast_catchup_partial_occlusion_preserves_source_and_current_sequence(self):
        pair = [cls(clock=self.clock, fast_catchup=True, patch_quality=BatchQuality())
                for cls in (OriginalTracker, FourTipTracker)]
        source = self.source()
        for sequence in (0, 3, 6):
            self.assertEqual(*(self.update(t, sequence) for t in pair))
            self.clock.value += .1
        frame = self.image([(0, 0), (8, 12), (0, 0), (0, 0)], hidden=(0,))
        outputs = [self.update(t, 9, frame, source) for t in pair]
        self.assertEqual(*outputs)
        self.assertEqual(set(outputs[1]["points"]), set(POINT_NAMES) - {"index"})
        self.assertEqual(outputs[1]["source_sequence"], 0)
        self.assertEqual(outputs[1]["sequence"], 9)
        self.assertTrue(outputs[1]["diagnostics"]["direct_accepted"])
        self.assertEqual(outputs[1]["diagnostics"]["catchup_steps"], 1)
        self.assert_state_equal(*pair)

    def test_patch_bounds_fractional_positions_and_invalid_arity_match(self):
        small = cv2.resize(self.frame, (320, 200), interpolation=cv2.INTER_AREA)
        for point in ((5., 5.), (314.999, 194.999), (315., 195.), (4.999, 5.),
                      (49.75, 49.708332), (np.nan, 20.), (20., np.inf)):
            original, optimized = (cls._patch(small, point) for cls in (OriginalTracker, FourTipTracker))
            if original is None:
                self.assertIsNone(optimized)
            else:
                np.testing.assert_array_equal(original, optimized)
        for cls in (OriginalTracker, FourTipTracker):
            for point in ((1.,), (1., 2., 3.)):
                with self.assertRaises(ValueError):
                    cls._patch(small, point)


if __name__ == "__main__":
    unittest.main()
