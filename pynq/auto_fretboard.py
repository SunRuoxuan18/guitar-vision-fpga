"""Uncalibrated, conservative detection of a visible six-string fret lattice.

Corners bound the *observed lattice*, not the wooden neck or all 22 frets.
Every update detects afresh; no geometry survives a frame without support.
Indices describe image order only. Confidence is a fit score, not probability.
"""

from copy import deepcopy
import math


class AutoFretboardTracker:
    """Detect six supported strings and at least five compatible fret wires.

    OpenCV and NumPy load only on the first non-None frame. A small/occluded
    neck, poor contrast, or ambiguous lattice returns searching without geometry.
    Corners are string0/fret0, string0/fretLast, string5/fretLast,
    string5/fret0, including vertical and inverted views. Instances are
    caller-owned; serialize access when using multiple threads.
    """

    def __init__(self):
        self._cv2 = self._np = None
        self.invalidate("not_detected")

    def snapshot(self):
        return deepcopy(self._snapshot)

    def invalidate(self, reason):
        self._snapshot = {"status": "searching", "reason": str(reason), "mode": "auto",
                          "corners": None, "strings": [], "frets": [],
                          "orientation": "image_order", "absolute_frets_known": False,
                          "confidence": 0.0, "string_count": 0, "fret_count": 0}
        return self.snapshot()

    def _dependencies(self):
        if self._cv2 is None:
            import cv2
            import numpy as np
            self._cv2, self._np = cv2, np

    def _cluster(self, segments, axis, normal, reference):
        """Merge a wire's two Canny edges and fragmented collinear segments."""
        np = self._np
        merge_distance = 6.0
        lines = []
        for segment in segments:
            uv = np.column_stack((segment @ axis, segment @ normal))
            if uv[1, 0] < uv[0, 0]:
                uv = uv[::-1]
            span = uv[1, 0] - uv[0, 0]
            if span < 1:
                continue
            slope = (uv[1, 1] - uv[0, 1]) / span
            offset = float(uv[0, 1] + slope * (reference - uv[0, 0]))
            lines.append({"slope": float(slope), "offset": offset,
                          "lo": float(uv[0, 0]), "hi": float(uv[1, 0]),
                          "weight": float(span)})
        groups = []
        for line in sorted(lines, key=lambda item: -item["weight"]):
            for group in groups:
                mid = (min(line["hi"], group["hi"]) + max(line["lo"], group["lo"])) / 2
                difference = abs(line["offset"] - group["offset"] +
                                 (line["slope"] - group["slope"]) * (mid - reference))
                if (difference <= merge_distance and abs(line["offset"] - group["offset"]) <= merge_distance
                        and abs(line["slope"] - group["slope"]) <= .045):
                    total = line["weight"] + group["weight"]
                    for key in ("slope", "offset"):
                        group[key] = (line[key] * line["weight"] +
                                      group[key] * group["weight"]) / total
                    group["weight"] = total
                    group["lo"] = min(group["lo"], line["lo"])
                    group["hi"] = max(group["hi"], line["hi"])
                    break
            else:
                groups.append(dict(line))
        groups.sort(key=lambda item: item["offset"])
        merged = []
        for group in groups:
            if (merged and group["offset"] - merged[-1]["offset"] <= merge_distance
                    and abs(group["slope"] - merged[-1]["slope"]) < .045):
                previous = merged[-1]
                total = previous["weight"] + group["weight"]
                for key in ("offset", "slope"):
                    previous[key] = (previous[key] * previous["weight"] +
                                     group[key] * group["weight"]) / total
                previous["weight"] = total
                previous["lo"] = min(previous["lo"], group["lo"])
                previous["hi"] = max(previous["hi"], group["hi"])
            else:
                merged.append(group)
        return merged

    @staticmethod
    def _intersection(string, fret, center_u, center_v):
        # v = a(u-center_u)+b; u = c(v-center_v)+d.
        a, b = string["slope"], string["offset"]
        c, d = fret["slope"], fret["offset"]
        u = (c * (b - a * center_u - center_v) + d) / (1 - a * c)
        return u, a * (u - center_u) + b

    def _spacing(self, positions):
        """Fit consecutive equal-tempered positions after planar rectification.

        A regular grid is affine-uniform after this same rectification and must
        fit substantially worse. Missing/irregular wires are omitted, never filled.
        """
        np = self._np
        if len(positions) < 5:
            return None
        gaps = np.diff(positions)
        if np.min(gaps) < .008:
            return None
        best = None
        for direction in (-1, 1):
            basis = np.exp(direction * math.log(2) / 12 * np.arange(len(positions)))
            design = np.column_stack((np.ones(len(basis)), basis))
            fitted = design @ np.linalg.lstsq(design, positions, rcond=None)[0]
            error = float(np.sqrt(np.mean((fitted - positions) ** 2)))
            if best is None or error < best:
                best = error
        uniform = np.linspace(positions[0], positions[-1], len(positions))
        uniform_error = float(np.sqrt(np.mean((positions - uniform) ** 2)))
        # Five wires span four gaps: require a measurable progression, not just
        # an unconstrained projective curve that also explains a regular grid.
        if best > .012 or uniform_error < .012 or best >= uniform_error * .55:
            return None
        return best

    def _wire_support(self, gray, start, end):
        """Require a bright/dark ridge, rather than a one-sided neck border."""
        cv2, np = self._cv2, self._np
        vector = end - start
        length = float(np.linalg.norm(vector))
        if length < 10:
            return 0.0
        normal = np.array([-vector[1], vector[0]]) / length
        points = np.linspace(start, end, max(16, int(length / 2)))
        xy = points[:, None, :] + np.arange(-6, 7)[None, :, None] * normal
        profile = cv2.remap(gray, xy[:, :, 0].astype(np.float32),
                            xy[:, :, 1].astype(np.float32), cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE).astype(float)
        left, right = profile[:, :2].mean(axis=1), profile[:, -2:].mean(axis=1)
        center = profile[:, 4:9]
        bright = np.minimum(center - left[:, None], center - right[:, None]).max(axis=1)
        dark = np.minimum(left[:, None] - center, right[:, None] - center).max(axis=1)
        samples = np.maximum(bright, dark) >= 12
        return float(min(samples.mean(), min(part.mean() for part in np.array_split(samples, 4)) + .25))

    def _refine(self, gray, ends):
        """Refit wire centers using local intensity ridges instead of edge sides."""
        cv2, np = self._cv2, self._np
        start, end = ends
        vector = end - start
        normal = np.array([-vector[1], vector[0]]) / np.linalg.norm(vector)
        points = np.linspace(start, end, 100)
        offsets = np.linspace(-5, 5, 21)
        xy = points[:, None, :] + offsets[None, :, None] * normal
        profiles = cv2.remap(gray, xy[:, :, 0].astype(np.float32),
                             xy[:, :, 1].astype(np.float32), cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE).astype(float)
        baseline = (profiles[:, :2].mean(axis=1) + profiles[:, -2:].mean(axis=1)) / 2
        center = profiles[:, 6:15]
        sign = 1 if np.median(center.max(axis=1) - baseline) >= np.median(baseline - center.min(axis=1)) else -1
        weight = np.maximum(sign * (profiles - baseline[:, None]) - 10, 0)
        total = weight.sum(axis=1)
        good = total > 30
        if good.sum() < 25:
            return ends
        shifts = (weight[good] @ offsets) / total[good]
        ridge = points[good] + shifts[:, None] * normal
        vx, vy, x, y = cv2.fitLine(ridge.astype(np.float32), cv2.DIST_HUBER, 0, .01, .01).ravel()
        direction = np.array([vx, vy], dtype=float)
        origin = np.array([x, y], dtype=float)
        return origin + ((ends - origin) @ direction)[:, None] * direction

    def _candidate(self, gray, segments, axis):
        cv2, np = self._cv2, self._np
        normal = np.array([-axis[1], axis[0]])
        if normal[int(np.argmax(np.abs(normal)))] < 0:
            normal = -normal
        vectors = segments[:, 1] - segments[:, 0]
        along, across = vectors @ axis, vectors @ normal
        longitudinal = (np.abs(across) < np.abs(along) * .16) & (np.abs(along) >= 90)
        if longitudinal.sum() < 6:
            return None
        center_u = float(np.median(segments[longitudinal].reshape(-1, 2) @ axis))
        strings = self._cluster(segments[longitudinal], axis, normal, center_u)
        strings = sorted(strings, key=lambda line: -(line["hi"] - line["lo"]))[:24]
        supported_strings = []
        for line in strings:
            u = np.array([line["lo"], line["hi"]])
            v = line["slope"] * (u - center_u) + line["offset"]
            ends = u[:, None] * axis + v[:, None] * normal
            if self._wire_support(gray, *ends) >= .60:
                refined = self._refine(gray, ends)
                uv = np.column_stack((refined @ axis, refined @ normal))
                line["slope"] = float((uv[1, 1] - uv[0, 1]) / (uv[1, 0] - uv[0, 0]))
                line["offset"] = float(uv[0, 1] + line["slope"] * (center_u - uv[0, 0]))
                supported_strings.append(line)
        strings = supported_strings
        strings.sort(key=lambda line: line["offset"])
        transverse = (np.abs(along) < np.abs(across) * .35) & (np.abs(across) >= 20)
        if transverse.sum() < 5:
            return None
        best = None
        for start in range(len(strings) - 5):
            lattice = strings[start:start + 6]
            offsets = np.array([line["offset"] for line in lattice])
            pitch = (offsets[-1] - offsets[0]) / 5
            if not 6 <= pitch <= 45:
                continue
            lattice_error = float(np.max(np.abs(np.diff(offsets) / pitch - 1)))
            if lattice_error > .16:
                continue
            if np.median([line["hi"] - line["lo"] for line in lattice]) < pitch * 12.5:
                continue
            lo = min(line["lo"] for line in lattice)
            hi = max(line["hi"] for line in lattice)
            if hi - lo < max(100, pitch * 10):
                continue
            center_v = float((offsets[0] + offsets[-1]) / 2)
            frets = self._cluster(segments[transverse], normal, axis, center_v)
            # Hough fragments need not reach the outer strings. Verify the
            # full intersection-to-intersection ridge below, in image space.
            frets = [line for line in frets if lo - 5 <= line["offset"] <= hi + 5]
            supported_frets = []
            for line in frets:
                uv = np.array([self._intersection(string, line, center_u, center_v)
                               for string in (lattice[0], lattice[-1])])
                if line["hi"] < uv[:, 1].min() or line["lo"] > uv[:, 1].max():
                    continue
                ends = uv[:, :1] * axis + uv[:, 1:] * normal
                ends = self._refine(gray, ends)
                v, u = ends @ normal, ends @ axis
                line = dict(line)
                line["slope"] = float((u[1] - u[0]) / (v[1] - v[0]))
                line["offset"] = float(u[0] + line["slope"] * (center_v - v[0]))
                if self._wire_support(gray, *ends) >= .60:
                    supported_frets.append(line)
            frets = supported_frets
            if len(frets) < 5 or len(frets) > 25:
                continue
            first, last = frets[0], frets[-1]
            uv_quad = np.float32([self._intersection(lattice[0], first, center_u, center_v),
                                  self._intersection(lattice[0], last, center_u, center_v),
                                  self._intersection(lattice[-1], last, center_u, center_v),
                                  self._intersection(lattice[-1], first, center_u, center_v)])
            quad = uv_quad[:, :1] * axis + uv_quad[:, 1:] * normal
            if (not cv2.isContourConvex(quad.astype(np.float32))
                    or (quad < 0).any() or (quad > (gray.shape[1] - 1, gray.shape[0] - 1)).any()):
                continue
            if np.linalg.norm(quad[1] - quad[0]) < max(80, pitch * 4):
                continue
            transform = cv2.getPerspectiveTransform(quad.astype(np.float32),
                                                     np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]))
            middle = {"slope": (lattice[0]["slope"] + lattice[-1]["slope"]) / 2,
                      "offset": center_v}
            fret_uv = np.float32([self._intersection(middle, line, center_u, center_v) for line in frets])
            fret_xy = fret_uv[:, :1] * axis + fret_uv[:, 1:] * normal
            positions = cv2.perspectiveTransform(fret_xy[:, None, :], transform)[:, 0, 0]
            spacing_error = self._spacing(positions)
            if spacing_error is None:
                continue
            string_ends = []
            for line in lattice:
                uv = np.array([self._intersection(line, fret, center_u, center_v) for fret in (first, last)])
                string_ends.append(uv[:, :1] * axis + uv[:, 1:] * normal)
            fret_ends = []
            for line in frets:
                uv = np.array([self._intersection(string, line, center_u, center_v) for string in (lattice[0], lattice[-1])])
                fret_ends.append(uv[:, :1] * axis + uv[:, 1:] * normal)
            supports = [self._wire_support(gray, *ends) for ends in string_ends + fret_ends]
            if min(supports) < .60:
                continue
            confidence = float(min(1.0, np.mean(supports) * (1 - lattice_error) * (1 - spacing_error * 15)))
            if best is None or confidence > best[0]:
                best = confidence, quad, string_ends, fret_ends
        return best

    def update(self, gray, edge=None):
        """Return only geometry independently supported in this uint8 frame."""
        self.invalidate("no_supported_lattice")
        if gray is None:
            return self.invalidate("invalid_frame")
        self._dependencies()
        cv2, np = self._cv2, self._np
        if (not isinstance(gray, np.ndarray) or gray.dtype != np.uint8
                or gray.ndim != 2 or min(gray.shape) < 24):
            return self.invalidate("invalid_frame")
        height, width = gray.shape
        scale = min(1.0, 640 / max(gray.shape))
        work = cv2.resize(gray, (max(2, round(width * scale)), max(2, round(height * scale))),
                          interpolation=cv2.INTER_AREA) if scale < 1 else gray
        try:
            if edge is None:
                edges = cv2.Canny(cv2.GaussianBlur(work, (3, 3), .6), 35, 90)
            else:
                if (not isinstance(edge, np.ndarray) or edge.dtype != np.uint8
                        or edge.shape != gray.shape):
                    return self.invalidate("invalid_edge_frame")
                # PL's threshold mask is thick; Hough needs localized edges.
                # Canny supplies localization, gated by real hardware support.
                support = cv2.resize(edge, (work.shape[1], work.shape[0]),
                                   interpolation=cv2.INTER_NEAREST) if scale < 1 else edge
                localized = cv2.Canny(cv2.GaussianBlur(work, (3, 3), .6), 35, 90)
                edges = cv2.bitwise_and(localized, cv2.dilate(support, np.ones((3, 3), np.uint8)))
            # Half-degree voting retains thin-wire/rotation accuracy without
            # the original 720-angle accumulator cost on ARM OpenCV 4.2.
            lines = cv2.HoughLinesP(edges, 1, math.pi / 360, 35,
                                    minLineLength=30, maxLineGap=9)
            if lines is None:
                return self.snapshot()
            segments = lines[:, 0].reshape(-1, 2, 2).astype(float)
            lengths = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
            segments = segments[np.argsort(-lengths)[:240]]
            vectors = segments[:, 1] - segments[:, 0]
            angles = np.arctan2(vectors[:, 1], vectors[:, 0]) % math.pi
            weights = np.linalg.norm(vectors, axis=1)
            histogram, _ = np.histogram(angles[weights >= 90], bins=60, range=(0, math.pi),
                                        weights=weights[weights >= 90])
            candidates = []
            chosen = []
            for index in np.argsort(-histogram):
                if histogram[index] == 0 or len(chosen) == 2:
                    break
                angle = (index + .5) * math.pi / 60
                if any(abs(math.sin(angle - previous)) < .2 for previous in chosen):
                    continue
                chosen.append(angle)
                if angle > math.pi / 2:
                    angle -= math.pi
                axis = np.array([math.cos(angle), math.sin(angle)])
                if axis[int(np.argmax(np.abs(axis)))] < 0:
                    axis = -axis
                candidate = self._candidate(work, segments, axis)
                if candidate is not None:
                    candidates.append(candidate)
            if not candidates:
                return self.snapshot()
            confidence, quad, strings, frets = max(candidates, key=lambda item: item[0])
            # Pixel centers use the working frame's exact resized dimensions.
            denominator = np.array([work.shape[1] - 1, work.shape[0] - 1])

            def point(xy):
                x, y = xy / denominator
                return {"x": float(x), "y": float(y)}

            def line(ends):
                return {"start": point(ends[0]), "end": point(ends[1]), "observed": True}

            self._snapshot.update(status="tracking", reason=None, confidence=confidence,
                                  corners=[point(xy) for xy in quad],
                                  strings=[line(ends) for ends in strings],
                                  frets=[dict(line(ends), index=index) for index, ends in enumerate(frets)],
                                  string_count=len(strings), fret_count=len(frets))
        except cv2.error:
            return self.invalidate("opencv_error")
        return self.snapshot()
