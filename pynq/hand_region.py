"""Lightweight, non-neural hand-near-fretboard region tracking.

This module deliberately produces only a padded hand-near-neck overlay. It neither
estimates hand landmarks nor treats a region as evidence of a pressed string.
Its detect-then-local-track split takes only high-level inspiration from the
Vitis meanShift examples and MediaPipe's detection/tracking split; it uses
neither implementation.
"""
import time

import cv2
import numpy as np


class HandRegionTracker:
    """Track skin-coloured motion local to a locked fretboard ROI on the PS."""

    def __init__(self, flow_backend=None, target_side="any"):
        if target_side not in ("any", "right"):
            raise ValueError("target_side must be any or right")
        self.target_side = target_side
        self.flow_backend = flow_backend
        self._identity = None
        self._previous_gray = None
        self._previous_center = None
        self._previous_points = None
        self._last_points = None
        self._feature_frames = 0
        self._last_seen_at = None
        self._color_misses = 0
        self._template = None
        self._template_offset = None
        self._hand_extent = None

    def update(self, rgba, pl_gray, options, neck_analysis):
        """Return a JSON-safe near-neck region result for one camera frame.

        ``pl_gray`` is accepted as a cheap PL capture product.  Sparse LK is
        intentionally limited to the prior local observation rather than a
        full-frame optical-flow pass.
        """
        started = time.monotonic()
        options = options or {}
        stream_id = options.get("stream_id", options.get("stream", ""))
        generation = options.get("generation", 0)
        sequence = options.get("sequence", 0)
        identity = (stream_id, generation)
        if identity != self._identity:
            self._identity = identity
            self._previous_gray = self._previous_center = self._previous_points = None
            self._last_seen_at = self._template = self._template_offset = None
            self._hand_extent = None
            self._color_misses = self._feature_frames = 0

        base = self._result("waiting_for_neck", stream_id, generation, sequence, started)
        quad = self._locked_quad(neck_analysis, rgba.shape[:2])
        if quad is None:
            self._previous_gray = self._previous_center = self._previous_points = None
            self._last_seen_at = self._template = self._template_offset = None
            self._hand_extent = None
            return base

        small_rgba, scale = self._resize(rgba)
        small_gray = self._gray(small_rgba, pl_gray, scale)
        small_quad = quad * scale
        # Current colour evidence owns the fast path, including large jumps.
        # Running forward/back LK first delayed every successful observation.
        predicted, motion = self._previous_center, 0.
        center, geometry, extent = self._skin_candidate(small_rgba, small_quad, predicted,
                                                       self.target_side)
        if center is not None and predicted is not None:
            motion = min(1., float(np.linalg.norm(center-predicted))/8.)
        source = "ps_skin_motion" if motion > .05 else "ps_skin_geometry"
        if center is not None:
            self._last_seen_at = started
            self._color_misses = 0
            self._remember_template(small_gray, center)
            self._hand_extent = extent-center
        else:
            self._color_misses += 1
            if (self._last_seen_at is not None and started-self._last_seen_at <= .65
                    and self._color_misses <= 3):
                center, geometry = self._match_template(small_gray, small_quad, predicted)
                source = "ps_local_template"
        if center is not None and self.target_side == 'right' and center[0] < (small_gray.shape[1]-1)*.5:
            center = None
        self._previous_gray = small_gray

        if center is None:
            self._previous_center = self._previous_points = None
            self._template = self._template_offset = None
            self._hand_extent = None
            return self._result("searching", stream_id, generation, sequence, started)

        self._previous_center = center
        extent = ((self._hand_extent+center)/scale if self._hand_extent is not None else None)
        band = self._hand_box(quad, center / scale, rgba.shape[:2], extent)
        if band is None:
            return self._result("searching", stream_id, generation, sequence, started)
        confidence = min(.95, .30 + .55 * geometry + .15 * motion)
        result = self._result("tracking", stream_id, generation, sequence, started)
        result["regions"] = [dict(corners=[dict(x=round(float(x), 4), y=round(float(y), 4))
                                           for x, y in band],
                                  confidence=round(float(confidence), 4), source=source)]
        result["processing_ms"] = round((time.monotonic() - started) * 1000, 3)
        return result

    @staticmethod
    def _result(status, stream_id, generation, sequence, started):
        return dict(status=status, regions=[], stream_id=stream_id, generation=generation,
                    sequence=sequence, inference_location="ps_linux", mode="neck_local_lightweight",
                    processing_ms=round((time.monotonic() - started) * 1000, 3))

    @staticmethod
    def _resize(rgba):
        height, width = rgba.shape[:2]
        # Region localization needs a few pixels of accuracy, not pose detail.
        ratio = min(1.0, 160.0 / width, 120.0 / height)
        if ratio == 1.0:
            return rgba, np.array([1.0, 1.0], np.float32)
        small = cv2.resize(rgba, (round(width * ratio), round(height * ratio)))
        return small, np.array([(small.shape[1]-1)/(width-1),
                                (small.shape[0]-1)/(height-1)],np.float32)

    @staticmethod
    def _gray(rgba, pl_gray, scale):
        if pl_gray is not None and np.asarray(pl_gray).ndim == 2:
            gray = np.asarray(pl_gray)
            if gray.shape != rgba.shape[:2]:
                gray = cv2.resize(gray, (rgba.shape[1], rgba.shape[0]))
            return gray.astype(np.uint8, copy=False)
        return cv2.cvtColor(rgba, cv2.COLOR_RGBA2GRAY)

    @staticmethod
    def _locked_quad(neck_analysis, shape):
        if not isinstance(neck_analysis, dict) or neck_analysis.get("roi_status") not in ("locked", "tracking"):
            return None
        corners = neck_analysis.get("roi_corners")
        if not isinstance(corners, (list, tuple)) or len(corners) != 4:
            return None
        try:
            quad = np.float32([[point["x"], point["y"]] for point in corners])
        except (KeyError, TypeError, ValueError):
            return None
        if not np.isfinite(quad).all():
            return None
        height, width = shape
        if np.max(np.abs(quad)) <= 1.5:
            quad *= np.array([width - 1, height - 1], np.float32)
        if (cv2.contourArea(quad) < 24 or not cv2.isContourConvex(quad) or
                (quad < 0).any() or (quad > (width-1,height-1)).any()):
            return None
        return quad

    @staticmethod
    def _skin_candidate(rgba, quad, predicted=None, target_side="any"):
        height, width = rgba.shape[:2]
        from tracking_native import library
        native_cv = library()
        if native_cv is not None and width <= 160 and height <= 120:
            image = np.ascontiguousarray(rgba)
            polygon = np.ascontiguousarray(quad,dtype=np.float32)
            output = np.empty(7,np.float32)
            px,py = (float('nan'),float('nan')) if predicted is None else predicted
            found = native_cv.tracking_cv_skin(image.ctypes.data,width,height,polygon.ctypes.data,
                float(px),float(py),int(target_side == 'right'),output.ctypes.data)
            if found < 0:
                raise ValueError('Native OpenCV hand region failed')
            return ((output[:2].copy(),float(output[2]),output[3:].reshape(2,2).copy())
                if found else (None,0.,None))
        rgb = np.ascontiguousarray(rgba[:, :, :3])
        ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
        cr = ycrcb[:, :, 1]
        # The real hand is warmer than the nearby shirt and wooden neck.  The
        # deliberately narrower Cr floor avoids allowing a shirt-sized region
        # to dominate a compact fretting hand at the neck.
        skin = cv2.inRange(ycrcb, (36, 150, 75), (255, 180, 135))
        # The expanded neck is a locality gate, not a claim that the whole
        # hand lies on the instrument.
        axis = (quad[1]+quad[2]-quad[0]-quad[3])*.5
        normal = np.array([-axis[1],axis[0]],np.float32)/max(float(np.linalg.norm(axis)),1.)
        width_hint = max(8, int(np.linalg.norm(quad[3] - quad[0]) * .8))
        # Filling an offset hull is linear in pixels; an 80px ellipse dilation
        # was quadratic in kernel area and took >500ms on the Cortex-A9.
        expanded = cv2.convexHull(np.concatenate((quad+normal*width_hint*1.25,
                                                 quad-normal*width_hint*1.25)).astype(np.float32))
        mask = np.zeros((height, width), np.uint8)
        cv2.fillConvexPoly(mask, np.round(expanded).astype(np.int32), 255)
        if target_side == 'right':
            mask[:, :width//2] = 0
        skin = cv2.morphologyEx(skin & mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(skin)
        # Score the part actually inside the neck polygon. The expanded mask
        # only admits a forearm entering from outside; using it for the center
        # pulled the box toward the shirt in the recorded guitar view.
        neck_mask = np.zeros((height, width), np.uint8)
        cv2.fillConvexPoly(neck_mask, np.round(quad).astype(np.int32), 255)
        from neck_region import native_backend
        native = getattr(native_backend(), 'tracking_skin_component', None)
        if native is not None and width <= 160 and height <= 120:
            cr = np.ascontiguousarray(cr)
            quad = np.ascontiguousarray(quad, dtype=np.float32)
            output = np.empty(7, np.float32)
            px, py = (float('nan'), float('nan')) if predicted is None else predicted
            found = native(labels.ctypes.data, cr.ctypes.data, neck_mask.ctypes.data,
                stats.ctypes.data, quad.ctypes.data, count, width, height, float(px), float(py),
                int(target_side == 'right'), width_hint, output.ctypes.data)
            return ((output[:2].copy(), float(output[2]), output[3:].reshape(2,2).copy())
                    if found else (None, 0., None))
        best = None
        for index in range(1, count):
            x, yy, component_width, component_height, area = stats[index]
            if area < 18 or area > width * height * .20:
                continue
            aspect = max(component_width, component_height) / max(1, min(component_width, component_height))
            # A warm wooden neck can pass broad YCrCb bounds, but is typically
            # a large, long rectangle rather than a compact near-hand blob.
            fill = area / max(1, component_width * component_height)
            if aspect > 5 or fill > .92:
                continue
            # Reduce label work to this component's bounding box.  The exact
            # neck polygon, rather than the expanded search hull, owns the
            # reported center.
            local = labels[yy:yy + component_height, x:x + component_width] == index
            local_neck = neck_mask[yy:yy + component_height, x:x + component_width]
            contact = local & (local_neck != 0)
            contact_area = int(np.count_nonzero(contact))
            if contact_area < 8:
                continue
            contact_y, contact_x = np.nonzero(contact)
            center = np.array([x + np.mean(contact_x), yy + np.mean(contact_y)], np.float32)
            if target_side == 'right' and center[0] < (width-1)*.5:
                continue
            overlap = cv2.pointPolygonTest(quad, tuple(center), True)
            distance = max(0., -overlap) / max(1., width_hint)
            temporal = 1.
            if predicted is not None:
                jump = float(np.linalg.norm(center - predicted))
                # A prior ranks competing candidates; it must not prevent a
                # visible hand from crossing the neck between sparse frames.
                temporal = .5 + .5*max(0., 1.-jump/max(1., width_hint*1.5))
            warm = float(np.mean(cr[yy:yy + component_height, x:x + component_width][local]))
            score = min(1., contact_area / 180.) * max(0., 1. - distance) * temporal * (warm / 180.)
            if best is None or score > best[0]:
                best = (score, center, np.float32([[x, yy],
                    [x+component_width-1, yy+component_height-1]]))
        return (None, 0., None) if best is None else (best[1], best[0], best[2])

    def _local_motion(self, gray):
        self._last_points = None
        if self._previous_gray is None or self._previous_points is None:
            return None, 0.
        points = self._previous_points
        tracked, valid, error = cv2.calcOpticalFlowPyrLK(self._previous_gray, gray, points, None,
            winSize=(11, 11), maxLevel=1,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 8, .05))
        if tracked is None or valid is None:
            return None, 0.
        mask = valid.reshape(-1).astype(bool) & np.isfinite(tracked.reshape(-1, 2)).all(axis=1)
        if error is not None:
            mask &= np.isfinite(error.reshape(-1)) & (error.reshape(-1) < 30)
        old = points[mask].reshape(-1, 2)
        new = tracked[mask].reshape(-1, 2)
        if len(old) < 2:
            return None, 0.
        returned, valid_back, _ = cv2.calcOpticalFlowPyrLK(gray, self._previous_gray,
            new.reshape(-1, 1, 2), None, winSize=(11, 11), maxLevel=1,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 8, .05))
        if returned is None or valid_back is None:
            return None, 0.
        good = (valid_back.reshape(-1).astype(bool) &
                np.isfinite(returned.reshape(-1, 2)).all(axis=1) &
                (np.linalg.norm(returned.reshape(-1, 2)-old, axis=1) <= 1.5))
        old, new = old[good], new[good]
        if len(old) < 2:
            return None, 0.
        delta = np.median(new-old, axis=0)
        good = np.linalg.norm(new-old-delta, axis=1) <= 3.
        if np.count_nonzero(good) < 2:
            return None, 0.
        old, new = old[good], new[good]
        self._last_points = new.reshape(-1,1,2).copy()
        displacement = np.median(np.linalg.norm(new - old, axis=1))
        center = self._previous_center + np.median(new-old, axis=0)
        return center.astype(np.float32), min(1., float(displacement) / 8.)

    def _remember_template(self, gray, center):
        x, y = np.round(center).astype(int)
        left, top = max(0, x-12), max(0, y-10)
        right, bottom = min(gray.shape[1], x+13), min(gray.shape[0], y+11)
        patch = gray[top:bottom, left:right]
        if min(patch.shape) >= 12 and float(patch.std()) >= 8:
            self._template = patch.copy()
            self._template_offset = center - (left, top)
        else:
            self._template = self._template_offset = None

    def _match_template(self, gray, quad, predicted):
        if self._template is None or self._previous_center is None:
            return None, 0.
        anchor = self._previous_center if predicted is None else predicted
        offset = self._template_offset
        th, tw = self._template.shape
        left = max(0, int(anchor[0]-offset[0])-24)
        top = max(0, int(anchor[1]-offset[1])-16)
        right = min(gray.shape[1], int(anchor[0]-offset[0])+tw+25)
        bottom = min(gray.shape[0], int(anchor[1]-offset[1])+th+17)
        window = gray[top:bottom, left:right]
        if window.shape[0] < th or window.shape[1] < tw:
            return None, 0.
        scores = cv2.matchTemplate(window, self._template, cv2.TM_CCOEFF_NORMED)
        # Only a currently visible, textured match inside the neck can bridge
        # a short colour miss. A timer alone must never keep a red box alive.
        yy, xx = np.indices(scores.shape)
        centers = np.stack((xx+left+offset[0], yy+top+offset[1]), axis=-1)
        mask = np.zeros(gray.shape, np.uint8)
        cv2.fillConvexPoly(mask, np.round(quad).astype(np.int32), 1)
        px = np.clip(np.round(centers[..., 0]).astype(int), 0, gray.shape[1]-1)
        py = np.clip(np.round(centers[..., 1]).astype(int), 0, gray.shape[0]-1)
        scores[(mask[py, px] == 0) | ~np.isfinite(scores)] = -1
        if self.target_side == 'right':
            scores[centers[...,0] < (gray.shape[1]-1)*.5] = -1
        _, score, _, position = cv2.minMaxLoc(scores)
        if score < .84:
            return None, 0.
        x, y = position
        current = window[y:y+th, x:x+tw].astype(np.float32)
        original = self._template.astype(np.float32)
        residual = np.mean(np.abs((current-current.mean())-(original-original.mean())))
        if float(current.std()) < 8 or residual > 18:
            return None, 0.
        return np.array([left+x, top+y], np.float32)+offset, float(score)

    @staticmethod
    def _features(gray, center):
        x, y = np.round(center).astype(int)
        half = 14
        left, top = max(0, x - half), max(0, y - half)
        right, bottom = min(gray.shape[1], x + half), min(gray.shape[0], y + half)
        patch = gray[top:bottom, left:right]
        points = cv2.goodFeaturesToTrack(patch, maxCorners=6, qualityLevel=.02, minDistance=3)
        if points is not None:
            points[:, 0, 0] += left
            points[:, 0, 1] += top
        return points

    @staticmethod
    def _neck_band(quad, center, shape):
        # Quad ordering follows roi_fretboard: 0->1 and 3->2 run along neck.
        a, b = (quad[0] + quad[3]) / 2, (quad[1] + quad[2]) / 2
        axis = b - a
        length = float(np.linalg.norm(axis))
        if length < 8:
            return None
        direction = axis / length
        position = float(np.clip(np.dot(center - a, direction) / length, 0., 1.))
        half = min(.14, max(.045, 20. / length))
        low, high = max(0., position - half), min(1., position + half)
        left0, left1 = quad[0] * (1 - low) + quad[1] * low, quad[0] * (1 - high) + quad[1] * high
        right0, right1 = quad[3] * (1 - low) + quad[2] * low, quad[3] * (1 - high) + quad[2] * high
        height, width = shape
        band = np.array([left0, left1, right1, right0], np.float32)
        # The public region schema is normalized regardless of the source ROI
        # convention.  Clip only final floating-point roundoff at the frame.
        return np.clip(band / np.array([width - 1, height - 1], np.float32), 0., 1.)

    @staticmethod
    def _hand_box(quad, center, shape, extent):
        """Include the visible hand outside the fingerboard, with a small margin."""
        height, width = shape
        band = HandRegionTracker._neck_band(quad, center, shape)
        if band is None:
            return None
        pixels = band*np.float32([width-1,height-1])
        if extent is not None:
            # Cap a connected wrist/forearm without clipping an ordinary hand.
            board_width = max(np.linalg.norm(quad[3]-quad[0]),
                              np.linalg.norm(quad[2]-quad[1]))
            radius = max(28.,board_width*1.6)
            pixels = np.concatenate((pixels,np.clip(extent,center-radius,center+radius)))
        low, high = pixels.min(axis=0), pixels.max(axis=0)
        padding = np.maximum((high-low)*.15, np.float32([width,height])*.018)
        low = np.maximum(0,low-padding)
        high = np.minimum([width-1,height-1],high+padding)
        return np.float32([low,[high[0],low[1]],high,[low[0],high[1]]])/np.float32([width-1,height-1])
