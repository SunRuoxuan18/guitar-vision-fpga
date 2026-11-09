"""Conservative, explicitly calibrated planar tracking; imports stay dependency lazy."""

from numbers import Real
import time


class FretboardTracker:
    """Follow four perimeter-ordered corners using sparse LK flow and RANSAC.

    Coordinates are normalized to the first/last pixel centers, so (1, 1) is
    (width - 1, height - 1). Frames must be uint8 grayscale arrays of a fixed
    size. Call initialize again after loss; update never reacquires or seeds
    new features. Snapshots contain corners only while status is tracking.

    Instances belong to one caller; serialize access if used across threads.
    """

    _MIN_POINTS = 16
    _MAX_POINTS = 240
    def __init__(self, flow_backend=None, fast_motion=False, endpoint_weighted=False):
        self._flow_backend = flow_backend
        self._fast_motion = bool(fast_motion)
        self._endpoint_weighted = bool(endpoint_weighted)
        self.timings = {}
        self._cv2 = None
        self._np = None
        self._patch_grid = None
        self._spread_native = None
        self._warp_native = None
        self._previous = None
        self._quad = None
        self._points = None
        self._status = "lost"
        self._reason = "not_initialized"
        self._point_count = 0
        self._inliers = 0
        self._inlier_ratio = 0.0
        self._sampling = "full_roi"
        self._endpoint_points = (0, 0)
        self._motion_mode = "full"
        self._partial_frames = 0
        self._partial_appearance_frames = 0
        self._appearance_mode = "full"

    def snapshot(self):
        """Return detached, JSON-compatible state, never stale usable corners."""
        corners = None
        if self._status == "tracking":
            height, width = self._previous.shape
            corners = [{"x": float(x / (width - 1)),
                        "y": float(y / (height - 1))} for x, y in self._quad]
        return {"status": self._status, "corners": corners,
                "points": self._point_count, "inliers": self._inliers,
                "inlier_ratio": self._inlier_ratio, "sampling": self._sampling,
                "endpoint_points": list(self._endpoint_points),
                "motion_mode": self._motion_mode,
                "appearance_mode": self._appearance_mode,
                "reason": self._reason}

    def invalidate(self, reason):
        """Discard calibration and frame history; retain last attempt's counts."""
        self._status = "lost"
        self._reason = str(reason)
        self._previous = self._quad = self._points = None
        self._motion_mode = "full"
        self._partial_frames = 0
        self._partial_appearance_frames = 0
        self._appearance_mode = "full"
        return self.snapshot()

    def _dependencies(self):
        if self._cv2 is None:
            import cv2
            import numpy as np
            self._cv2, self._np = cv2, np
            from neck_region import native_backend
            library = native_backend()
            self._spread_native = getattr(library, 'tracking_well_spread', None)
            from tracking_native import library as tracking_library
            self._warp_native = getattr(tracking_library(),'tracking_cv_patches',None)

    def _endpoint_mask(self, quad, shape, band=.34):
        """Mask features to the two longitudinal ends of the tracked board."""
        cv2, np = self._cv2, self._np
        unit = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
        transform = cv2.getPerspectiveTransform(unit, quad.astype(np.float32))
        regions = np.float32([
            [[0, 0], [band, 0], [band, 1], [0, 1]],
            [[1-band, 0], [1, 0], [1, 1], [1-band, 1]],
        ])
        mapped = cv2.perspectiveTransform(
            regions.reshape(-1, 1, 2), transform).reshape(2, 4, 2)
        mask = np.zeros(shape, np.uint8)
        for region in mapped:
            cv2.fillConvexPoly(mask, np.rint(region).astype(np.int32), 255)
        return mask

    def _feature_points(self, gray, mask, min_distance=5, erosion=7):
        """Return strong, spread-out feature candidates inside one mask."""
        cv2, np = self._cv2, self._np
        mask = cv2.erode(mask, np.ones((erosion, erosion), dtype=np.uint8))
        points = cv2.goodFeaturesToTrack(
            gray, maxCorners=self._MAX_POINTS, qualityLevel=.02,
            minDistance=min_distance, mask=mask, blockSize=7)
        if points is None:
            return None
        strength = cv2.cornerMinEigenVal(gray, blockSize=7)
        xy = np.rint(points[:, 0]).astype(int)
        return points[strength[xy[:, 1], xy[:, 0]] >= .0002]

    def _select_points(self, gray, quad, mask, allow_relaxed=True):
        """Prefer the two ends; fall back to all visible board texture."""
        cv2, np = self._cv2, self._np

        def balanced_endpoint(points, board, ends, counts):
            # Equal caps stop the busier end, often the fretting-hand side,
            # from starving the other end during RANSAC. Keep each cap
            # transversely stratified instead of choosing only the brightest
            # edge, which can collapse onto one row.
            strength = cv2.cornerMinEigenVal(gray, blockSize=7)
            strong_indices, stratified_indices = [], []
            for item in ends:
                indices = np.flatnonzero(item)
                if len(indices) > 16:
                    strong_indices.extend(self._strongest_indices(
                        points, indices, strength).tolist())
                    stratified_indices.extend(self._stratified_indices(
                        points, board, indices, strength).tolist())
                else:
                    strong_indices.extend(indices.tolist())
                    stratified_indices.extend(indices.tolist())
            strong_points = points[strong_indices]
            strong_board = board[strong_indices]
            stratified_points = points[stratified_indices]
            stratified_board = board[stratified_indices]
            selected_counts = (
                int((strong_board[:, 0] <= .34).sum()),
                int((strong_board[:, 0] >= .66).sum()))
            selected, selected_board = strong_points, strong_board
            spread = (self._endpoint_spread(selected, quad)
                      or self._well_spread(selected, quad))
            if not spread:
                selected = stratified_points
                selected_board = stratified_board
                selected_counts = (
                    int((selected_board[:, 0] <= .34).sum()),
                    int((selected_board[:, 0] >= .66).sum()))
                spread = (self._endpoint_spread(selected, quad)
                          or self._well_spread(selected, quad))
            if not spread:
                selected = self._distributed_subset(selected, quad)
                spread = (self._endpoint_spread(selected, quad)
                          or self._well_spread(selected, quad))
            if not spread:
                return None
            self._sampling = "endpoint_bands"
            self._endpoint_points = selected_counts
            return selected

        if self._endpoint_weighted:
            endpoints = self._endpoint_mask(quad, gray.shape)
            endpoints[mask == 0] = 0
            points = self._feature_points(
                gray, endpoints, min_distance=3, erosion=5)
            if points is not None and len(points) >= self._MIN_POINTS:
                to_board = cv2.getPerspectiveTransform(
                    quad.astype(np.float32), np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]))
                board = cv2.perspectiveTransform(points, to_board)[:, 0]
                ends = (board[:, 0] <= .34, board[:, 0] >= .66)
                counts = tuple(int(item.sum()) for item in ends)
                # Both ends must be genuinely present before endpoint mode
                # starts. After acquisition, a temporarily missing end can
                # still be bridged by the guarded one-end motion model.
                if min(counts) >= 12:
                    selected = balanced_endpoint(points, board, ends, counts)
                    if selected is not None:
                        return selected

        fallback = self._feature_points(gray, mask)
        if (fallback is not None and len(fallback) >= self._MIN_POINTS
                and self._well_spread(fallback, quad)):
            self._sampling = "full_roi_fallback"
            self._endpoint_points = (0, 0)
            return fallback

        # Sparse headstock crops sometimes need denser candidates. Use that
        # relaxation only after the strict endpoint and full-board paths both
        # fail, so a stable track is never displaced by weak specular points.
        if self._endpoint_weighted and allow_relaxed:
            relaxed = self._feature_points(
                gray, endpoints, min_distance=2, erosion=3)
            if relaxed is not None and len(relaxed) >= self._MIN_POINTS:
                to_board = cv2.getPerspectiveTransform(
                    quad.astype(np.float32),
                    np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]))
                relaxed_board = cv2.perspectiveTransform(
                    relaxed, to_board)[:, 0]
                relaxed_ends = (
                    relaxed_board[:, 0] <= .34,
                    relaxed_board[:, 0] >= .66)
                relaxed_counts = tuple(
                    int(item.sum()) for item in relaxed_ends)
                if min(relaxed_counts) >= 12:
                    selected = balanced_endpoint(
                        relaxed, relaxed_board, relaxed_ends, relaxed_counts)
                    if selected is not None:
                        return selected

        points = fallback
        self._sampling = "full_roi_fallback"
        self._endpoint_points = (0, 0)
        return points

    def _strongest_indices(self, points, indices, strength, limit=16):
        np = self._np
        indices = np.asarray(indices, dtype=int)
        xy = np.rint(points[indices, 0]).astype(int)
        strengths = strength[xy[:, 1], xy[:, 0]]
        order = np.argsort(-strengths, kind="stable")
        return indices[order[:limit]]

    def _stratified_indices(self, points, board, indices, strength, limit=16):
        """Keep strong features while covering every transverse cap band."""
        np = self._np
        indices = np.asarray(indices, dtype=int)
        coordinates = board[indices]
        xy = np.rint(points[indices, 0]).astype(int)
        strengths = strength[xy[:, 1], xy[:, 0]]
        order = np.argsort(-strengths, kind="stable")
        chosen = order[:min(8, limit)].tolist()
        for low, high in zip(np.linspace(0., .75, 4),
                             np.linspace(.25, 1., 4)):
            band = ((coordinates[:, 1] >= low)
                    & ((coordinates[:, 1] < high)
                       | ((high == 1.) & (coordinates[:, 1] <= high))))
            for candidate in order[band[order]]:
                if candidate not in chosen:
                    chosen.append(int(candidate))
                    break
        for candidate in order:
            if len(chosen) >= min(limit, len(indices)):
                break
            if candidate not in chosen:
                chosen.append(int(candidate))
        return indices[np.asarray(chosen[:limit], dtype=int)]

    def _valid_frame(self, gray):
        np = self._np
        return (isinstance(gray, np.ndarray) and gray.dtype == np.uint8
                and gray.ndim == 2 and min(gray.shape) >= 24)

    def _valid_quad(self, quad, shape):
        cv2, np = self._cv2, self._np
        height, width = shape
        if not np.isfinite(quad).all():
            return False
        if (quad < 0).any() or (quad > (width - 1, height - 1)).any():
            return False
        contour = quad.astype(np.float32)
        if not cv2.isContourConvex(contour):
            return False
        area = cv2.contourArea(contour)
        edges = np.linalg.norm(quad - np.roll(quad, 1, axis=0), axis=1)
        return (area >= max(80, width * height * .002)
                and edges.min() >= 8 and area / edges.max() >= 8)

    def _well_spread(self, points, quad):
        """Require support across the board, rather than a small moving patch."""
        cv2, np = self._cv2, self._np
        if len(points) < self._MIN_POINTS:
            return False
        unit_quad = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
        to_board = cv2.getPerspectiveTransform(quad.astype(np.float32), unit_quad)
        board = cv2.perspectiveTransform(points.reshape(-1, 1, 2), to_board)[:, 0]
        if self._spread_native is not None:
            board = np.ascontiguousarray(board, dtype=np.float32)
            return bool(self._spread_native(board.ctypes.data, len(board)))
        if not np.isfinite(board).all() or (board < -.03).any() or (board > 1.03).any():
            return False
        quantiles = np.quantile(board, [.1, .9], axis=0)
        span = quantiles[1] - quantiles[0]
        cells = np.clip((board * (3, 2)).astype(int), (0, 0), (2, 1))
        occupied = len(np.unique(cells[:, 0] + 3 * cells[:, 1]))
        hull_area = cv2.contourArea(cv2.convexHull(board))
        return bool(span[0] >= .55 and span[1] >= .45
                    and hull_area >= .28 and occupied >= 4)

    def _endpoint_spread(self, points, quad):
        """Require two populated, transversely supported longitudinal ends."""
        cv2, np = self._cv2, self._np
        if len(points) < self._MIN_POINTS:
            return False
        unit_quad = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
        to_board = cv2.getPerspectiveTransform(quad.astype(np.float32), unit_quad)
        board = cv2.perspectiveTransform(
            np.asarray(points, dtype=np.float32).reshape(-1, 1, 2),
            to_board)[:, 0]
        if (not np.isfinite(board).all()
                or (board < -.03).any() or (board > 1.03).any()):
            return False
        ends = (board[:, 0] <= .38, board[:, 0] >= .62)
        counts = [int(item.sum()) for item in ends]
        if min(counts) < 8:
            return False
        spans = []
        for item in ends:
            quantiles = np.quantile(board[item], [.1, .9], axis=0)
            spans.append(quantiles[1] - quantiles[0])
        overall = np.quantile(board, [.1, .9], axis=0)
        longitudinal_span = float(overall[1, 0] - overall[0, 0])
        centre_separation = float(board[ends[0], 0].mean()
                                  - board[ends[1], 0].mean())
        hull_area = float(cv2.contourArea(cv2.convexHull(board)))
        # Four-quadrant occupancy is inappropriate here: both ends may sit
        # inside the same transverse image cell while still supporting the
        # entire visible neck width independently.
        return bool(longitudinal_span >= .55
                    and centre_separation <= -.55
                    and all(float(span[1]) >= .24 for span in spans)
                    and hull_area >= .08)

    def _spread_supported(self, points, quad):
        if self._endpoint_weighted and self._sampling == "endpoint_bands":
            return (self._endpoint_spread(points, quad)
                    or self._well_spread(points, quad))
        return self._well_spread(points, quad)

    def _appearance_supported(self, correlations):
        """Allow only a short, high-confidence appearance outlier streak."""
        np = self._np
        if not np.isfinite(correlations).all():
            return False
        median = float(np.median(correlations))
        good_fraction = float(np.mean(correlations >= .8))
        if median >= .88 and good_fraction >= .75:
            self._partial_appearance_frames = 0
            self._appearance_mode = "full"
            return True
        guarded = (median >= .90 and good_fraction >= .60
                   and self._inliers >= 24 and self._inlier_ratio >= .90
                   and self._partial_appearance_frames < 2)
        if guarded:
            self._partial_appearance_frames += 1
            self._appearance_mode = "bounded_outlier"
            return True
        return False

    def initialize(self, gray, corners, exclusion_mask=None, allow_relaxed=True):
        """Replace calibration; invalid geometry/texture returns a lost snapshot."""
        self.invalidate("initializing")
        self._point_count = self._inliers = 0
        self._inlier_ratio = 0.0
        self._dependencies()
        cv2, np = self._cv2, self._np
        if not self._valid_frame(gray):
            return self.invalidate("invalid_frame")
        if (not isinstance(corners, (list, tuple)) or len(corners) != 4
                or any(not isinstance(p, dict)
                       or any(not isinstance(p.get(k), Real)
                              or isinstance(p.get(k), (bool, np.bool_))
                              for k in ("x", "y")) for p in corners)):
            return self.invalidate("invalid_corners")
        normalized = np.array([[p["x"], p["y"]] for p in corners], dtype=np.float64)
        if (not np.isfinite(normalized).all() or (normalized < 0).any()
                or (normalized > 1).any()):
            return self.invalidate("invalid_corners")
        height, width = gray.shape
        if (exclusion_mask is not None and
                (not isinstance(exclusion_mask, np.ndarray) or
                 exclusion_mask.dtype != np.uint8 or exclusion_mask.shape != gray.shape)):
            return self.invalidate("invalid_exclusion_mask")
        quad = (normalized * (width - 1, height - 1)).astype(np.float32)
        if not self._valid_quad(quad, gray.shape):
            return self.invalidate("invalid_corners")
        try:
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.fillConvexPoly(mask, np.rint(quad).astype(np.int32), 255)
            if exclusion_mask is not None:
                mask[exclusion_mask != 0] = 0
            points = self._select_points(
                gray, quad, mask, allow_relaxed=allow_relaxed)
            if points is None:
                return self.invalidate("insufficient_texture")
            self._point_count = len(points)
            if len(points) < self._MIN_POINTS:
                return self.invalidate("insufficient_texture")
            if not self._well_spread(points, quad):
                # Strong texture near the hand can dominate GFTT's quantiles
                # even when independently observed board points span the ROI.
                # Recheck a spatially balanced subset with the same gates.
                balanced = self._distributed_subset(points, quad)
                if not self._well_spread(balanced, quad):
                    return self.invalidate("insufficient_spread")
                points = balanced
                self._point_count = len(points)
        except cv2.error:
            return self.invalidate("opencv_error")
        self._previous, self._quad, self._points = gray.copy(), quad, points
        self._status, self._reason = "tracking", None
        self._inliers, self._inlier_ratio = len(points), 1.0
        return self.snapshot()

    def _distributed_subset(self, points, quad):
        cv2, np = self._cv2, self._np
        transform = cv2.getPerspectiveTransform(quad.astype(np.float32),
            np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]))
        board = cv2.perspectiveTransform(points, transform)[:, 0]
        chosen = [int(np.argmin(board[:, 0] + board[:, 1]))]
        distance = np.full(len(board), np.inf)
        while len(chosen) < min(24, len(points)):
            distance = np.minimum(distance, ((board - board[chosen[-1]]) ** 2).sum(axis=1))
            distance[chosen] = -1
            chosen.append(int(np.argmax(distance)))
        return points[chosen].copy()

    def _plausible_motion(self, homography, quad, require_convex=True,
                          outside_margin=0.):
        cv2, np = self._cv2, self._np
        if not np.isfinite(homography).all():
            return False
        # A projective pole inside the board can leave finite-looking corners.
        denominators = self._quad @ homography[2, :2] + homography[2, 2]
        if not ((denominators > 0).all() or (denominators < 0).all()):
            return False
        if np.abs(denominators).min() < .5 * np.abs(denominators).max():
            return False
        height, width = self._previous.shape
        if (not np.isfinite(quad).all()
                or (quad < -outside_margin).any()
                or (quad > (width - 1 + outside_margin,
                            height - 1 + outside_margin)).any()):
            return False
        edges = np.linalg.norm(quad - np.roll(quad, 1, axis=0), axis=1)
        area = cv2.contourArea(quad)
        if (area < max(80, width * height * .002)
                or edges.min() < 8 or area / edges.max() < 8):
            return False
        if require_convex and not cv2.isContourConvex(quad.astype(np.float32)):
            return False
        old_area = cv2.contourArea(self._quad, oriented=True)
        new_area = cv2.contourArea(quad, oriented=True)
        if self._fast_motion:
            area_limits = (.55, 1.8)
            scale_limits = (.62, 1.65)
            maximum_angle = 35
            maximum_corner_motion = .30
            maximum_center_motion = .25
        else:
            area_limits = (.65, 1.5)
            scale_limits = (.75, 1.33)
            maximum_angle = 25
            maximum_corner_motion = .15
            maximum_center_motion = .12
        if not area_limits[0] <= new_area / old_area <= area_limits[1]:
            return False
        old_edges = np.roll(self._quad, -1, axis=0) - self._quad
        new_edges = np.roll(quad, -1, axis=0) - quad
        old_lengths = np.linalg.norm(old_edges, axis=1)
        new_lengths = np.linalg.norm(new_edges, axis=1)
        scales = new_lengths / old_lengths
        cosines = (old_edges * new_edges).sum(axis=1) / (old_lengths * new_lengths)
        diagonal = float(np.hypot(*self._previous.shape))
        return bool((scales >= scale_limits[0]).all() and (scales <= scale_limits[1]).all()
                    and (cosines >= np.cos(np.deg2rad(maximum_angle))).all()
                    and np.linalg.norm(quad - self._quad, axis=1).max()
                    <= maximum_corner_motion * diagonal
                    and np.linalg.norm(quad.mean(axis=0) - self._quad.mean(axis=0))
                    <= maximum_center_motion * diagonal)

    def _one_end_motion(self, old, new, inliers):
        """Bridge a brief one-end occlusion with shape-preserving motion."""
        if (not self._endpoint_weighted
                or self._sampling not in ("endpoint_bands", "full_roi_fallback")
                or inliers.sum() < 8):
            return None
        cv2, np = self._cv2, self._np
        to_board = cv2.getPerspectiveTransform(
            self._quad, np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]))
        board = cv2.perspectiveTransform(
            old[inliers].reshape(-1, 1, 2), to_board)[:, 0]
        counts = (int((board[:, 0] <= .40).sum()),
                  int((board[:, 0] >= .60).sum()))
        supported = []
        for end_index, item in enumerate(
                (board[:, 0] <= .40, board[:, 0] >= .60)):
            if counts[end_index] < 8:
                supported.append(False)
                continue
            span = np.quantile(board[item], [.1, .9], axis=0)
            supported.append(float(span[1, 1] - span[0, 1]) >= .24)
        if sum(supported) != 1:
            return None
        strong = (board[:, 0] <= .40) if supported[0] else (board[:, 0] >= .60)
        strong_old, strong_new = old[inliers][strong], new[inliers][strong]
        fit = None
        for threshold, ratio, percentile in ((1.5, .75, 1.5),
                                             (3.0, .85, 2.5)):
            model, accepted = cv2.estimateAffinePartial2D(
                strong_old, strong_new, method=cv2.RANSAC,
                ransacReprojThreshold=threshold, maxIters=2000, confidence=.999)
            if model is None or accepted is None:
                continue
            accepted = accepted.ravel().astype(bool)
            if accepted.sum() < 8 or accepted.mean() < ratio:
                continue
            predicted = cv2.transform(strong_old[:, None, :], model)[:, 0]
            residual = np.linalg.norm(predicted - strong_new, axis=1)
            if np.percentile(residual, 90) > percentile:
                continue
            fit = model
            break
        if fit is None:
            return None
        homography = np.vstack((fit, (0., 0., 1.)))
        quad = cv2.perspectiveTransform(self._quad.reshape(-1, 1, 2),
                                        homography)[:, 0]
        height, width = self._previous.shape
        if not self._plausible_motion(
                homography, quad, require_convex=False, outside_margin=4.):
            return None
        quad[:, 0] = np.clip(quad[:, 0], 0, width - 1)
        quad[:, 1] = np.clip(quad[:, 1], 0, height - 1)
        return homography, quad

    def update(self, gray, exclusion_mask=None):
        """Advance one frame, or remain lost until an explicit initialize call."""
        if self._status != "tracking":
            return self.snapshot()
        self._point_count = self._inliers = 0
        self._inlier_ratio = 0.0
        if not self._valid_frame(gray):
            return self.invalidate("invalid_frame")
        if gray.shape != self._previous.shape:
            return self.invalidate("frame_size_changed")
        cv2, np = self._cv2, self._np
        if (exclusion_mask is not None and
                (not isinstance(exclusion_mask, np.ndarray) or
                 exclusion_mask.dtype != np.uint8 or exclusion_mask.shape != gray.shape)):
            return self.invalidate("invalid_exclusion_mask")
        started = time.perf_counter()
        self.timings = {}
        lk = dict(winSize=(21, 21), maxLevel=3,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01),
                  minEigThreshold=.0001)
        try:
            forward, status, error = self._flow(
                self._previous, gray, self._points, lk)
            forward_done = time.perf_counter()
            self.timings["forward_ms"] = (forward_done - started) * 1000
            if self._flow_backend is not None:
                self.timings["forward_backend"] = dict(getattr(self._flow_backend, "timings", {}))
            if forward is None or status is None or error is None:
                return self.invalidate("optical_flow_failed")
            height, width = gray.shape
            old = self._points[:, 0]
            new = forward[:, 0]
            good = (status.ravel() == 1) & np.isfinite(new).all(axis=1)
            good &= (new >= 0).all(axis=1) & (new <= (width - 1, height - 1)).all(axis=1)
            good &= np.isfinite(error.ravel()) & (error.ravel() <= 20)
            if exclusion_mask is not None:
                xy = np.rint(new).astype(int)
                inside = ((xy[:, 0] >= 0) & (xy[:, 0] < width) &
                          (xy[:, 1] >= 0) & (xy[:, 1] < height))
                visible = np.zeros(len(xy), dtype=bool)
                visible[inside] = exclusion_mask[xy[inside, 1], xy[inside, 0]] == 0
                good &= visible
            old, new = old[good], new[good]
            self._point_count = len(new)
            if len(new) < self._MIN_POINTS:
                return self.invalidate("insufficient_flow")
            backward, status, error = self._flow(
                gray, self._previous, new.reshape(-1, 1, 2), lk)
            backward_done = time.perf_counter()
            self.timings["backward_ms"] = (backward_done - forward_done) * 1000
            if self._flow_backend is not None:
                self.timings["backward_backend"] = dict(getattr(self._flow_backend, "timings", {}))
            if backward is None or status is None or error is None:
                return self.invalidate("optical_flow_failed")
            good = ((status.ravel() == 1) & np.isfinite(backward[:, 0]).all(axis=1)
                    & np.isfinite(error.ravel()) & (error.ravel() <= 20)
                    & (np.linalg.norm(backward[:, 0] - old, axis=1) <= 1.25))
            old, new = old[good], new[good]
            self._point_count = len(new)
            minimum_consistent = max(self._MIN_POINTS, .6 * len(self._points))
            if (len(new) < self._MIN_POINTS or
                    (len(new) < minimum_consistent
                     and self._sampling != "endpoint_bands")):
                return self.invalidate("insufficient_consistent_flow")
            homography, inliers = cv2.findHomography(
                old, new, cv2.RANSAC, 2.0, maxIters=2000, confidence=.995)
            self.timings["ransac_ms"] = (time.perf_counter() - backward_done) * 1000
            if homography is None or inliers is None:
                return self.invalidate("homography_failed")
            inliers = inliers.ravel().astype(bool)
            self._inliers = int(inliers.sum())
            self._inlier_ratio = self._inliers / len(new)
            if self._inliers < self._MIN_POINTS or self._inlier_ratio < .7:
                return self.invalidate("insufficient_inliers")
            quad = cv2.perspectiveTransform(self._quad.reshape(-1, 1, 2), homography)[:, 0]
            if not self._plausible_motion(homography, quad):
                return self.invalidate("invalid_motion_or_out_of_frame")
            spread = (self._spread_supported(old[inliers], self._quad)
                      and self._spread_supported(new[inliers], quad))
            if not spread:
                bridged = self._one_end_motion(old, new, inliers)
                if bridged is None or self._partial_frames >= 20:
                    return self.invalidate("insufficient_spread")
                homography, quad = bridged
                self._partial_frames += 1
                self._motion_mode = "one_end_guarded"
            else:
                self._partial_frames = 0
                self._motion_mode = "full"
            appearance_started = time.perf_counter()
            self.timings["geometry_validation_ms"] = (appearance_started - backward_done) * 1000
            # Shared fret/string edges can fit even after a scene cut. Verify
            # local appearance in the previous frame's projective coordinates.
            correlations = self._patch_correlations(gray, self._previous, old[inliers], homography)
            self.timings["appearance_ms"] = (time.perf_counter() - appearance_started) * 1000
            if not self._appearance_supported(correlations):
                return self.invalidate("appearance_changed")
        except cv2.error:
            return self.invalidate("opencv_error")
        self._previous, self._quad = gray.copy(), quad
        self._points = new[inliers].reshape(-1, 1, 2).copy()
        self.timings["total_ms"] = (time.perf_counter() - started) * 1000
        return self.snapshot()

    def _aligned_patches(self, gray, homography, points):
        """Warp only pixels needed by the 15x15 appearance patches."""
        cv2, np = self._cv2, self._np
        origins = np.floor(points - 7).astype(np.float32)
        if self._patch_grid is None:
            yy, xx = np.indices((16,16), dtype=np.float32)
            self._patch_grid = np.stack((xx, yy), axis=-1)
        coordinates = origins[:, None, None, :] + self._patch_grid
        # getRectSubPix on the old full warp replicated its image border.
        coordinates[..., 0] = np.clip(coordinates[..., 0], 0, gray.shape[1]-1)
        coordinates[..., 1] = np.clip(coordinates[..., 1], 0, gray.shape[0]-1)
        source = cv2.perspectiveTransform(coordinates.astype(np.float32).reshape(-1,1,2), homography)
        source = source.reshape(len(points)*16,16,2)
        atlas = cv2.remap(gray, source[...,0], source[...,1], cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        centers = points - origins
        return np.stack([cv2.getRectSubPix(atlas[i*16:(i+1)*16], (15,15), tuple(center))
                         for i,center in enumerate(centers)])

    def _patch_correlations(self, aligned, previous, points, homography=None):
        """Batch equal-sized normalized patch correlations; preserve flat-template semantics."""
        cv2, np = self._cv2, self._np
        if homography is not None and self._warp_native is not None:
            current, old = np.ascontiguousarray(aligned), np.ascontiguousarray(previous)
            coordinates = np.ascontiguousarray(points,dtype=np.float32)
            transform = np.ascontiguousarray(homography,dtype=np.float64)
            first,second = np.empty((2,len(points),15,15),np.uint8)
            height,width = current.shape
            if self._warp_native(current.ctypes.data,old.ctypes.data,width,height,
                    coordinates.ctypes.data,len(points),transform.ctypes.data,
                    first.ctypes.data,second.ctypes.data) != 0:
                raise ValueError('Native OpenCV patch extraction failed')
            from patch_similarity import patch_correlations
            return patch_correlations(first,second,flat_template=True)
        first = (self._aligned_patches(aligned,homography,points) if homography is not None
                 else np.stack([cv2.getRectSubPix(aligned,(15,15),tuple(p)) for p in points]))
        second = np.stack([cv2.getRectSubPix(previous,(15,15),tuple(p)) for p in points])
        from patch_similarity import patch_correlations
        return patch_correlations(first,second,flat_template=True)

    def _flow(self, previous, current, points, options):
        if self._flow_backend is not None:
            return self._flow_backend(previous, current, points)
        return self._cv2.calcOpticalFlowPyrLK(previous, current, points, None, **options)
