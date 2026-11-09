"""Manual-ROI fretboard lattice tracking with conservative current-frame support."""

from copy import deepcopy
import ctypes
import math
import time
import os
import threading
from pathlib import Path

from fretboard_tracker import FretboardTracker


class RoiFretboardTracker:
    """Track a supplied neck ROI, then measure its visible string/fret lattice.

    The ROI is followed by :class:`FretboardTracker`. Default mode measures
    the lattice every frame. Opt-in fast mode projects the last measured
    lattice only after current optical-flow, appearance and line-support
    checks, and bounds the interval between full measurements.
    """

    _WIDTH = 320
    _HEIGHT = 128
    _FLOW_SIZE = (320, 200)
    _FLOW_POINTS = 48
    _FAST_MEASURE_INTERVAL = .75  # Bounded remeasurement; LK remains per-frame.
    _PROJECTION_MAX_AGE = 1.75
    _MEASUREMENT_SLOT = threading.BoundedSemaphore(1)

    def __init__(self, flow_backend=None, projection_backend=None, fast_tracking=False,
                 async_measurements=False, endpoint_weighted=True):
        self.fast_tracking = fast_tracking
        self.async_measurements = async_measurements
        self.endpoint_weighted = bool(endpoint_weighted)
        self._measurement_lock = threading.Lock()
        self._measurement_busy = False
        self._measurement_result = None
        self._measurement_epoch = 0
        self._measurement_failures = 0
        self._FLOW_POINTS = int(os.environ.get("FRETLENS_FLOW_POINTS", "48"))
        if self._FLOW_POINTS not in (24, 32, 48):
            raise ValueError("FRETLENS_FLOW_POINTS must be 24, 32 or 48")
        self._full_at = 0
        self._fast_frames = 0
        self._flow_backend = flow_backend
        self._projection_backend = projection_backend
        self.timings = {}
        self._cv2 = self._np = None
        self._ridge_native = None
        self._fret_models = {}
        self._last_strings_estimated = False
        self._roi = FretboardTracker(flow_backend=flow_backend,
                                     fast_motion=fast_tracking,
                                     endpoint_weighted=self.endpoint_weighted)
        self._roi._MAX_POINTS = 240
        self._snapshot = None
        self._updates = 0
        self._refreshes = 0
        self._refresh_attempt_update = 0
        self._refresh_point_count = 0
        self._refresh_attempt_points = 0
        self._last_roi_corners = None
        self._reacquisitions = 0
        self._search_at = float('-inf')
        self._measure_at = float('-inf')
        self.invalidate("not_initialized")

    def _dependencies(self):
        if self._cv2 is None:
            import cv2
            import numpy as np
            self._cv2, self._np = cv2, np
            library = Path(__file__).with_name('libhand_q16.so')
            if library.is_file():
                native = ctypes.CDLL(str(library))
                if hasattr(native,'fret_ridge_profiles'):
                    native.fret_ridge_profiles.argtypes = [ctypes.c_void_p]*3+[ctypes.c_int]*2
                    native.fret_ridge_profiles.restype = ctypes.c_int
                    self._ridge_native = native

    def snapshot(self):
        return deepcopy(self._snapshot)

    def invalidate(self, reason):
        with self._measurement_lock:
            self._measurement_epoch += 1
            self._measurement_result = None
            self._measurement_failures = 0
        self._roi.invalidate(reason)
        self._last_roi_corners = None
        self._reacquisitions = 0
        self._snapshot = self._empty(str(reason), None, "lost")
        return self.snapshot()

    def _empty(self, reason, roi_corners, roi_status):
        return {"mode": "auto", "backend": "roi_lk_sobel_projection",
                "status": "searching", "reason": reason,
                "corners": None, "strings": [], "frets": [],
                "absolute_frets_known": False, "orientation": "image_order",
                "confidence": 0.0, "string_count": 0, "fret_count": 0,
                "roi_corners": roi_corners, "roi_status": roi_status,
                "stage": "roi_tracking", "flow": {}}

    def initialize(self, gray, corners, color=None):
        """Start from four normalized, perimeter-ordered ROI corners.

        ``color`` is optional BGR or RGBA at the same resolution as ``gray``.
        It is used only while measuring the rectified fretboard, never for
        global frame segmentation.
        """
        # Measure before initializing the native flow tracker. On the board,
        # loading the tracking backend first can change the ridge response of
        # the independent lattice detector and turn a valid lattice into a
        # miss. The acquisition path needs one reliable measurement here.
        primed_lattice = None
        if self.fast_tracking and not self.async_measurements:
            try:
                primed_lattice = self._detect(gray, corners, color)
            except (ValueError, TypeError):
                primed_lattice = None
        with self._measurement_lock:
            self._measurement_epoch += 1
            self._measurement_result = None
            self._measurement_failures = 0
        self._updates = self._refreshes = 0
        self._reacquisitions = 0
        self._refresh_attempt_update = 0
        self._refresh_point_count = 0
        self._refresh_attempt_points = 0
        self._measure_at = float('-inf')
        flow_gray = self._flow_gray(gray)
        exclusion = self._hand_exclusion_mask(color, flow_gray.shape)
        state = self._roi.initialize(flow_gray, corners, exclusion)
        if state["status"] == "tracking":
            self._last_roi_corners = state["corners"]
            self._subsample_flow_points()
            state = self._roi.snapshot()
            self._refresh_point_count = state['points']
            self._refresh_attempt_points = state['points']
        return self._from_roi(
            gray, state, False, color, primed_lattice=primed_lattice)

    def update(self, gray, color=None):
        """Advance the ROI flow track and return only present-frame lattice data."""
        flow_gray = self._flow_gray(gray)
        exclusion = self._hand_exclusion_mask(color, flow_gray.shape)
        started = time.perf_counter()
        state = self._roi.update(flow_gray, exclusion)
        flow_done = time.perf_counter()
        self.timings = dict(self._roi.timings)
        self.timings["flow_validation_ms"] = (flow_done - started) * 1000
        self._updates += 1
        if state["status"] == "tracking":
            self._last_roi_corners = state["corners"]
        elif color is not None:
            recovered = self._color_reacquire(flow_gray, color)
            if recovered is not None:
                state = recovered
                self._last_roi_corners = state["corners"]
                self._subsample_flow_points()
                state = self._roi.snapshot()
                state["reacquired"] = True
                self._refresh_point_count = state["points"]
                self._refresh_attempt_points = state["points"]
        # Refresh only after flow, RANSAC, spatial and appearance checks pass.
        # A failed candidate cannot destroy the existing validated track.
        # Sparse but stable necks cannot necessarily supply 28 usable points.
        # Refresh on actual attrition, with bounded retries even on failure.
        minimum = self._roi._MIN_POINTS
        depleted = state['points'] < min(self._refresh_point_count,
            max(minimum+2, min(28, .65*self._refresh_point_count)))
        urgent = state['points'] <= minimum+2 and state['points'] < self._refresh_attempt_points
        refresh_started = time.perf_counter()
        if (state["status"] == "tracking" and state["inlier_ratio"] >= .85
                and (urgent or (self._updates-self._refresh_attempt_update >= 5
                     and (self._updates % 15 == 0 or depleted)))):
            self._refresh_attempt_update = self._updates
            self._refresh_attempt_points = state['points']
            candidate = FretboardTracker(flow_backend=self._flow_backend,
                                         fast_motion=self.fast_tracking,
                                         endpoint_weighted=self.endpoint_weighted)
            proposed = candidate.initialize(
                flow_gray, state["corners"], exclusion, allow_relaxed=False)
            if proposed["status"] == "tracking" and proposed['points'] >= state['points']:
                self._roi = candidate
                self._subsample_flow_points()
                state = self._roi.snapshot()
                self._refresh_point_count = state['points']
                self._refresh_attempt_points = state['points']
                self._refreshes += 1
        self.timings['feature_refresh_ms'] = (time.perf_counter()-refresh_started)*1000
        refresh_done = time.perf_counter()
        previous = self._snapshot
        result = None
        if (self.fast_tracking and state["status"] == "tracking"
                and previous.get("status") == "tracking"
                and not state.get("reacquired")):
            result = self._project_lattice(gray, previous, state)
        if result is None:
            result = self._from_roi(gray, state, True, color)
        else:
            self._fast_frames += 1
        self.timings["lattice_refresh_ms"] = (time.perf_counter() - refresh_done) * 1000
        result["flow"]["feature_refreshes"] = self._refreshes
        self._snapshot = result
        return self.snapshot()

    def _color_reacquire(self, gray, color):
        """Recover in the loss frame when a colour proposal is available."""
        self._dependencies()
        cv2, np = self._cv2, self._np
        if (not isinstance(color, np.ndarray) or color.dtype != np.uint8
                or color.ndim != 3 or color.shape[2] not in (3, 4)):
            return None
        started = time.perf_counter()
        try:
            from neck_color import find_color_neck_regions
            bgr = (cv2.cvtColor(color, cv2.COLOR_RGBA2BGR)
                   if color.shape[2] == 4 else color)
            if bgr.shape[:2] != gray.shape:
                bgr = cv2.resize(bgr, (gray.shape[1], gray.shape[0]),
                                 interpolation=cv2.INTER_AREA)
            proposals = sorted(find_color_neck_regions(bgr, limit=3),
                               key=lambda item: item["score"], reverse=True)
            exclusion = self._hand_exclusion_mask(color, gray.shape)
            for proposal in proposals:
                candidate = FretboardTracker(
                    flow_backend=self._flow_backend,
                    fast_motion=self.fast_tracking,
                    endpoint_weighted=self.endpoint_weighted)
                state = candidate.initialize(
                    gray, proposal["corners"], exclusion)
                if state["status"] == "tracking":
                    self._roi = candidate
                    self._reacquisitions += 1
                    return state
        except Exception:
            return None
        finally:
            self.timings["color_reacquire_ms"] = (
                time.perf_counter() - started) * 1000
        return None

    def _project_lattice(self, gray, previous, state):
        cv2, np = self._cv2, self._np
        if not previous.get("strings") or not previous.get("frets"):
            return None
        if float(gray.std()) < 4.:
            return None
        age_ms = (time.monotonic() - self._full_at) * 1000
        if age_ms < 0 or age_ms > self._PROJECTION_MAX_AGE * 1000:
            return None
        source = np.float32([[p["x"], p["y"]] for p in previous["roi_corners"]])
        target = np.float32([[p["x"], p["y"]] for p in state["corners"]])
        transform = cv2.getPerspectiveTransform(source, target)
        result = deepcopy(previous)
        lines = result["strings"] + result["frets"]
        raw = np.float32([[[line[k]["x"], line[k]["y"]] for k in ("start", "end")] for line in lines])
        mapped = cv2.perspectiveTransform(raw.reshape(-1,1,2), transform).reshape(-1,2,2)
        if not np.isfinite(mapped).all() or np.any(mapped < 0) or np.any(mapped > 1):
            return None
        for index, (line, ends) in enumerate(zip(lines, mapped)):
            for key, (x,y) in zip(("start", "end"), ends):
                line[key] = {"x": float(x), "y": float(y)}
            line["observed"] = False
            line["estimated"] = True
            line["projected"] = True
            line["source"] = ("tracked_string_lattice" if index < 6
                              else "tracked_fret_lattice")
        # Current-frame contrast across every projected wire rejects partial
        # occlusion evidence for diagnostics. The neck geometry remains the
        # authoritative gate: rejecting every line after a short measurement
        # expiry made the web overlay disappear even while the ROI still tracked.
        height, width = gray.shape
        pixels = mapped * np.array([width-1,height-1])
        delta = pixels[:,1]-pixels[:,0]
        normals = np.column_stack((-delta[:,1],delta[:,0])) / np.maximum(np.linalg.norm(delta,axis=1),1)[:,None]
        center = pixels[:,0,None,:] + np.linspace(.1,.9,16)[None,:,None]*delta[:,None,:]
        locations = np.rint(center[:,None,:,:]+np.array([-3,0,3])[None,:,None,None]*normals[:,None,None,:]).astype(int)
        samples = gray[np.clip(locations[...,1],0,height-1),np.clip(locations[...,0],0,width-1)].astype(float)
        support = np.maximum(abs(samples[:,1]-samples[:,0]),abs(samples[:,1]-samples[:,2])).mean(axis=1)
        if state.get("status") != "tracking":
            return None
        if result.get("string_observation") != "estimated_live_neck_width" and min(support[:6]) < 6:
            return None
        if len(support) > 6 and float(np.median(support[6:])) < 5:
            return None
        result["roi_corners"] = state["corners"]
        result["corners"] = [result["strings"][0]["start"],result["strings"][0]["end"],
                             result["strings"][-1]["end"],result["strings"][-1]["start"]]
        result["stage"] = "tracked_lattice"
        result["flow"] = {k:state.get(k) for k in (
            "points", "inliers", "inlier_ratio", "sampling", "endpoint_points",
            "motion_mode")}
        result["flow"]["reacquired"] = bool(state.get("reacquired", False))
        result["flow"]["reacquisitions"] = self._reacquisitions
        result["lattice_age_ms"] = age_ms
        result["lattice_ttl_ms"] = self._PROJECTION_MAX_AGE * 1000
        result["lattice_observation"] = "flow_projected"
        result["line_support"] = {
            "strings": support[:6].tolist(),
            "frets": support[6:].tolist(),
            "string_median": float(np.median(support[:6])),
            "fret_median": float(np.median(support[6:])),
        }
        return result

    def _flow_gray(self, gray):
        self._dependencies()
        if (not isinstance(gray, self._np.ndarray) or gray.dtype != self._np.uint8
                or gray.ndim != 2):
            return gray
        return self._cv2.resize(gray, self._FLOW_SIZE, interpolation=self._cv2.INTER_LINEAR)

    def _hand_exclusion_mask(self, color, shape):
        """Exclude visible skin from neck flow so a fretting hand cannot steer the ROI."""
        if color is None:
            return None
        self._dependencies()
        cv2, np = self._cv2, self._np
        if (not isinstance(color, np.ndarray) or color.dtype != np.uint8 or
                color.ndim != 3 or color.shape[2] not in (3, 4)):
            return None
        rgb = color[:, :, :3] if color.shape[2] == 4 else cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        small_rgb = cv2.resize(rgb, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
        ycrcb = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2YCrCb)
        skin = cv2.inRange(ycrcb, (36, 150, 75), (255, 180, 135))
        return cv2.dilate(skin, np.ones((5, 5), np.uint8), iterations=1)

    def _subsample_flow_points(self):
        """Keep a spatially diverse, verified subset for bounded LK work."""
        np = self._np
        points = self._roi._points[:, 0]
        if len(points) <= self._FLOW_POINTS:
            return
        chosen = [int(np.argmin(points[:, 0] + points[:, 1])),
                  int(np.argmin(points[:, 0] - points[:, 1])),
                  int(np.argmax(points[:, 0] - points[:, 1])),
                  int(np.argmax(points[:, 0] + points[:, 1]))]
        chosen = list(dict.fromkeys(chosen))
        distance = np.full(len(points), np.inf)
        while len(chosen) < self._FLOW_POINTS:
            latest = points[chosen[-1]]
            distance = np.minimum(distance, ((points - latest) ** 2).sum(axis=1))
            distance[chosen] = -1
            chosen.append(int(np.argmax(distance)))
        subset = points[chosen].reshape(-1, 1, 2).copy()
        if self._roi._well_spread(subset, self._roi._quad):
            self._roi._points = subset
            self._roi._point_count = len(subset)
            self._roi._inliers = len(subset)

    def _from_roi(self, gray, state, is_update, color=None, primed_lattice=None):
        roi_corners = state["corners"]
        flow = {key: state.get(key) for key in (
            "points", "inliers", "inlier_ratio", "sampling", "endpoint_points",
            "motion_mode")}
        flow["reacquired"] = bool(state.get("reacquired", False))
        flow["reacquisitions"] = self._reacquisitions
        if state["status"] != "tracking":
            with self._measurement_lock:
                self._measurement_epoch += 1
                self._measurement_result = None
            reason = "roi_tracking_lost" if is_update else state["reason"]
            self._snapshot = self._empty(reason, None, state["status"])
            self._snapshot["flow"] = flow
            if is_update:
                self._snapshot["flow_reason"] = state["reason"]
            return self.snapshot()
        now = time.monotonic()
        if primed_lattice is not None:
            confidence, strings, frets = primed_lattice
            strings_estimated = bool(self._last_strings_estimated)
            self._full_at = now
            self._fast_frames = 0
            result = self._lattice_result(
                primed_lattice, roi_corners, strings_estimated)
            result["flow"] = flow
            result["stage"] = "lattice"
            self._snapshot = result
            return self.snapshot()
        result = self._empty("no_supported_lattice", roi_corners, "tracking")
        result["flow"] = flow
        # A failed search supplies no geometry. Retain only current validated
        # neck flow while bounding expensive lattice reacquisition work.
        if self.async_measurements:
            with self._measurement_lock:
                completed, self._measurement_result = self._measurement_result, None
            if completed is not None:
                measured_at, measured = completed
                if now-measured_at < .6:
                    self._full_at = measured_at
                    projected = self._project_lattice(gray, measured, state)
                    if projected is not None:
                        self._snapshot = projected
                        self._fast_frames = 0
                        return self.snapshot()
            previous = self._snapshot
            if previous is not None and previous.get("status") == "tracking":
                projected = self._project_lattice(gray, previous, state)
                if projected is not None:
                    self._snapshot = projected
                    self._fast_frames += 1
                    return self.snapshot()
            self._submit_measurement(gray, roi_corners, now, color)
            self._snapshot = result
            return self.snapshot()
        if (self.fast_tracking and is_update and self._snapshot.get('status') != 'tracking'
                and now-self._measure_at < self._FAST_MEASURE_INTERVAL):
            return result
        self._search_at = now
        self._measure_at = now
        try:
            lattice = self._detect(gray, roi_corners, color)
        except (ValueError, TypeError):
            lattice = None
        if lattice is not None:
            self._full_at = now
            self._fast_frames = 0
            confidence, strings, frets = lattice
            strings_estimated = bool(self._last_strings_estimated)
            result = self._lattice_result(lattice, roi_corners, strings_estimated)
            result["flow"] = flow
        self._snapshot = result
        return self.snapshot()

    def _lattice_result(self, lattice, roi_corners, strings_estimated):
        confidence, strings, frets = lattice
        result = self._empty(None, roi_corners, "tracking")
        result.update(
            status="tracking", reason=None,
            corners=[strings[0][0], strings[0][1],
                     strings[-1][1], strings[-1][0]],
            strings=[self._line(x, observed=not strings_estimated)
                     for x in strings],
            frets=[dict(self._line(x), index=i)
                   for i, x in enumerate(frets)],
            confidence=confidence, string_count=6,
            fret_count=len(frets), stage="lattice")
        if strings_estimated:
            result["string_observation"] = "estimated_live_neck_width"
        return result

    def _submit_measurement(self, gray, corners, now, color=None):
        with self._measurement_lock:
            interval = min(2., self._FAST_MEASURE_INTERVAL*(1+self._measurement_failures))
            if self._measurement_busy or now-self._measure_at < interval:
                return
            if not self._MEASUREMENT_SLOT.acquire(blocking=False):
                return
            self._measurement_busy = True
            self._measure_at = now
            epoch = self._measurement_epoch
        source, quad = gray.copy(), deepcopy(corners)
        source_color = None if color is None else color.copy()

        def measure():
            try:
                if hasattr(os, 'setpriority') and hasattr(threading, 'get_native_id'):
                    try:
                        os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 15)
                    except OSError:
                        pass
                # Independent PS workspace; the live PL flow lock is never taken.
                measurement = RoiFretboardTracker()
                lattice = measurement._detect(source, quad, source_color)
                with self._measurement_lock:
                    if epoch == self._measurement_epoch:
                        self._measurement_failures = (0 if lattice is not None else
                            min(3,self._measurement_failures+1))
                if lattice is not None:
                    confidence, strings, frets = lattice
                    strings_estimated = bool(measurement._last_strings_estimated)
                    result = self._empty(None, quad, 'tracking')
                    result.update(status='tracking', confidence=confidence, string_count=6,
                        fret_count=len(frets), strings=[self._line(x) for x in strings],
                        frets=[dict(self._line(x), index=i) for i,x in enumerate(frets)])
                    if strings_estimated:
                        result["strings"] = [self._line(x, observed=False) for x in strings]
                        result["string_observation"] = "estimated_live_neck_width"
                    with self._measurement_lock:
                        if epoch == self._measurement_epoch:
                            self._measurement_result = now, result
            except (ValueError, TypeError):
                pass
            finally:
                with self._measurement_lock:
                    self._measurement_busy = False
                    self._measure_at = time.monotonic()
                self._MEASUREMENT_SLOT.release()

        threading.Thread(target=measure, name='ps-roi-measurement', daemon=True).start()

    @staticmethod
    def _line(points, observed=True):
        return {"start": points[0], "end": points[1], "observed": bool(observed),
                "estimated": not observed,
                "source": "measured" if observed else "live_neck_width_six_equal_lanes"}

    def _peaks(self, profile, minimum, limit, contrast=3.):
        """Merge the two Sobel sides of a bright or dark narrow wire."""
        np = self._np
        profile = profile.astype(float)
        baseline = float(np.median(profile))
        mad = float(np.median(np.abs(profile - baseline))) + 1.0
        indices = np.flatnonzero((profile[1:-1] >= profile[:-2]) &
            (profile[1:-1] >= profile[2:]) &
            (profile[1:-1] >= baseline + contrast * mad)) + 1
        candidates = list(zip(indices.tolist(), profile[indices].tolist()))
        groups = []
        for index, strength in sorted(candidates, key=lambda item: -item[1]):
            for group in groups:
                if abs(index - group[0]) <= minimum:
                    total = group[1] + strength
                    group[0] = (group[0] * group[1] + index * strength) / total
                    group[1] = total
                    break
            else:
                groups.append([float(index), float(strength)])
        return sorted(groups, key=lambda item: -item[1])[:limit]

    def _six_strings(self, candidates):
        np = self._np
        candidates = sorted(candidates, key=lambda item: item[0])
        if len(candidates) < 6:
            return None
        coordinates = np.array([item[0] for item in candidates])
        strengths = np.array([item[1] for item in candidates])
        strengths = np.minimum(strengths,np.median(strengths))
        # Six observed ridges need not be consecutive among background peaks.
        # Endpoint hypotheses bound the work; no missing string is synthesized.
        starts,ends = np.triu_indices(len(candidates),k=5)
        pitch = (coordinates[ends]-coordinates[starts])/5
        keep = (pitch >= 9)&(pitch <= 38)
        if not keep.any():
            return None
        starts,ends,pitch = starts[keep],ends[keep],pitch[keep]
        targets = coordinates[starts,None]+pitch[:,None]*np.arange(6)
        indices = np.abs(coordinates[None,:,None]-targets[:,None,:]).argmin(axis=1)
        positions = coordinates[indices]
        error = np.max(np.abs(np.diff(positions,axis=1)-pitch[:,None]),axis=1)/pitch
        fit_error = np.max(np.abs(positions-targets),axis=1)/pitch
        valid = (np.diff(indices,axis=1)>0).all(axis=1)&(error<=.20)&(fit_error<=.12)
        if not valid.any():
            return None
        score = np.where(valid,strengths[indices].sum(axis=1)*(1-error),-np.inf)
        best = int(np.argmax(score))
        return float(score[best]),positions[best],float(error[best])

    def _ridge_strings(self, rectified):
        """Measure narrow two-sided ridges away from bright transverse wires."""
        cv2,np = self._cv2,self._np
        if self._ridge_native is not None:
            image = np.ascontiguousarray(rectified,dtype=np.uint8)
            profiles = np.empty((2,self._HEIGHT),np.float32)
            valid = np.empty((2,self._HEIGHT),np.uint8)
            if self._ridge_native.fret_ridge_profiles(image.ctypes.data,profiles.ctypes.data,
                    valid.ctypes.data,self._WIDTH,self._HEIGHT) != 0:
                raise ValueError('Invalid native ridge dimensions')
            best = None
            for profile,support in zip(profiles,valid):
                candidates = [(y,strength) for y,strength in self._peaks(profile,2,24,contrast=2.)
                              if support[int(round(y))]]
                fit = self._six_strings(candidates)
                if fit is not None and (best is None or fit[0] > best[0]):
                    best = fit
            return best
        image = rectified.astype(np.float32)
        change = np.abs(np.diff(image,axis=1)).mean(axis=0)
        change = np.r_[change,change[-1]]
        columns = np.flatnonzero(change <= np.quantile(change,.60))
        if len(columns) < self._WIDTH//4:
            return None
        best = None
        for sign in (1.,-1.):
            center = sign*image
            ridge = np.zeros_like(center)
            ridge[3:-3] = np.minimum(center[3:-3]-center[:-6],center[3:-3]-center[6:])
            profile = np.maximum(ridge[:,columns],0).mean(axis=1)
            profile[:4] = profile[-4:] = 0
            peaks = self._peaks(profile,2,24,contrast=2.)
            local = cv2.dilate(ridge,np.ones((5,1),np.uint8))
            supported = local[:,columns]>3
            valid = supported.mean(axis=1)>=.25
            for part in np.array_split(supported,4,axis=1):
                valid &= part.mean(axis=1)>=.10
            candidates = [(y,strength) for y,strength in peaks if valid[int(round(y))]]
            fit = self._six_strings(candidates)
            if fit is None:
                continue
            # Every returned string has present-frame, spatially distributed
            # intensity support. A lattice fit cannot supply an invisible wire.
            if best is None or fit[0] > best[0]:
                best = fit
        return best

    def _fret_spacing(self, positions):
        """Reject affine grids while accepting a contiguous equal-tempered run."""
        np = self._np
        if not 5 <= len(positions) <= 25:
            return None
        gaps = np.diff(positions)
        if np.min(gaps) < 4:
            return None
        uniform = float(np.sqrt(np.mean((positions - np.linspace(positions[0], positions[-1], len(positions))) ** 2)))
        model = self._fret_models.get(len(positions))
        if model is None:
            basis = np.exp(-math.log(2)*np.arange(len(positions))/12.)
            design = np.column_stack((np.ones(len(basis)), basis))
            projector = np.linalg.lstsq(design, np.eye(len(basis)), rcond=None)[0]
            model = self._fret_models[len(positions)] = basis, design, projector
        basis, design, projector = model
        best = None
        for reverse in (False, True):
            values = positions[::-1] if reverse else positions
            fitted = design.dot(projector.dot(values))
            error = float(np.sqrt(np.mean((fitted - values) ** 2)))
            best = error if best is None else min(best, error)
        ideal = (basis-basis[0])/(basis[-1]-basis[0])*(positions[-1]-positions[0])
        expected_curvature = float(np.sqrt(np.mean((ideal-np.linspace(0,ideal[-1],len(ideal)))**2)))
        if (best is None or best > 2.4 or
                uniform < max(.6,expected_curvature*.45) or best >= uniform * .62):
            return None
        return best

    def _ridge_frets(self, rectified, y0, y1):
        """Observe wire centers despite Sobel double edges and partial fingers."""
        cv2,np = self._cv2,self._np
        image = rectified[y0:y1].astype(np.float32)
        change = np.abs(np.diff(image,axis=0)).mean(axis=1)
        change = np.r_[change,change[-1]]
        rows = np.flatnonzero(change<=np.quantile(change,.6))
        best = None
        for sign in (1.,-1.):
            center = image*sign
            ridge = np.zeros_like(center)
            ridge[:,3:-3] = np.minimum(center[:,3:-3]-center[:,:-6],center[:,3:-3]-center[:,6:])
            profile = np.maximum(ridge[rows],0).mean(axis=0)
            support = cv2.dilate(ridge,np.ones((1,3),np.uint8))>5
            valid = support.mean(axis=0)>=.3
            parts = sum(part.mean(axis=0)>=.1 for part in np.array_split(support,4,axis=0))
            valid &= parts>=3
            candidates = self._peaks(profile,4,25,contrast=2.)
            positions = np.array(sorted(x for x,strength in candidates
                if 3<=x<self._WIDTH-4 and valid[int(round(x))]))
            # A nut or bright region edge can add an end peak. Never discard
            # interior wires to force a scale fit, or synthesize unseen frets.
            for left in range(3):
                for right in range(3):
                    selected = positions[left:len(positions)-right if right else None]
                    error = self._fret_spacing(selected)
                    if error is not None and (best is None or
                            (len(selected),-error)>(len(best[0]),-best[1])):
                        best = selected,error
        return best

    def _hough_frets(self, rectified, y0, y1):
        """Recover long, near-vertical fret wires after projection.

        The hand can split several wires into short segments and a global
        column profile then loses them. Hough segments preserve each wire's
        angle; the equal-tempered spacing gate still rejects unrelated grids.
        """
        cv2, np = self._cv2, self._np
        crop = rectified[max(2, y0):min(self._HEIGHT - 2, y1)]
        if crop.shape[0] < 24:
            return None
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(crop)
        edges = cv2.Canny(cv2.GaussianBlur(enhanced, (3, 3), 0),
                          30, 90, apertureSize=3)
        minimum_length = max(24, int(crop.shape[0] * .20))
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180., threshold=16,
                                minLineLength=minimum_length, maxLineGap=18)
        samples = []
        for line in ([] if lines is None else lines[:, 0]):
            x0, y0_line, x1, y1_line = [float(value) for value in line]
            dx, dy = x1-x0, y1_line-y0_line
            if abs(dy) < 3*abs(dx) or abs(dy) < minimum_length:
                continue
            center = x0 + dx*.5
            if 6 < center < self._WIDTH-7:
                samples.append((center, abs(dy)))
        if len(samples) < 5:
            return None

        clusters = []
        for position, weight in sorted(samples):
            if clusters and position-clusters[-1]["last"] <= 5.:
                cluster = clusters[-1]
                cluster["total"] += weight
                cluster["position"] += position*weight
                cluster["last"] = position
            else:
                clusters.append(dict(position=position*weight, total=weight,
                                     last=position))
        positions = np.asarray([item["position"]/item["total"]
                               for item in clusters], dtype=float)
        if len(positions) < 5:
            return None

        best = None
        for left in range(len(positions)-4):
            for right in range(left+5, min(len(positions), left+15)+1):
                selected = positions[left:right]
                if selected[-1]-selected[0] < 36.:
                    continue
                error = self._fret_spacing(selected)
                if error is None:
                    continue
                score = (len(selected), -error)
                if best is None or score > best[0]:
                    best = score, selected, error
        if best is None:
            return None
        _, selected, error = best
        return selected.copy(), error

    def _rectified_color(self, color, gray, transform):
        """Rectify optional camera color without converting the full frame."""
        cv2, np = self._cv2, self._np
        if (not isinstance(color, np.ndarray) or color.dtype != np.uint8
                or color.ndim != 3 or color.shape[:2] != gray.shape
                or color.shape[2] not in (3, 4)):
            return None
        rectified = cv2.warpPerspective(
            color, transform, (self._WIDTH, self._HEIGHT),
            flags=cv2.INTER_LINEAR)
        if rectified.shape[2] == 4:
            rectified = cv2.cvtColor(rectified, cv2.COLOR_RGBA2BGR)
        return rectified

    def _silver_fret_response(self, color, y0, y1):
        """Return narrow, spatially supported silver-wire evidence in the ROI."""
        cv2, np = self._cv2, self._np
        if (not isinstance(color, np.ndarray) or color.dtype != np.uint8
                or color.shape != (self._HEIGHT, self._WIDTH, 3)):
            return None
        hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
        saturation, value = hsv[:, :, 1], hsv[:, :, 2]
        band_value = value[y0:y1]
        if band_value.size == 0:
            return None
        # Thresholds adapt to exposure, but remain bounded so highly saturated
        # wood cannot silently turn into a metal candidate.
        saturation_limit = float(np.clip(
            np.percentile(saturation[y0:y1], 60) + 45, 95, 150))
        value_floor = float(np.clip(
            np.percentile(band_value, 20) + 12, 78, 125))
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        local = gray.astype(np.int16) - cv2.medianBlur(gray, 15)
        silver = ((saturation < saturation_limit) & (value > value_floor)
                  & (local >= 2)).astype(np.uint8)
        # Tolerate one-pixel perspective wobble while preserving narrow wires.
        band = cv2.dilate(silver, np.ones((1, 3), np.uint8))[y0:y1]
        support = band.mean(axis=0)
        parts = [part.mean(axis=0) for part in np.array_split(band, 4, axis=0)
                 if len(part)]
        spatial = np.median(np.stack(parts), axis=0)
        value_blur = cv2.GaussianBlur(value, (3, 3), 0)
        edge = np.abs(cv2.Sobel(
            value_blur, cv2.CV_32F, 1, 0, ksize=3))[y0:y1].mean(axis=0)
        edge_scale = max(30., min(120., float(np.percentile(edge, 90)) * .65))
        edge_support = np.clip(edge / edge_scale, 0., 1.)
        score = (.50 * np.clip(support / .55, 0., 1.)
                 + .20 * np.clip(spatial / .50, 0., 1.)
                 + .30 * edge_support)
        return {"support": support, "spatial": spatial,
                "edge": edge, "score": score}

    def _color_fret_fit(self, response):
        """Fit a complete equal-tempered run to color-supported peaks."""
        np = self._np
        candidates = self._peaks(response["score"] * 100., 3, 32, contrast=1.2)
        positions = []
        for x, strength in candidates:
            index = int(round(x))
            if (strength < 46. or response["edge"][index] < .5
                    or not 3 < x < self._WIDTH - 4):
                continue
            cutoff = max(.16, response["score"][index] * .55)
            left = index
            while left > 0 and response["score"][left - 1] >= cutoff:
                left -= 1
            right = index
            while right + 1 < self._WIDTH and response["score"][right + 1] >= cutoff:
                right += 1
            if right - left + 1 <= 9:
                positions.append(float(x))
        positions = np.array(sorted(positions))
        best = None
        for left in range(4):
            for right in range(4):
                selected = positions[left:len(positions)-right if right else None]
                error = self._fret_spacing(selected)
                if error is None:
                    continue
                indices = np.clip(np.rint(selected).astype(int), 0, self._WIDTH - 1)
                support = float(np.mean(response["score"][indices]))
                if best is None or (len(selected), support, -error) > (
                        len(best[0]), best[2], -best[1]):
                    best = selected, error, support
        return best

    def _refine_frets_with_silver(self, positions, response):
        """Snap geometric wire centers to nearby supported silver ridges."""
        np = self._np
        refined = []
        for position in positions:
            center = int(round(position))
            lo, hi = max(3, center - 2), min(self._WIDTH - 4, center + 3)
            if hi <= lo:
                refined.append(float(position))
                continue
            local = np.arange(lo, hi)
            scores = response["score"][local] - .025 * np.abs(local - position)
            index = int(local[int(np.argmax(scores))])
            refined.append(float(index) if
                response["score"][index] >= .40 and response["edge"][index] >= 7.
                else float(position))
        refined = np.asarray(refined, dtype=float)
        return (refined, self._fret_spacing(refined)) if len(refined) else (None, None)

    def _detect(self, gray, corners, color=None):
        self._dependencies()
        cv2, np = self._cv2, self._np
        if not isinstance(gray, np.ndarray) or gray.dtype != np.uint8 or gray.ndim != 2:
            return None
        height, width = gray.shape
        quad = np.float32([[p["x"] * (width - 1), p["y"] * (height - 1)] for p in corners])
        destination = np.float32([[0, 0], [self._WIDTH - 1, 0], [self._WIDTH - 1, self._HEIGHT - 1], [0, self._HEIGHT - 1]])
        transform = cv2.getPerspectiveTransform(quad, destination)
        pixel_started = time.perf_counter()
        if self._projection_backend is not None:
            rectified, gx, gy = self._projection_backend(gray, transform)
        else:
            rectified = cv2.warpPerspective(gray, transform, (self._WIDTH, self._HEIGHT), flags=cv2.INTER_LINEAR)
            blur = cv2.GaussianBlur(rectified, (3, 3), 0)
            gx = None
            gy = np.abs(cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3))
        self.timings["warp_gradient_ms"] = (time.perf_counter() - pixel_started) * 1000
        if self._projection_backend is not None:
            self.timings["roi_hls"] = dict(getattr(self._projection_backend, "timings", {}))
        if float(rectified.std()) < 9.0:
            return None
        sy = gy.mean(axis=1)
        strings = self._six_strings(self._peaks(sy, 4, 18))
        if strings is None:
            strings = self._ridge_strings(rectified)
        strings_estimated = strings is None
        if strings_estimated:
            # The tracked neck edges already define the visible six-string
            # band when specular highlights hide one or more wires.
            y_positions = np.linspace(self._HEIGHT/12., self._HEIGHT*11./12., 6)
            string_error = 1.
        else:
            _, y_positions, string_error = strings
        y0, y1 = max(2, int(y_positions[0]) - 3), min(self._HEIGHT - 2, int(y_positions[-1]) + 4)
        if strings_estimated:
            y0, y1 = 3, self._HEIGHT - 4
        if gx is None:
            fret_profile = np.abs(cv2.Sobel(blur[y0:y1], cv2.CV_32F, 1, 0, ksize=3)).mean(axis=0)
        else:
            # Interior rows have the same Sobel neighbourhood on PS and PL;
            # exclude two crop boundaries to avoid differing border policies.
            fret_profile = gx[y0+1:y1-1].mean(axis=0)
        gray_fit = None
        for contrast,merge in ((3.,3),(1.,4)):
            candidates = self._peaks(fret_profile,merge,32,contrast=contrast)
            floor = max(item[1] for item in candidates)*.20 if candidates else float("inf")
            x_positions = np.array(sorted(item[0] for item in candidates
                if item[1] >= floor and 3 < item[0] < self._WIDTH-4))
            # Dense visible frets can lift both the median and MAD above all
            # peaks. The lower-threshold retry still needs the physical scale
            # progression; background grids cannot pass by peak count alone.
            spacing = self._fret_spacing(x_positions)
            if spacing is not None:
                gray_fit = x_positions, spacing
                break
        if gray_fit is None:
            started = time.perf_counter()
            ridges = self._ridge_frets(rectified,y0,y1)
            self.timings["ridge_frets_ms"] = (time.perf_counter()-started)*1000
            if ridges is not None:
                gray_fit = ridges
        if gray_fit is None:
            started = time.perf_counter()
            hough = self._hough_frets(rectified,y0,y1)
            self.timings["hough_frets_ms"] = (time.perf_counter()-started)*1000
            if hough is not None:
                gray_fit = hough

        color_rectified = self._rectified_color(color, gray, transform)
        silver = None
        color_fit = None
        if color_rectified is not None:
            started = time.perf_counter()
            silver = self._silver_fret_response(color_rectified, y0, y1)
            if silver is not None:
                color_fit = self._color_fret_fit(silver)
                self.timings["silver_response_ms"] = (
                    time.perf_counter() - started) * 1000

        fit = gray_fit
        if fit is not None and silver is not None:
            refined, error = self._refine_frets_with_silver(fit[0], silver)
            if refined is not None:
                fit = refined, error
        # Color may recover a wire missed by grayscale, but only after the
        # complete sequence passes the same physical fret-spacing gate.
        if color_fit is not None and (fit is None or
                (len(color_fit[0]), color_fit[2]) > (len(fit[0]), 0.)):
            fit = color_fit[0], color_fit[1]
        if fit is None:
            return None
        x_positions, spacing = fit
        inverse = cv2.getPerspectiveTransform(destination, quad)

        raw = np.float32([((x_positions[0], y), (x_positions[-1], y)) for y in y_positions] +
                         [((x, y_positions[0]), (x, y_positions[-1])) for x in x_positions])
        mapped = cv2.perspectiveTransform(raw.reshape(-1, 1, 2), inverse).reshape(-1, 2, 2)
        mapped = mapped.astype(float)/np.array([width-1, height-1])
        lines = [[{"x": float(x), "y": float(y)} for x, y in ends] for ends in mapped]
        string_lines, fret_lines = lines[:6], lines[6:]
        confidence = float(min(1.0, .55 + .25 * (1 - string_error) + .20 * (1 - spacing / 2.4)))
        if silver is not None:
            indices = np.clip(np.rint(x_positions).astype(int), 0, self._WIDTH - 1)
            color_support = float(np.clip(
                np.mean(silver["score"][indices]), 0., 1.))
            self.timings["fret_color_support"] = color_support
            confidence = min(1.0, confidence + .06 * max(0., color_support - .35))
        self._last_strings_estimated = strings_estimated
        self.timings["string_observation"] = (
            "estimated_live_neck_width" if strings_estimated else "measured")
        return confidence, string_lines, fret_lines
