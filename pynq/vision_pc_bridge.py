"""PC RGBA capture -> PS packing -> real PL gray/Sobel -> JPEG response."""
import argparse
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import threading
import time
import zlib
import socket
import signal
import sys

# Tiny feature matrices should not start a BLAS thread pool on the dual-core PS.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENCV_IO_MAX_IMAGE_WIDTH", "640")
os.environ.setdefault("OPENCV_IO_MAX_IMAGE_HEIGHT", "480")
os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", "307200")

import cv2
import numpy as np

from vision_pl import VisionPL, VisionPLError
from roi_fretboard import RoiFretboardTracker
from auto_fretboard import AutoFretboardTracker
from neck_region import find_neck_regions, find_neck_regions_fast, native_backend
from visual_positions import project_positions, validate_note_calibration
from fingertip_notes import FingertipNotes
from fingertip_tracking import FourTipTracker
from av_fusion import publish_visual
from work_priority import WorkPriority
from hand_region import HandRegionTracker
from string_geometry import estimate_strings

FRAME_BYTES = 640 * 480 * 4
_JPEG_DECODER = None


class StreamBusy(RuntimeError):
    pass


class LatticeSearch:
    """One PS acquisition job; results require current-frame revalidation."""
    def __init__(self, detector=None, acquisition_mode="auto"):
        if acquisition_mode not in ("auto", "color_border"):
            raise ValueError("Unknown neck acquisition mode")
        self.acquisition_mode = acquisition_mode
        if acquisition_mode == "color_border":
            # OpenCV builds the Lab conversion tables on first use. Keep that
            # one-time cost out of the first camera acquisition job.
            cv2.cvtColor(np.zeros((1, 1, 3), np.uint8), cv2.COLOR_BGR2LAB)
        self.lock = threading.Lock()
        self.busy = False
        self.completed = None
        self.next_at = float('-inf')
        self.identity = None
        self.lattice_only = False
        self.stopping = False
        self.detector = detector
        self.model_fallback = detector is not None and acquisition_mode == "auto"
        self.model_busy = False
        self.model_next_at = float('-inf')
        self.fast_status = None
        self.fast_negative = False
        self.detector_status = {"status": "disabled", "reason": "no_model_configured"} if detector is None else detector.status()

    @staticmethod
    def select_proposal(proposals, shape):
        pixels = [np.float32([[p['x']*(shape[1]-1), p['y']*(shape[0]-1)] for p in q])
                  for q in proposals]
        extents = [np.linalg.norm(p[1]-p[0])+np.linalg.norm(p[2]-p[3]) for p in pixels]
        # Preserve the visible long extent, then avoid the expanded background
        # envelope when an equally long, observed tight border is available.
        eligible = [i for i,e in enumerate(extents) if e >= .95*max(extents)]
        return proposals[min(eligible, key=lambda i: cv2.contourArea(pixels[i]))]

    def submit(self, gray, edge, identity, can_start=None, lattice_only=False, bgr=None,
               geometry_only=False):
        with self.lock:
            if (self.stopping or self.busy or
                    identity == self.identity and lattice_only == self.lattice_only and
                    time.monotonic() < self.next_at or
                    can_start is not None and not can_start()):
                return False
            self.busy = True
            alternate = self.fast_negative if identity == self.identity else False
            self.identity = identity
            self.lattice_only = lattice_only
        source = gray.copy()
        support = None if edge is None else edge.copy()
        color = None if bgr is None else bgr.copy()

        def run():
            try:
                # Linux nice is per-thread. Detection yields CPU to live video
                # and audio; it never takes the PL inference/flow lock.
                if hasattr(os, 'setpriority') and hasattr(threading, 'get_native_id'):
                    try:
                        os.setpriority(os.PRIO_PROCESS, threading.get_native_id(),
                                       10 if lattice_only or geometry_only else 5)
                    except OSError:
                        pass
                detected = None
                proposals = []
                if not lattice_only and self.acquisition_mode == "color_border" and not geometry_only:
                    from neck_color import find_color_neck_regions, native_backend as color_native
                    started = time.perf_counter()
                    candidates = [] if color is None else find_color_neck_regions(color, limit=3)
                    with self.lock:
                        self.fast_status = dict(proposals=len(candidates),
                            backend='ps_color_region_contour_fit_native' if color_native() is not None
                                else 'ps_color_region_contour_fit_python',
                            processing_ms=(time.perf_counter()-started)*1000,
                            candidate=candidates[0] if candidates else None)
                    if candidates:
                        detected = dict(status='tracking', corners=candidates[0]['corners'],
                            alternatives=[candidate['corners'] for candidate in candidates[1:]],
                            acquisition='color_border_candidate', lattice_confirmed=False)
                    else:
                        detected = dict(status='searching',
                            reason='color_frame_required' if color is None else 'no_color_neck_candidate',
                            strings=[], frets=[], acquisition='color_border_candidate')
                elif not lattice_only and self.detector is not None and color is not None and not geometry_only:
                    # YOLO decides which image region is the fretboard. Geometry
                    # may refine that box but must not nominate a global candidate.
                    self.submit_model(source, color, identity, proposals)
                    detected = dict(status='searching', reason='yolo_detection_pending',
                                    strings=[], frets=[], acquisition='yolo_pending')
                elif not lattice_only and not geometry_only:
                    started = time.perf_counter()
                    # Long visible borders are enough to start LK tracking.
                    # String/fret lattice confirmation remains a later stage.
                    proposals = find_neck_regions_fast(source, alternate=alternate,
                                                       require_frets=False)
                    with self.lock:
                        self.fast_negative = not alternate if not proposals else False
                        self.fast_status = dict(proposals=len(proposals),
                            alternate_edges=alternate,
                            backend='ps_native_geometry' if native_backend() is not None else 'ps_python_geometry',
                            processing_ms=(time.perf_counter()-started)*1000)
                if geometry_only:
                    detected = dict(status='searching', reason=('color_frame_required' if
                                    self.acquisition_mode == 'color_border' else 'yolo_required_for_acquisition'),
                                    strings=[], frets=[])
                if proposals and detected is None:
                    # A short unoccluded patch can validate six strings while
                    # excluding the fretting hand. Acquire the longest observed
                    # neck proposal first; measure its lattice independently.
                    corners = self.select_proposal(proposals, source.shape)
                    detected = dict(status='tracking', corners=corners,
                                    acquisition='neck_border_candidate', lattice_confirmed=False)
                if detected is None:
                    detected = (AutoFretboardTracker().update(source, edge=support)
                                if lattice_only else
                                dict(status='searching', reason='no_fast_neck_proposal', strings=[], frets=[]))
                result = (identity, source, detected)
            except Exception as exc:
                with self.lock:
                    self.fast_status = dict(status='failed', reason=str(exc))
                result = None
            with self.lock:
                # A fast negative cannot erase a successful neural result.
                if (self.completed is None or result is not None and
                        result[2].get('status') == 'tracking'):
                    self.completed = result
                self.busy = False
                self.next_at = time.monotonic()+(3 if lattice_only else .15)

        threading.Thread(target=run,name='ps-lattice-search',daemon=True).start()
        return True

    def submit_model(self, source, color, identity, proposals):
        with self.lock:
            if self.stopping or self.model_busy or time.monotonic() < self.model_next_at:
                return False
            self.model_busy = True

        def run():
            try:
                if hasattr(os, 'setpriority') and hasattr(threading, 'get_native_id'):
                    try:
                        os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 15)
                    except OSError:
                        pass
                detected = self.detector.detect(color, proposals=proposals)
                with self.lock:
                    self.detector_status = self.detector.status()
                    if self.identity == identity:
                        if detected.get('status') == 'tracking':
                            result = detected
                        else:
                            result = dict(status='searching',
                                reason='yolo_' + str(detected.get('reason', detected.get('status', 'no_detection'))),
                                strings=[], frets=[], acquisition='yolo')
                        self.completed = (identity, source, result)
            except Exception as exc:
                with self.lock:
                    self.detector_status = dict(status='failed', reason=str(exc))
            finally:
                with self.lock:
                    self.model_busy = False
                    self.model_next_at = time.monotonic()+10

        threading.Thread(target=run, name='ps-neck-model', daemon=True).start()
        return True

    def take(self, identity):
        with self.lock:
            completed, self.completed = self.completed, None
        return completed if completed is not None and completed[0] == identity else None

    def status(self):
        with self.lock:
            status = dict(self.detector_status)
            status["busy"] = self.busy
            status["model_busy"] = self.model_busy
            status["model_primary_enabled"] = self.model_fallback
            status["model_fallback_enabled"] = False
            status["acquisition_mode"] = ("color_border_then_flow" if self.acquisition_mode == "color_border"
                else "yolo_first_then_flow" if self.detector is not None else "bounded_geometry_then_flow")
            status["fast_acquisition"] = self.fast_status
            return status


def decode_frame(body, encoding="identity", expected_bytes=FRAME_BYTES):
    if encoding == 'jpeg':
        if expected_bytes != FRAME_BYTES or not 4 <= len(body) <= FRAME_BYTES or not body.startswith(b'\xff\xd8'):
            raise ValueError('Invalid JPEG frame')
        if _JPEG_DECODER is not None:
            return _JPEG_DECODER(body).tobytes()
        try:
            bgr = cv2.imdecode(np.frombuffer(body,np.uint8),cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        except cv2.error as exc:
            raise ValueError('Invalid JPEG frame') from exc
        if bgr is None or bgr.shape != (480,640,3):
            raise ValueError('JPEG must decode to 640x480')
        return cv2.cvtColor(bgr,cv2.COLOR_BGR2RGBA).tobytes()
    if encoding == "identity":
        if len(body) != expected_bytes:
            raise ValueError("Expected fixed RGBA frame")
        return body
    if encoding != "deflate" or not 0 < len(body) <= FRAME_BYTES:
        raise ValueError("Invalid frame encoding or length")
    decoder = zlib.decompressobj()
    try:
        raw = decoder.decompress(body, expected_bytes + 1)
    except zlib.error as exc:
        raise ValueError("Invalid compressed frame") from exc
    if (len(raw) != expected_bytes or not decoder.eof or decoder.unused_data
            or decoder.unconsumed_tail):
        raise ValueError("Invalid decompressed frame length or trailing data")
    return raw


def analysis_header(analysis):
    """Keep optional JPEG thumbnails below the proxy's HTTP header budget."""
    encoded = json.dumps(analysis, separators=(",", ":"), allow_nan=False)
    if len(encoded) <= 14 * 1024:
        return encoded
    compact = dict(analysis)
    neural = compact.get('fingertip_neural')
    if isinstance(neural, dict):
        neural = dict(neural)
        neural.pop('preview', None)
        if isinstance(neural.get('last_result'), dict):
            neural['last_result'] = dict(neural['last_result'])
            neural['last_result'].pop('preview', None)
        compact['fingertip_neural'] = neural
    return json.dumps(compact, separators=(",", ":"), allow_nan=False)


class FrameTracker:
    def __init__(self, flow_backend=None, projection_backend=None, fast_tracking=False, search_worker=None,
                 discovery_allowed=None, recovery_worker=None):
        self.fast_tracking = fast_tracking
        self.flow_backend = flow_backend
        self.projection_backend = projection_backend
        self.tracker = None
        self.identity = None
        self.last_sequence = -1
        self.last_gray_at = None
        self.streak = 0
        self.last_lattice = None
        self.last_positions = None
        self.last_search_at = None
        self.lattice_epoch = 0
        self.unsupported_since = None
        self.last_recovery_at = float('-inf')
        self.seed_source = None
        self.last_acquisition = None
        self.neck_reference = None
        self.last_relocalization_at = float('-inf')
        self.last_extent_at = float('-inf')
        self.extent_interval = 2.
        # Standalone users still acquire automatically, but full discovery
        # always belongs to this bounded background worker.
        self.search_worker = LatticeSearch() if search_worker is None else search_worker
        self.recovery_worker = self.search_worker if recovery_worker is None else recovery_worker
        self.discovery_allowed = discovery_allowed

    def full_search(self, gray, edge, lattice_only=False, bgr=None):
        worker = self.recovery_worker if lattice_only else self.search_worker
        completed = worker.take(self.identity)
        if completed is not None:
            return completed[1:]
        if lattice_only:
            submitted = worker.submit(gray, edge, self.identity,
                self.discovery_allowed, lattice_only=True)
        else:
            if bgr is None:
                submitted = self.search_worker.submit(gray, edge, self.identity, self.discovery_allowed)
            else:
                submitted = self.search_worker.submit(gray, edge, self.identity,
                    self.discovery_allowed, bgr=bgr)
        return gray, {'status':'searching',
                      'reason':'background_lattice_search' if submitted else 'background_lattice_deferred',
                      'strings':[],'frets':[]}

    def initialize_search_result(self, source, gray, detected, color=None):
        proposed = RoiFretboardTracker(flow_backend=self.flow_backend,
            projection_backend=self.projection_backend, fast_tracking=self.fast_tracking,
            async_measurements=self.fast_tracking)
        options = [detected['corners']] + detected.get('alternatives', [])[:2]
        for corners in options:
            if source is not gray:
                corners = self.align_candidate(source, gray, corners)
                if corners is None:
                    result = proposed.invalidate('acquisition_frame_mismatch')
                    continue
            result = proposed.initialize(gray, corners, color)
            if result.get('status') == 'tracking':
                break
        self.last_acquisition = dict(status=result.get('roi_status'), reason=result.get('reason'),
                                     source=self.seed_source_for(detected))
        return proposed, result

    @staticmethod
    def seed_source_for(detected, manual=False):
        if manual:
            return "manual"
        source = detected.get("acquisition") if isinstance(detected, dict) else None
        return source if source in ("neck_model_geometry_refined", "neck_model_geometry_gated", "neck_border_candidate", "color_border_candidate") else "rulesfallback"

    @staticmethod
    def align_candidate(source, current, corners, prefer_features=False):
        """Revalidate a delayed detector proposal against the current image.

        Acquisition can take seconds. Adjacent-frame LK must not jump directly
        over that gap. Bounded template matching handles small translation;
        rotation/scene changes that do not match require fresh acquisition.
        """
        if not isinstance(corners, (list, tuple)) or len(corners) != 4:
            return None
        if prefer_features:
            aligned = FrameTracker.align_features(source, current, corners)
            if aligned is not None:
                return aligned
        size = (160, 120)
        first = cv2.resize(source, size, interpolation=cv2.INTER_AREA)
        second = cv2.resize(current, size, interpolation=cv2.INTER_AREA)
        quad = np.float32([[p['x'],p['y']] for p in corners])
        if not np.isfinite(quad).all() or (quad<0).any() or (quad>1).any():
            return None
        pixels = quad*np.array([159,119],np.float32)
        low = np.maximum(0,np.floor(pixels.min(axis=0)).astype(int))
        high = np.minimum([160,120],np.ceil(pixels.max(axis=0)).astype(int)+1)
        x,y = low
        right,bottom = high
        template = first[y:bottom,x:right]
        if min(template.shape)<8 or np.std(template)<5:
            return None
        search_low = np.maximum(0,low-12)
        search_high = np.minimum([160,120],high+12)
        sx,sy = search_low
        ex,ey = search_high
        response = cv2.matchTemplate(second[sy:ey,sx:ex],template,cv2.TM_CCOEFF_NORMED)
        _,score,_,position = cv2.minMaxLoc(response)
        if not np.isfinite(score) or score<.82:
            return None if prefer_features else FrameTracker.align_features(source, current, corners)
        displacement = (np.array(position)+search_low-low)/np.array([159,119])
        mapped = quad+displacement
        if not np.isfinite(mapped).all() or (mapped<0).any() or (mapped>1).any():
            return None
        return [dict(x=float(x),y=float(y)) for x,y in mapped]

    @staticmethod
    def align_features(source, current, corners):
        """Bounded ORB/RANSAC relocalization for rotated delayed detections."""
        size = (320, 240)
        first = cv2.resize(source, size, interpolation=cv2.INTER_AREA)
        second = cv2.resize(current, size, interpolation=cv2.INTER_AREA)
        quad = np.float32([[p['x']*319, p['y']*239] for p in corners])
        mask = np.zeros(first.shape, np.uint8)
        cv2.fillConvexPoly(mask, np.rint(quad).astype(np.int32), 255)
        orb = cv2.ORB_create(nfeatures=400, edgeThreshold=8, fastThreshold=12)
        old, old_desc = orb.detectAndCompute(first, mask)
        new, new_desc = orb.detectAndCompute(second, None)
        if old_desc is None or new_desc is None or len(new_desc)<2:
            return None
        matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(old_desc, new_desc, k=2)
        good = [a for pair in matches if len(pair)==2 for a,b in [pair]
                if a.distance < .72*b.distance]
        if len(good)<12:
            return None
        a = np.float32([old[m.queryIdx].pt for m in good])
        b = np.float32([new[m.trainIdx].pt for m in good])
        transform, valid = cv2.findHomography(a, b, cv2.RANSAC, 2.)
        if transform is None or valid is None:
            return None
        keep = valid.ravel().astype(bool)
        if keep.sum()<10 or keep.mean()<.65:
            return None
        if cv2.contourArea(cv2.convexHull(a[keep])) < .18*cv2.contourArea(quad):
            return None
        mapped = cv2.perspectiveTransform(quad.reshape(-1,1,2),transform)[:,0]
        if (not np.isfinite(mapped).all() or not cv2.isContourConvex(mapped) or
                (mapped<0).any() or (mapped>(319,239)).any() or
                not .6<cv2.contourArea(mapped)/cv2.contourArea(quad)<1.65):
            return None
        return [dict(x=float(x/319),y=float(y/239)) for x,y in mapped]

    @staticmethod
    def align_motion(source, current, corners):
        """Relocalize a translated neck after adjacent-frame flow breaks.

        A masked, whole-frame template search handles displacements well beyond
        the conservative LK motion gate. The candidate is accepted only after
        a full-resolution appearance correlation over the entire board mask.
        """
        if (not isinstance(corners, (list, tuple)) or len(corners) != 4
                or source is None or current is None
                or source.shape != current.shape):
            return None
        height, width = source.shape
        if min(height, width) < 24:
            return None
        work_width = 160
        work_height = max(24, int(round(work_width * height / width)))
        first = cv2.resize(source, (work_width, work_height),
                           interpolation=cv2.INTER_AREA)
        second = cv2.resize(current, (work_width, work_height),
                            interpolation=cv2.INTER_AREA)
        quad = np.float32([[p["x"] * (work_width - 1),
                            p["y"] * (work_height - 1)] for p in corners])
        if (not np.isfinite(quad).all() or (quad < 0).any()
                or (quad > (work_width - 1, work_height - 1)).any()):
            return None
        x, y, box_width, box_height = cv2.boundingRect(
            np.rint(quad).astype(np.int32))
        right = min(work_width, x + box_width)
        bottom = min(work_height, y + box_height)
        x, y = max(0, x), max(0, y)
        template = first[y:bottom, x:right]
        mask = np.zeros((work_height, work_width), np.uint8)
        cv2.fillConvexPoly(mask, np.rint(quad).astype(np.int32), 255)
        template_mask = mask[y:bottom, x:right]
        if (min(template.shape) < 8 or template_mask.sum() < 120
                or template.shape[0] >= work_height
                or template.shape[1] >= work_width):
            return None
        response = cv2.matchTemplate(second, template, cv2.TM_CCORR_NORMED,
                                     mask=template_mask)
        columns = np.arange(response.shape[1], dtype=np.float32)[None, :, None]
        rows = np.arange(response.shape[0], dtype=np.float32)[:, None, None]
        mapped_x = quad[None, None, :, 0] + columns - x
        mapped_y = quad[None, None, :, 1] + rows - y
        valid = ((mapped_x >= 0).all(axis=2) & (mapped_x <= work_width - 1).all(axis=2)
                 & (mapped_y >= 0).all(axis=2) & (mapped_y <= work_height - 1).all(axis=2))
        if not valid.any():
            return None
        scores = np.where(valid, response, -1.)
        _, score, _, position = cv2.minMaxLoc(scores)
        if not np.isfinite(score) or score < .90:
            return None
        candidate = quad + np.float32((position[0] - x, position[1] - y))
        normalized = candidate / (work_width - 1, work_height - 1)

        # Repeated fret/string patterns can win a coarse translation search.
        # Compare the complete interior board appearance at source resolution.
        full_quad = np.float32([[p["x"] * (width - 1), p["y"] * (height - 1)]
                                for p in corners])
        mapped_quad = np.float32(normalized * (width - 1, height - 1))
        transform = cv2.getPerspectiveTransform(full_quad, mapped_quad)
        mapped_source = cv2.warpPerspective(source, transform, (width, height))
        validation_mask = np.zeros((height, width), np.uint8)
        cv2.fillConvexPoly(validation_mask, np.rint(full_quad).astype(np.int32), 255)
        kernel = max(5, int(round(min(height, width) * .012)) | 1)
        validation_mask = cv2.erode(
            validation_mask, np.ones((kernel, kernel), np.uint8))
        selected = validation_mask != 0
        if selected.sum() < 150:
            return None
        first_values = mapped_source[selected].astype(float)
        second_values = current[selected].astype(float)
        first_values -= first_values.mean()
        second_values -= second_values.mean()
        denominator = (np.linalg.norm(first_values)
                       * np.linalg.norm(second_values) + 1e-9)
        correlation = float(np.dot(first_values, second_values) / denominator)
        if not np.isfinite(correlation) or correlation < .82:
            return None
        return [dict(x=float(x), y=float(y)) for x, y in normalized]

    def discover(self, gray, edge, bgr=None, color=None):
        now = time.monotonic()
        # The worker bounds submissions; consume a finished candidate next frame.
        self.last_search_at = now
        source, detected = self.full_search(gray, edge, bgr=bgr)
        if detected["status"] != "tracking":
            return detected
        # Initialize from a structurally validated lattice, then require the
        # current rectified lattice to agree on the next LK frame below.
        # This avoids a second expensive full-frame Hough acquisition.
        proposed, result = self.initialize_search_result(
            source,gray,detected,color=color)
        if result.get("roi_status") == "tracking":
            self.tracker = proposed
            self.seed_source = self.seed_source_for(detected)
        return result

    def relocalize_recent(self, gray, force=False, color=None):
        reference = self.neck_reference
        now = time.monotonic()
        if reference is None or now-reference[0] > 3:
            return None
        if cv2.resize(gray,(160,120)).std() < 5:
            return None
        if not force and now-self.last_relocalization_at < .25:
            return None
        self.last_relocalization_at = now
        corners = self.align_motion(reference[1], gray, reference[2])
        backend = 'masked_motion_template'
        if corners is None:
            corners = self.align_candidate(
                reference[1], gray, reference[2], prefer_features=True)
            backend = 'orb_feature_fallback'
        if corners is None:
            return None
        proposed, result = self.initialize_search_result(gray,gray,
            dict(corners=corners,acquisition=reference[3]),color=color)
        if result.get('roi_status') != 'tracking':
            return None
        self.tracker, self.seed_source = proposed, reference[3]
        return dict(result, recovery='verified_recent_neck',
                    recovery_detail=backend)

    def recover_lattice(self, gray, edge, result, now):
        if result.get('status') == 'tracking' or result.get('roi_status') != 'tracking':
            self.unsupported_since = None
            return result
        if self.unsupported_since is None:
            self.unsupported_since = now
        if self.fast_tracking and os.environ.get('FRETLENS_FULL_LATTICE_RECOVERY', '0') != '1':
            return (result if result.get('recovery') is not None
                    else dict(result, recovery='local_lattice_measurement'))
        if now-self.unsupported_since < 1.5 or now-self.last_recovery_at < 6:
            return result
        self.last_recovery_at = now
        # A border proposal can follow a small, incorrect patch indefinitely.
        # Periodically retry the independent lattice detector, without replacing
        # a live region unless the replacement also validates six strings/frets.
        source, detected = self.full_search(gray, edge, lattice_only=True)
        if detected.get('status') != 'tracking':
            return dict(result, recovery='full_lattice_search_no_match')
        proposed, replacement = self.initialize_search_result(source,gray,detected)
        if replacement.get('status') == 'tracking':
            def extent(corners):
                points = np.float32([[p['x'],p['y']] for p in corners])
                return np.linalg.norm(points[1]-points[0])+np.linalg.norm(points[2]-points[3])
            if extent(replacement['roi_corners']) < .85*extent(result['roi_corners']):
                return dict(result,recovery='short_lattice_patch_rejected')
            self.tracker = proposed
            self.seed_source = self.seed_source_for(detected)
            self.unsupported_since = None
            return dict(replacement, recovery='validated_full_lattice')
        return dict(result, recovery='full_lattice_not_confirmed')

    def refresh_neck_extent(self, gray, edge, result, now):
        """Extend a live short lock only with overlapping observed neck borders."""
        if result.get('roi_status') != 'tracking' or self.seed_source == 'manual':
            return result
        completed = self.search_worker.take(self.identity)
        if completed is not None:
            extended = False
            source, detected = completed[1:]
            if detected.get('status') == 'tracking':
                corners = self.align_candidate(source, gray, detected['corners'])
                if corners is not None:
                    old = np.float32([[p['x'],p['y']] for p in result['roi_corners']])
                    new = np.float32([[p['x'],p['y']] for p in corners])
                    pixel_scale = np.float32([gray.shape[1]-1, gray.shape[0]-1])
                    old, new = old*pixel_scale, new*pixel_scale
                    a = (old[1]+old[2]-old[0]-old[3])*.5
                    b = (new[1]+new[2]-new[0]-new[3])*.5
                    length, next_length = np.linalg.norm(a), np.linalg.norm(b)
                    overlap, _ = cv2.intersectConvexConvex(old,new)
                    area, next_area = cv2.contourArea(old), cv2.contourArea(new)
                    if (length > 1 and next_length >= length*1.18 and
                            abs(np.dot(a,b))/(length*next_length) >= .985 and
                            area > 1 and overlap >= area*.8 and
                            .65 <= (next_area/next_length)/(area/length) <= 1.6):
                        proposed, replacement = self.initialize_search_result(gray,gray,
                            dict(detected,corners=corners))
                        if replacement.get('roi_status') == 'tracking':
                            self.tracker = proposed
                            self.seed_source = self.seed_source_for(detected)
                            result = dict(replacement, coverage='observed_neck_extended')
                            extended = True
            # A validated live ROI keeps following every frame. Repeated
            # full-image extent searches need not compete with that flow.
            self.extent_interval = 2. if extended else min(8., self.extent_interval*2.)
        if now-self.last_extent_at >= self.extent_interval:
            if self.search_worker.submit(gray,edge,self.identity,self.discovery_allowed,geometry_only=True):
                self.last_extent_at = now
        return result

    @staticmethod
    def validate_options(options):
        if options is None:
            return
        if not isinstance(options, dict):
            raise ValueError("Invalid tracking options")
        stream, sequence, generation = (options.get(key) for key in ("stream", "sequence", "generation"))
        if (not isinstance(stream, str) or not 1 <= len(stream) <= 80
                or not all(c.isascii() and (c.isalnum() or c == "-") for c in stream)
                or type(sequence) is not int or not 0 <= sequence < 2 ** 53
                or type(generation) is not int or not 0 <= generation < 2 ** 31):
            raise ValueError("Invalid stream/sequence/generation")
        owner = options.get("owner")
        if owner is not None and (not isinstance(owner, str) or not 8 <= len(owner) <= 64
                or not all(c.isascii() and (c.isalnum() or c == "-") for c in owner)):
            raise ValueError("Invalid stream owner")
        if type(options.get("enabled")) is not bool:
            raise ValueError("Expected boolean enabled")
        validate_note_calibration(options.get('note_calibration'))
        for key in ("fast_tracking", "coordinates_only"):
            if key in options and type(options[key]) is not bool:
                raise ValueError("Expected boolean " + key)
        roi = options.get("transport_roi")
        if roi is not None:
            if (not isinstance(roi, dict) or set(roi) != {"x","y","width","height"}
                    or any(type(v) is not int for v in roi.values())
                    or not 0 <= roi["x"] < 640 or not 0 <= roi["y"] < 480
                    or not 1 <= roi["width"] <= 640-roi["x"]
                    or not 1 <= roi["height"] <= 480-roi["y"]):
                raise ValueError("Invalid transport ROI")
        if type(options.get("preview_scale", 1)) is not int or options.get("preview_scale", 1) not in (1, 2):
            raise ValueError("Invalid preview scale")
        if options.get('preview_encoding','jpeg') not in ('jpeg','planes'):
            raise ValueError('Invalid preview encoding')
        corners = options.get("corners")
        if corners is not None:
            if not isinstance(corners, list) or len(corners) != 4:
                raise ValueError("Expected four ROI corners")
            for point in corners:
                if not isinstance(point, dict) or any(type(point.get(key)) not in (int, float)
                        or not np.isfinite(point[key]) or not 0 <= point[key] <= 1 for key in ("x", "y")):
                    raise ValueError("Invalid ROI corners")

    def track(self, gray, options, edge=None, rgba=None):
        if options is None:
            return None
        identity = options["stream"], options["generation"]
        if self.identity != identity:
            self.identity, self.tracker = identity, None
            self.last_sequence, self.last_gray_at = -1, None
            self.streak, self.last_lattice = 0, None
            self.last_positions = None
            self.last_search_at = None
            self.unsupported_since = None
            self.last_recovery_at = float('-inf')
            self.seed_source = None
            self.last_acquisition = None
            self.neck_reference = None
            self.last_relocalization_at = float('-inf')
            self.last_extent_at = float('-inf')
            self.extent_interval = 2.
        if options["sequence"] <= self.last_sequence:
            raise ValueError("Frame sequence did not advance")
        self.last_sequence = options["sequence"]
        if not options["enabled"]:
            self.tracker = None
            self.last_search_at = None
            self.seed_source = None
            self.neck_reference = None
            self.last_relocalization_at = float('-inf')
            return {"status": "disabled", "strings": [], "frets": []}
        if self.tracker is None:
            if options.get("transport_roi") is not None:
                result = {"status":"searching", "reason":"full_frame_required", "strings":[], "frets":[], "roi_status":"lost"}
            elif options.get("corners") is None:
                result = self.relocalize_recent(
                    gray,color=rgba) if self.fast_tracking else None
                if result is None:
                    bgr = None if rgba is None else cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
                    result = self.discover(gray, edge, bgr=bgr, color=rgba)
            else:
                self.tracker = RoiFretboardTracker(flow_backend=self.flow_backend, projection_backend=self.projection_backend, fast_tracking=self.fast_tracking,
                                                 async_measurements=self.fast_tracking)
                result = self.tracker.initialize(
                    gray, options["corners"], color=rgba)
                if result.get("roi_status") == "tracking":
                    self.seed_source = "manual"
        elif self.last_gray_at is not None and time.monotonic() - self.last_gray_at > 2.0:
            result = self.tracker.invalidate("frame_gap")
        else:
            result = self.tracker.update(gray, color=rgba)
            if (self.fast_tracking and result.get('roi_status') == 'lost'
                    and options.get('corners') is None
                    and options.get('transport_roi') is None):
                recovered = (self.relocalize_recent(
                    gray, force=True, color=rgba) if rgba is not None
                    else self.relocalize_recent(gray, force=True))
                if recovered is not None:
                    result = recovered
        if options.get('corners') is None and options.get('transport_roi') is None:
            if self.fast_tracking:
                result = self.refresh_neck_extent(gray,edge,result,time.monotonic())
            result = self.recover_lattice(gray, edge, result, time.monotonic())
        if result.get("roi_status") == "lost" and options.get("corners") is None:
            self.tracker = None
            self.last_search_at = None
            self.seed_source = None
        self.last_gray_at = time.monotonic()
        if self.fast_tracking and result.get('roi_status') == 'tracking':
            self.neck_reference = (self.last_gray_at, gray.copy(), result['roi_corners'], self.seed_source)
        # Geometry must agree across consecutive current frames. Never keep
        # previous lines alive after a missing/occluded lattice.
        if result["status"] == "tracking":
            current = (len(result["strings"]), len(result["frets"]))
            quad = np.float32([[p["x"], p["y"]] for p in result["roi_corners"]])
            transform = cv2.getPerspectiveTransform(quad, np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]))
            lines = result["strings"] + result["frets"]
            endpoints = np.float32([[line[key]["x"], line[key]["y"]]
                                   for line in lines for key in ("start", "end")]).reshape(-1, 1, 2)
            rectified = cv2.perspectiveTransform(endpoints, transform).reshape(-1, 2, 2).mean(axis=1)
            positions = np.concatenate((rectified[:6, 1], rectified[6:, 0]))
            consistent = (current == self.last_lattice and self.last_positions is not None
                          and np.max(np.abs(positions - self.last_positions)) <= .035)
            if not consistent:
                self.lattice_epoch += 1
            self.streak = self.streak + 1 if consistent else 1
            self.last_lattice = current
            self.last_positions = positions
            if self.streak < 2:
                result = dict(result, status="searching", reason="confirming_lattice",
                              strings=[], frets=[], corners=None)
        else:
            if self.last_lattice is not None:
                self.lattice_epoch += 1
            self.streak, self.last_lattice = 0, None
            self.last_positions = None
        string_model = (estimate_strings(result.get('roi_corners'))
                        if result.get('roi_status') == 'tracking' else None)
        if string_model is not None:
            string_model.update(stream_id=options['stream'], generation=options['generation'],
                                sequence=options['sequence'])
        return dict(result, estimated_strings=string_model,
                    stream_id=options["stream"], sequence=options["sequence"],
                    generation=options["generation"], stable_frames=self.streak,
                    lattice_epoch=self.lattice_epoch,
                    acquisition_attempt=self.last_acquisition,
                    processing_location="ps_linux", flow_backend=(self.flow_backend.backend_name if self.flow_backend
                        else "opencv_pyramidal_lk_fb_ransac"),
                    stages_ms=self.tracker.timings if self.tracker is not None else {},
                    roi_backend="pl_warp_gaussian_sobel" if self.projection_backend else "ps_opencv",
                    acquisition=(self.seed_source or
                                 ("manual" if options.get("corners") is not None else "rulesfallback")),
                    discovery_backend=(self.seed_source or
                                       ("manual" if options.get("corners") is not None else "rulesfallback")),
                    preprocessing_backend="pl", absolute_frets_known=False)


class Bridge:
    validate_options = staticmethod(FrameTracker.validate_options)

    def __init__(self, processor, flow_backend=None, projection_backend=None, neck_detector=None):
        self.processor = processor
        self.flow_backend = flow_backend
        self.projection_backend = projection_backend
        self.lock = threading.Lock()
        self.fault = None
        self.sessions = OrderedDict()
        self.analysis = None
        self.hand_pipeline = None
        self.hand_region_tracker = HandRegionTracker(flow_backend, target_side="right")
        self.neural = None
        self.fingertip_worker = None
        self.fingertip_tracker = self.create_fingertip_tracker(flow_backend)
        self.fingertip_notes = FingertipNotes()
        self.stopping = False
        self.active_stream = None
        self.active_at = float('-inf')
        self.last_analysis = None
        self.last_processed_at = None
        self.search_worker = LatticeSearch(neck_detector,
            acquisition_mode=os.environ.get('FRETLENS_NECK_ACQUISITION', 'auto'))
        self.recovery_worker = LatticeSearch()
        self.work_priority = WorkPriority()

    @staticmethod
    def create_fingertip_tracker(flow_backend):
        quality = None
        if os.environ.get('FRETLENS_TIP_NATIVE_PATCH', '1') != '0':
            from pathlib import Path
            if Path(__file__).with_name('libfingertip_patch_quality.so').is_file():
                from fingertip_tracking_native import NativePatchQuality
                quality = NativePatchQuality()
        return FourTipTracker(flow_backend,
            fast_catchup=os.environ.get('FRETLENS_TIP_FAST_CATCHUP', '1') != '0',
            patch_quality=quality)

    def allow_background_discovery(self):
        return self.work_priority.allow_background_discovery(self.hand_pipeline)

    def update_fingertips(self, rgba, options, gray=None):
        if self.fingertip_worker is None:
            return
        identity = (options['stream'], options['generation'])
        region = self.analysis.get('hand_region', {})
        if region.get('status') == 'tracking' and region.get('regions'):
            corners = region['regions'][0].get('corners', [])
            if len(corners) == 4:
                points = np.asarray([[p['x'], p['y']] for p in corners], np.float64)
                if np.isfinite(points).all() and np.all((points >= 0) & (points <= 1)):
                    height, width = rgba.shape[:2]
                    lo = np.floor(points.min(axis=0)*[width,height]).astype(int)
                    hi = np.ceil(points.max(axis=0)*[width,height]).astype(int)
                    x, y = lo
                    right, bottom = hi
                    transport = options.get('transport_roi')
                    covered = transport is None or (x >= transport['x'] and y >= transport['y']
                        and right <= transport['x']+transport['width']
                        and bottom <= transport['y']+transport['height'])
                    if covered and right > x and bottom > y:
                        self.fingertip_worker.submit(rgba, (int(x),int(y),int(right-x),int(bottom-y)),
                            identity[0], identity[1], options['sequence'])
        self.analysis['fingertip_neural'] = self.fingertip_worker.snapshot(
            identity[0], identity[1], options['sequence'])
        if gray is not None:
            self.analysis['fingertip_tracking'] = self.fingertip_tracker.update(
                gray, self.analysis['fingertip_neural'], options, region)

    def check_stream(self, options):
        if options is None:
            return
        stream = (options['stream'], options['generation'], options.get('owner'))
        if (self.active_stream is not None and stream != self.active_stream and
                (time.monotonic()-self.active_at < 3 or self.lock.locked()) and options['sequence'] != 1):
            raise StreamBusy('Another camera stream owns PL; stop or reload this camera tab')

    def process(self, body, options=None):
        if self.stopping:
            raise RuntimeError('Board vision is stopping')
        self.validate_options(options)
        self.check_stream(options)
        roi = options.get("transport_roi") if options else None
        expected = roi["width"]*roi["height"]*4 if roi else FRAME_BYTES
        if len(body) != expected:
            raise ValueError("Expected 640x480 RGBA bytes")
        if self.fault:
            raise VisionPLError(self.fault)
        if options is not None:
            self.active_stream = (options['stream'], options['generation'], options.get('owner'))
            self.active_at = time.monotonic()
        started = time.monotonic()
        if roi:
            rgba = np.zeros((480,640,4),np.uint8)
            rgba[roi["y"]:roi["y"]+roi["height"],roi["x"]:roi["x"]+roi["width"]] = np.frombuffer(body,np.uint8).reshape(roi["height"],roi["width"],4)
        else:
            rgba = np.frombuffer(body, np.uint8).reshape(480, 640, 4)
        direct_rgba = callable(getattr(type(self.processor), "process_rgba", None))
        frame = rgba if direct_rgba else cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        converted = time.monotonic()
        try:
            selected = {"outputs": ("gray", "edge")} if isinstance(self.processor, VisionPL) else {}
            result = (self.processor.process_rgba(frame, **selected) if direct_rgba
                      else self.processor.process(frame, **selected))
        except VisionPLError as exc:
            self.fault = str(exc)
            raise
        tracking_started = time.monotonic()
        self.analysis = None
        if options is not None:
            key = options["stream"]
            if key not in self.sessions:
                if len(self.sessions) >= 4:
                    self.sessions.popitem(last=False)
                self.sessions[key] = FrameTracker(flow_backend=self.flow_backend, projection_backend=self.projection_backend,
                    fast_tracking=options.get("fast_tracking", False),search_worker=self.search_worker,
                    discovery_allowed=self.allow_background_discovery, recovery_worker=self.recovery_worker)
            self.sessions.move_to_end(key)
            try:
                self.analysis = self.sessions[key].track(result["gray"], options, result["edge"], rgba=rgba)
            except (OSError, RuntimeError) as exc:
                if any(getattr(core, "fault", False)
                       for core in (self.flow_backend, self.projection_backend)):
                    self.fault = str(exc)
                raise
        tracked = time.monotonic()
        if options is not None and self.analysis is not None:
            self.fingertip_notes.remember(self.analysis, options.get('note_calibration'))
            self.analysis['hand_region'] = self.hand_region_tracker.update(
                rgba, result['gray'], options, self.analysis)
            try:
                self.update_fingertips(rgba, options, result['gray'])
            except (OSError, RuntimeError) as exc:
                if getattr(self.flow_backend, 'fault', False):
                    self.fault = str(exc)
                raise
        if self.hand_pipeline is not None and options is not None and self.analysis is not None:
            try:
                self.analysis['hand_detection'] = self.hand_pipeline.update(rgba, result['gray'], options)
                self.work_priority.observe_hand_update(self.analysis['hand_detection'])
            except (OSError, RuntimeError) as exc:
                if getattr(self.flow_backend, 'fault', False):
                    self.fault = str(exc)
                raise
        hand_done = time.monotonic()
        if self.analysis is not None:
            self.analysis['fingertip_notes'] = self.fingertip_notes.project(
                self.analysis.get('fingertip_neural'), self.analysis,
                tracking=self.analysis.get('fingertip_tracking'))
            if self.analysis['fingertip_notes']['status'] == 'current':
                self.analysis['visual_positions'] = self.analysis['fingertip_notes']['projection']
            else:
                self.analysis['visual_positions'] = project_positions(self.analysis)
            self.analysis['processing_completed_monotonic'] = time.monotonic()
            publish_visual(os.environ.get('FRETLENS_VISUAL_STATE'),self.analysis,started)
            self.analysis["analysis_ms"] = (time.monotonic() - tracking_started) * 1000
            self.analysis["preprocess_ms"] = result.get("timings_ms", {})
            self.last_analysis = self.analysis
            self.last_processed_at = time.monotonic()
        projection_done = time.monotonic()
        if options is not None and options.get("coordinates_only"):
            self.analysis["bridge_stages_ms"] = {"input_convert_ms": (converted-started)*1000,
                "preprocess_ms": (tracking_started-converted)*1000,
                "tracking_ms": (tracked-tracking_started)*1000,
                "hand_tracking_ms": (hand_done-tracked)*1000,
                "visual_projection_ms": (projection_done-hand_done)*1000, "preview_encode_ms": 0.0}
            return b"", (time.monotonic()-started)*1000
        if options is not None and options.get('preview_encoding') == 'planes':
            scale = options.get('preview_scale',1)
            # Nearest sampling preserves the [::2, ::2] display contract,
            # without Python's slow strided bytes iteration on the PS.
            planes = b''.join((cv2.resize(result[name], (640//scale,480//scale),
                interpolation=cv2.INTER_NEAREST) if scale != 1 else result[name]).tobytes()
                for name in ('gray','edge'))
            if self.analysis is not None:
                self.analysis['bridge_stages_ms'] = dict(input_convert_ms=(converted-started)*1000,
                    preprocess_ms=(tracking_started-converted)*1000,
                    tracking_ms=(tracked-tracking_started)*1000,
                    hand_tracking_ms=(hand_done-tracked)*1000,
                    visual_projection_ms=(projection_done-hand_done)*1000,
                    preview_encode_ms=(time.monotonic()-projection_done)*1000)
            return planes,(time.monotonic()-started)*1000
        image = np.concatenate((result["gray"], result["edge"]), axis=1)
        if options is not None and options.get("preview_scale") == 2:
            # Diagnostic planes only. Tracking always consumes full PL output.
            image = cv2.resize(image, (640, 240), interpolation=cv2.INTER_AREA)
        ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            raise RuntimeError("Output JPEG encoding failed")
        if self.analysis is not None:
            self.analysis["bridge_stages_ms"] = {
                "input_convert_ms": (converted-started)*1000,
                "preprocess_ms": (tracking_started-converted)*1000,
                "tracking_ms": (tracked-tracking_started)*1000,
                "hand_tracking_ms": (hand_done-tracked)*1000,
                "visual_projection_ms": (projection_done-hand_done)*1000,
                "preview_encode_ms": (time.monotonic()-projection_done)*1000}
        return jpeg.tobytes(), (time.monotonic() - started) * 1000


def make_server(state, host="0.0.0.0", port=8084):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def setup(self):
            super().setup()
            self.connection.settimeout(3)
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        def reply(self, code, body, mime="application/json", headers=None):
            if code >= 400 and self.command == 'POST' and not getattr(self,'_body_consumed',False):
                # Small rejected bodies must be drained before close: unread
                # TCP data can reset the socket and discard the error response.
                text = self.headers.get('Content-Length','')
                length = int(text) if text.isascii() and text.isdecimal() else 0
                if 0 < length <= 4096 and not self.headers.get('Transfer-Encoding'):
                    previous_timeout = self.connection.gettimeout()
                    try:
                        self.connection.settimeout(.02)
                        self.rfile.read(length)
                    except OSError:
                        pass
                    finally:
                        self.connection.settimeout(previous_timeout)
            self.send_response(code)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            if code >= 400:
                # An invalid request may leave unread body bytes on the socket.
                self.close_connection = True
                self.send_header("Connection", "close")
            for name, value in (headers or {}).items():
                self.send_header(name, str(value))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != "/api/vision/pc/status":
                self.reply(404, b'{}')
                return
            memory = {}
            try:
                with open('/proc/self/status') as handle:
                    for line in handle:
                        if line.startswith(('VmRSS:','VmSwap:')):
                            key,value,*_ = line.split()
                            memory[key.rstrip(':')+'_kb'] = int(value)
            except OSError:
                pass
            link_speed = None
            try:
                with open('/sys/class/net/eth0/speed') as handle:
                    link_speed = int(handle.read().strip())
            except (OSError, ValueError):
                pass
            neural_stream = getattr(state.neural, 'stream', None)
            self.reply(200, json.dumps({"backend": "pl", "source": "pc_camera",
                "status": "fault" if state.fault else "ready", "error": state.fault,
                "hand_model": getattr(state.hand_pipeline.model,'neural_backend',None) if state.hand_pipeline is not None else None,
                "hand_region_mode": "neck_local_lightweight",
                "neck_model": state.search_worker.status(),
                "neural_hand_enabled": state.hand_pipeline is not None,
                "fingertip_neural_enabled": state.fingertip_worker is not None,
                "fingertip_backend": getattr(state.neural,'backend_name',None) if state.fingertip_worker is not None else None,
                "neural_pl_calls": getattr(state.neural,'calls',0),
                "neural_pl_kernel_ms": getattr(state.neural,'kernel_ms',0.),
                "neural_stream_available": neural_stream is not None,
                "neural_stream_calls": getattr(neural_stream,'calls',0),
                "neural_stream_kernel_ms": getattr(neural_stream,'kernel_ms',0.),
                "neural_requant_available": getattr(state.neural,'requant',None) is not None,
                "neural_requant_calls": getattr(state.neural,'requant_calls',0),
                "neural_requant_ms": getattr(state.neural,'requant_ms',0.),
                "neural_pl_fault": bool(getattr(state.neural,'fault',False)),
                "hand_tracking": getattr(state.flow_backend, "backend_name", None),
                "tracking_priority_enabled": state.work_priority is not None,
                "jpeg_decoder": "ps_libjpeg_rgba" if _JPEG_DECODER is not None else "ps_opencv",
                "analysis": state.last_analysis,
                "analysis_age_seconds": (None if state.last_processed_at is None else
                                         max(0.,time.monotonic()-state.last_processed_at)),
                "memory": memory,
                "link_speed_mbps": link_speed,
                "hardware": state.processor.fingerprint}).encode())

        def do_POST(self):
            self._body_consumed = False
            if self.path != "/api/vision/pc/process":
                self.reply(404, b'{}')
                return
            encoding = self.headers.get("X-Vision-Encoding", "identity")
            try:
                raw_options = self.headers.get("X-Vision-Options")
                if raw_options is not None and len(raw_options) > 2048:
                    raise ValueError("Tracking options too large")
                options = None if raw_options is None else json.loads(raw_options)
                state.validate_options(options)
                roi = options.get("transport_roi") if options else None
                expected_bytes = roi["width"]*roi["height"]*4 if roi else FRAME_BYTES
            except (ValueError, TypeError):
                self.reply(400, b'{"error":"Invalid tracking options"}')
                return
            length_text = self.headers.get("Content-Length", "")
            length = int(length_text) if length_text.isascii() and length_text.isdecimal() else -1
            if (self.headers.get("Content-Type") != "application/octet-stream"
                    or encoding not in ("identity", "deflate", "jpeg")
                    or not 0 < length <= FRAME_BYTES
                    or encoding == "identity" and length != expected_bytes
                    or self.headers.get("Transfer-Encoding")):
                self.reply(400, b'{"error":"Expected fixed 640x480 RGBA body"}')
                return
            token = self.headers.get("X-Vision-Token", "")
            if not token or len(token) > 100 or not all(c.isalnum() or c == "-" for c in token):
                self.reply(400, b'{"error":"Invalid frame token"}')
                return
            try:
                self._body_consumed = True
                body = self.rfile.read(length)
                if len(body) != length:
                    raise ValueError("Truncated frame")
            except (OSError, ValueError):
                self.reply(400, b'{"error":"Incomplete frame"}')
                return
            try:
                # A new camera's first frame reserves admission even while an
                # old request finishes. Otherwise an old polling tab can starve
                # takeover indefinitely by repeatedly winning the busy lock.
                state.check_stream(options)
                if options is not None and options['sequence'] == 1:
                    state.active_stream = (options['stream'], options['generation'], options.get('owner'))
                    state.active_at = time.monotonic()
            except StreamBusy as exc:
                self.reply(409, json.dumps({'error':str(exc)}).encode())
                return
            if not state.lock.acquire(blocking=False):
                self.reply(429, b'{"error":"PL busy; drop frame"}')
                return
            try:
                # Serialize admission before decompression: rejected browser
                # tabs must not consume a PS core decoding discarded frames.
                state.check_stream(options)
                decoded_started = time.monotonic()
                try:
                    body = decode_frame(body, encoding, expected_bytes)
                except ValueError:
                    self.reply(400, b'{"error":"Incomplete frame"}')
                    return
                decode_ms = (time.monotonic()-decoded_started)*1000
                jpeg, elapsed = state.process(body, options)
                if state.analysis is not None:
                    state.analysis["transport"] = {"encoding": encoding, "wire_bytes": length,
                                                   "raw_bytes": expected_bytes, "decode_ms": decode_ms}
                headers = {"X-Vision-Backend": "pl",
                    "X-Vision-Source": "pc_camera", "X-Vision-Token": token,
                    "X-Vision-Processing-Ms": elapsed}
                if state.analysis is not None:
                    headers["X-Vision-Fretboard"] = analysis_header(state.analysis)
                coordinates = options is not None and options.get("coordinates_only")
                planes = options is not None and options.get('preview_encoding') == 'planes'
                self.reply(200, b'{}' if coordinates else jpeg,
                    'application/json' if coordinates else
                    'application/x-fretlens-planes' if planes else 'image/jpeg', headers)
            except StreamBusy as exc:
                self.reply(409,json.dumps({'error':str(exc)}).encode())
            except (ValueError, OSError, RuntimeError) as exc:
                self.reply(503, json.dumps({"error": str(exc)}).encode())
            finally:
                if options is not None and state.active_stream == (
                        options['stream'], options['generation'], options.get('owner')):
                    state.active_at = time.monotonic()
                state.lock.release()

        def log_message(self, *_):
            pass

    return ThreadingHTTPServer((host, port), Handler)


def main():
    global _JPEG_DECODER
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bitfile", required=True)
    parser.add_argument("--load-overlay", action="store_true")
    parser.add_argument("--port", type=int, default=8084)
    parser.add_argument("--hls-flow", action="store_true")
    parser.add_argument("--hls-roi", action="store_true")
    parser.add_argument("--cpu-flow-reference")
    args = parser.parse_args()
    if args.hls_flow and args.cpu_flow_reference:
        parser.error("Choose HLS flow or exact CPU reference")
    cv2.setNumThreads(1)
    requested_nice = int(os.environ.get('FRETLENS_VISION_NICE', '0'))
    if not -10 <= requested_nice <= 19:
        raise ValueError('FRETLENS_VISION_NICE must be between -10 and 19')
    if requested_nice and hasattr(os, 'setpriority'):
        os.setpriority(os.PRIO_PROCESS, 0, requested_nice)
    # Short Python callbacks schedule both the neural worker and current-frame
    # checks on two ARM cores; the default 5 ms GIL interval penalizes each handoff.
    sys.setswitchinterval(.001)
    from pathlib import Path
    if Path(__file__).with_name('libfast_jpeg.so').is_file():
        from fast_jpeg import FastJpegDecoder
        _JPEG_DECODER = FastJpegDecoder()
    # Bind before touching PL, to avoid changing hardware on port conflicts.
    neck_model = os.environ.get('FRETLENS_NECK_MODEL')
    neck_metadata = os.environ.get('FRETLENS_NECK_METADATA')
    if bool(neck_model) != bool(neck_metadata):
        parser.error("FRETLENS_NECK_MODEL and FRETLENS_NECK_METADATA must be configured together")
    neck_detector = None
    if neck_model:
        from neck_model import NeckModel
        neck_detector = NeckModel(neck_model, neck_metadata)
    state = Bridge(None, neck_detector=neck_detector)
    server = make_server(state, port=args.port)
    try:
        state.processor = VisionPL(args.bitfile, download=args.load_overlay)
        if args.cpu_flow_reference:
            from vision_hls import CPUPyramidLK
            state.flow_backend = CPUPyramidLK(args.cpu_flow_reference)
        if args.hls_flow or args.hls_roi:
            from vision_hls import PyramidLK, RoiProjection
            if args.hls_flow:
                state.flow_backend = PyramidLK(state.processor)
                state.fingertip_tracker = state.create_fingertip_tracker(state.flow_backend)
            if args.hls_roi:
                state.projection_backend = RoiProjection(state.processor)
        model_directory = Path(args.bitfile).resolve().parent
        fingertip_directory = os.environ.get('FRETLENS_FINGERTIP_MODEL')
        if fingertip_directory:
            if os.environ.get('FRETLENS_HAND_NEURAL', '0') == '1':
                raise ValueError('Choose one neural model owner for the convolution core')
            from hand_neural_pl import create_neural_engine
            from fingertip_heatmap import FingertipHeatmap
            from fingertip_worker import FingertipWorker
            fused_transport = os.environ.get('FRETLENS_FINGERTIP_FUSED', '0') == '1'
            if fused_transport:
                from fingertip_transport import FingertipQ16NeuralPL, FingertipGraph
                state.neural = FingertipQ16NeuralPL(state.processor)
            else:
                state.neural = create_neural_engine(state.processor)
            if state.neural is None:
                raise RuntimeError('Fingertip model requires the active PL convolution core')
            model_path = Path(fingertip_directory)/'fingertip_heatmap_candidate.onnx'
            if fused_transport:
                graph = FingertipGraph(fingertip_directory, state.neural, model_path=model_path)
                model = FingertipHeatmap(graph=graph)
            else:
                model = FingertipHeatmap(fingertip_directory, state.neural, model_path=model_path)
            state.fingertip_worker = FingertipWorker(model, max_age_ms=1800)
        # Region following is the production fast path. Full landmarks are an
        # explicit opt-in experiment and cannot gate the region overlay.
        if (os.environ.get('FRETLENS_HAND_NEURAL', '0') == '1' and
                state.flow_backend is not None and (model_directory/'palm.onnx').is_file() and (model_directory/'handpose.onnx').is_file()):
            from hand_tracking import HandModel, HandPipeline
            from hand_neural_pl import create_neural_engine
            state.neural = create_neural_engine(state.processor)
            state.hand_pipeline = HandPipeline(HandModel(model_directory,state.neural), state.flow_backend)
        print("PC camera PL bridge ready", flush=True)
        def request_stop(*_):
            state.stopping = True
            threading.Thread(target=server.shutdown,daemon=True).start()
        signal.signal(signal.SIGTERM,request_stop)
        signal.signal(signal.SIGINT,request_stop)
        server.serve_forever()
    finally:
        if state.search_worker is not None:
            state.search_worker.stopping = True
        state.recovery_worker.stopping = True
        state.stopping = True
        server.server_close()
        with state.lock:
            if state.fingertip_worker is not None:
                state.fingertip_worker.shutdown(wait=True)
            if state.hand_pipeline is not None:
                state.hand_pipeline.close()
            if state.neural is not None:
                state.neural.close()
            for accelerator in (state.flow_backend, state.projection_backend):
                if accelerator is not None:
                    accelerator.close()
            if state.processor:
                state.processor.close()


if __name__ == "__main__":
    main()
