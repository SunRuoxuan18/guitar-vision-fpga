"""PS graph / PL neural inference with current-frame PL LK tracking.

Weights and pose crop/decode follow OpenCV Zoo (Apache-2.0). Neural layers run
on the selected backend. LK observations remain distinct from neural inference.
"""
from collections import deque
import os
from pathlib import Path
import threading
import time

import cv2
import numpy as np
from handpose_model import MPHandPose
from patch_similarity import patch_correlations


def classify_hand_motion(before, after, sequence):
    """Use verified joints for both single-finger changes and whole-hand shifts."""
    valid = before['joint_valid'] & after['joint_valid']
    tips = np.array([8, 12, 16, 20])
    valid_tips = tips[valid[tips]]
    anchors = np.array([0, 5, 9, 13, 17])
    anchors = anchors[valid[anchors]]
    delta = after['landmarks']-before['landmarks']
    palm_delta = np.median(delta[anchors], axis=0) if len(anchors) >= 3 else None
    articulation = (np.linalg.norm(delta[valid_tips]-palm_delta, axis=1)
                    if palm_delta is not None else np.array([]))
    finger_change = bool(len(articulation) and np.max(articulation) > .012)
    hand_shift = bool(palm_delta is not None and len(valid_tips) >= 2 and
                      np.linalg.norm(palm_delta) > .015 and
                      (np.linalg.norm(delta[valid_tips], axis=1) > .012).sum() >= 2)
    return dict(sequence=sequence, valid_tips=len(valid_tips), changing=finger_change or hand_shift,
                finger_change=finger_change, hand_shift=hand_shift,
                maximum_tip_change=float(np.max(articulation)) if len(articulation) else 0.,
                source='current_pl_joint_flow')


def recover_local_joint(previous, current, origin, predicted):
    """Bounded PS patch check for a joint rejected by the wider PL LK window."""
    def match(template_image, search_image, template_center, search_center):
        if (np.any(template_center < 6) or np.any(template_center >= np.array(template_image.shape[::-1])-6)
                or np.any(search_center < 14) or np.any(search_center >= np.array(search_image.shape[::-1])-14)):
            return None
        template = cv2.getRectSubPix(template_image, (11,11), tuple(np.float32(template_center)))
        if float(np.std(template)) < 10.:
            return None
        region = cv2.getRectSubPix(search_image, (27,27), tuple(np.float32(search_center)))
        scores = cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED)
        _, peak, _, location = cv2.minMaxLoc(scores)
        x,y = location
        alternative = scores.copy()
        alternative[max(0,y-2):y+3,max(0,x-2):x+3] = -1
        if peak < .92 or peak-float(np.max(alternative)) < .12:
            return None
        return np.float32(search_center)+[x-8,y-8]
    recovered = match(previous, current, origin, predicted)
    if recovered is None:
        return None
    reverse = match(current, previous, recovered, origin)
    if reverse is None or np.linalg.norm(reverse-origin) > 1.:
        return None
    return recovered


class HandModel:
    def __init__(self, directory, neural=None):
        directory = Path(directory)
        self.neural = neural
        self.neural_backend = getattr(neural,'backend_name','ps_opencv_dnn')
        suffix = '_pl.onnx' if neural is not None else '.onnx'
        if neural is not None:
            from hand_neural_pl import register
            register(neural)
        self.palm = cv2.dnn.readNetFromONNX(str(directory/('palm'+suffix)))
        palm_dir = directory/'palm_execution'
        if (neural is not None and palm_dir.is_dir() and
                os.environ.get('HAND_PALM_GRAPH_EXECUTOR', 'opencv') == 'direct'):
            from palm_graph import PalmGraph
            self.palm = PalmGraph(palm_dir, neural, directory/('palm'+suffix))
        self.pose = MPHandPose(str(directory/('handpose'+suffix)), confThreshold=.7)
        self.direct_graph = False
        graph_dir = directory/'handpose_execution'
        if neural is not None and graph_dir.is_dir() and os.environ.get('HAND_GRAPH_EXECUTOR', 'direct') == 'direct':
            from hand_graph import HandGraph
            self.pose.model = HandGraph(graph_dir, neural, self.pose.model_path)
            self.direct_graph = True
            self.neural_backend = self.neural_backend.replace('_ps_graph', '_ps_direct_graph')
        # OpenCV Zoo palm detector uses two anchors per 24x24 cell and six
        # anchors per 12x12 cell, with the duplicate anchors interleaved.
        self.anchors = np.float32([[x/size, y/size]
            for size, repeats in ((24, 2), (12, 6))
            for y in np.arange(size)+.5 for x in np.arange(size)+.5
            for _ in range(repeats)])
        self.roi_detections = 0
        self.diagnostics = {}
        self._snapshot_saved = False

    def _pose(self, rgba, palm):
        bgr = cv2.cvtColor(rgba,cv2.COLOR_RGBA2BGR)
        try:
            pose = self.pose.infer(bgr,palm,tracking_only=True)
        except (cv2.error,ValueError):
            if self.direct_graph:
                raise
            if self.neural is not None and (self.neural.fault or self.neural.stopping):
                raise
            return None
        if pose is None or not np.isfinite(pose).all():
            self.diagnostics['pose_rejected'] = True
            return None
        self.diagnostics['pose_score'] = float(pose[-1])
        height,width = rgba.shape[:2]
        return {'landmarks':pose[4:67].reshape(21,3)[:,:2]/[width,height], 'score':float(pose[-1])}

    def detect(self, rgba, proposal=None):
        self.diagnostics = {'stage': 'palm_search'}
        snapshot = os.environ.get('HAND_DIAGNOSTIC_FRAME')
        if snapshot and not self._snapshot_saved:
            self._snapshot_saved = bool(cv2.imwrite(snapshot, cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)))
        height, width = rgba.shape[:2]
        if proposal is not None and self.roi_detections < 8:
            landmarks = np.asarray(proposal,np.float32)[[0,5,9,13,17,1,2]]*[width,height]
            palm = np.r_[landmarks.min(axis=0),landmarks.max(axis=0),landmarks.reshape(-1),1.]
            pose = self._pose(rgba,palm)
            if pose is not None:
                self.roi_detections += 1
                self.diagnostics['stage'] = 'tracked_roi_hand_detected'
                return [dict(pose, detection_mode='pl_tracked_roi')]
        self.roi_detections = 0
        ratio = 192/max(width,height)
        w, h = int(width*ratio), int(height*ratio)
        left, top = (192-w)//2, (192-h)//2
        rgb = rgba[:,:,:3]
        resized = cv2.resize(rgb, (w,h))
        image = np.zeros((192,192,3), np.float32)
        image[top:top+h,left:left+w] = resized/255.
        self.palm.setInput(image[None])
        outputs = self.palm.forward(self.palm.getUnconnectedOutLayersNames())
        raw = next(x[0] for x in outputs if x.shape[-1] == 18)
        logits = next(x[0,:,0] for x in outputs if x.shape[-1] == 1)
        scores = 1/(1+np.exp(-np.clip(logits,-80,80)))
        self.diagnostics['palm_max_score'] = float(np.max(scores))
        selected = np.flatnonzero(scores >= .55)
        self.diagnostics['palm_candidates'] = int(len(selected))
        if not len(selected):
            self.diagnostics['stage'] = 'no_palm'
            return []
        centers = (raw[:,:2]/192+self.anchors)*max(width,height)-np.array([left,top])/ratio
        wh = raw[:,2:4]/192*max(width,height)
        boxes = np.column_stack((centers-wh/2, wh))
        keep = np.asarray(cv2.dnn.NMSBoxes(boxes[selected].tolist(), scores[selected].tolist(), .55, .3)).reshape(-1)
        hands = []
        # A second palm candidate is useful only after the top candidate's
        # pose has been rejected.  Do not double pose work on normal frames.
        for index in selected[keep][:2]:
            landmarks = (raw[index,4:].reshape(7,2)/192+self.anchors[index])*max(width,height)
            landmarks -= np.array([left,top])/ratio
            palm = np.r_[centers[index]-wh[index]/2, centers[index]+wh[index]/2,
                         landmarks.reshape(-1), scores[index]]
            pose = self._pose(rgba,palm)
            if pose is not None:
                hands.append(dict(pose,detection_mode='full_palm_search'))
                break
        self.diagnostics['stage'] = 'hand_detected' if hands else 'pose_rejected'
        return hands


class HandPipeline:
    """One neural worker and current-frame LK, with separate kernel locks."""
    # A PS/PL neural result can take several seconds.  Keep enough resized
    # frames to bridge that latency without accepting an old pose as current.
    _HISTORY_FRAMES = 192
    _MAX_CATCHUP_STEPS = _HISTORY_FRAMES - 1
    _CATCHUP_MIN_BUDGET_MS = 120.
    _CATCHUP_MAX_BUDGET_MS = 750.
    _RECOVERY_PROPOSAL_TTL = 2.
    def __init__(self, model, flow_backend, keypoint_mode='fingertips'):
        if keypoint_mode not in ('fingertips', 'full'):
            raise ValueError('Unknown hand keypoint mode')
        self.model, self.flow = model, flow_backend
        # Palm anchors retain crop orientation; index joints retain barre shape.
        self.tracked_joint_ids = np.array([0,1,2,5,6,7,8,9,12,13,16,17,20]
            if keypoint_mode == 'fingertips' else list(range(21)), dtype=int)
        self.feature_limit = 16 if keypoint_mode == 'fingertips' else 32
        self.keypoint_mode = keypoint_mode
        self.lock = threading.Lock()
        self.event = threading.Event()
        self.stopping = False
        self.pending = self.completed = None
        self.busy = False
        self.identity = None
        self.epoch = 0
        self.detector_ttl = 40. if getattr(model,'neural',None) is not None else 8.
        self.history = deque(maxlen=(self._HISTORY_FRAMES if getattr(model,'neural',None) is not None else 36))
        self.track = None
        self._model_proposal = None
        self._model_proposal_last_valid_at = float('-inf')
        self._model_proposal_identity = None
        self._model_proposal_epoch = None
        self.error = None
        self.detector_completed = 0
        self.last_inference_ms = 0.
        self.detector_hand_count = 0
        self.detector_diagnostics = {}
        self.catchup_diagnostics = {}
        self.acceptance_reason = 'waiting_for_detection'
        self.motion = {}
        self.motion_until = float('-inf')
        self.motion_event = {}
        self.motion_samples = deque(maxlen=16)
        self.next_detection_at = 0.
        self.stable_refresh_seconds = 1.5
        self.anchor_refresh_seconds = 4.
        self.detection_requests = 0
        self.worker = threading.Thread(target=self._run, name='ps-hand-model', daemon=True)
        self.worker.start()

    def _run(self):
        while not self.stopping:
            self.event.wait(.2)
            if self.stopping:
                break
            with self.lock:
                job = self.pending
                self.pending = None
                self.event.clear()
            if job is None:
                continue
            identity, sequence, rgba, started, proposal, epoch = job
            try:
                hands = self.model.detect(rgba,proposal)
                result = (identity, sequence, hands, started, (time.monotonic()-started)*1000, epoch)
                with self.lock:
                    if epoch == self.epoch and not self.stopping:
                        self.completed = result
                        self.error = None
                        self.detector_completed += 1
                        self.last_inference_ms = result[4]
                        self.detector_hand_count = len(hands)
                        self.detector_diagnostics = dict(getattr(self.model, 'diagnostics', {}))
            except Exception as exc:
                with self.lock:
                    if epoch == self.epoch:
                        self.completed = None
                        self.error = str(exc)
            finally:
                with self.lock:
                    self.busy = False

    def _initialize(self, record, hand, detected_at, inference_ms):
        sequence, gray = record
        points = np.float32(hand['landmarks'])
        lo = np.maximum(0, points.min(axis=0)-.04)
        hi = np.minimum(1, points.max(axis=0)+.04)
        mask = np.zeros_like(gray)
        x0,y0 = np.int32(lo*[320,200]); x1,y1 = np.int32(hi*[320,200])
        mask[y0:y1,x0:x1] = 255
        features = cv2.goodFeaturesToTrack(gray,self.feature_limit,.01,4,mask=mask)
        if features is None or len(features) < 8:
            return None
        return dict(sequence=sequence, detected_sequence=sequence, gray=gray,
                    features=features, landmarks=points, score=hand['score'],
                    detected_at=detected_at, inference_ms=inference_ms, inliers=len(features),
                    joint_valid=np.ones(21,dtype=bool),
                    detection_mode=hand.get('detection_mode','unknown'))

    def _recovery_proposal(self, identity, now):
        if (self._model_proposal is None or self._model_proposal_identity != identity or
                self._model_proposal_epoch != self.epoch or
                now-self._model_proposal_last_valid_at > self._RECOVERY_PROPOSAL_TTL):
            return None
        return self._model_proposal.copy()

    def _advance(self, track, record, flow=None):
        sequence, gray = record
        if sequence <= track['sequence']:
            return track
        features = track['features']
        count = len(features)
        joints = np.float32(track['landmarks']*[320,200]).reshape(-1,1,2)
        joint_ids = self.tracked_joint_ids
        all_points = np.concatenate((features,joints[joint_ids]))
        flow = self.flow if flow is None else flow
        forward, good, error = flow(track['gray'],gray,all_points)
        forward_kernel_ms = getattr(flow,'timings',{}).get('kernel_ms',0.)
        if forward is None:
            return None
        backward, reverse_good, reverse_error = flow(gray,track['gray'],forward)
        backward_kernel_ms = getattr(flow,'timings',{}).get('kernel_ms',0.)
        if backward is None or reverse_good is None or good is None:
            return None
        good = good.reshape(-1).astype(bool) & reverse_good.reshape(-1).astype(bool)
        good &= np.linalg.norm(backward.reshape(-1,2)-all_points.reshape(-1,2),axis=1) < 1.5
        good &= np.isfinite(forward.reshape(-1,2)).all(axis=1)
        good &= (forward.reshape(-1,2) >= [6,6]).all(axis=1) & (forward.reshape(-1,2) < [314,194]).all(axis=1)
        if error is not None:
            good &= np.isfinite(error.reshape(-1)) & (error.reshape(-1) < 20)
        if reverse_error is not None:
            good &= np.isfinite(reverse_error.reshape(-1)) & (reverse_error.reshape(-1) < 20)
        region_good = good[:count]
        if region_good.sum() < 8 or region_good.mean() < .6:
            return None
        before, after = features.reshape(-1,2)[region_good], forward[:count].reshape(-1,2)[region_good]
        transform, inliers = cv2.estimateAffinePartial2D(before,after,method=cv2.RANSAC,ransacReprojThreshold=2)
        if transform is None or inliers.sum() < 8 or inliers.mean() < .65:
            return None
        scale = np.linalg.norm(transform[:,0])
        if not .75 < scale < 1.3:
            return None
        predicted = cv2.transform(joints.reshape(1,-1,2),transform)[0]
        local = forward[count:].reshape(-1,2)
        local_valid = good[count:] & track['joint_valid'][joint_ids]
        # Local patches can follow fingers independently of the rigid palm.
        # Failed joints use affine coordinates for drawing only, never evidence.
        local_valid &= np.linalg.norm(local-predicted[joint_ids],axis=1) < 16
        indices = np.flatnonzero(local_valid)
        local_strength = np.zeros(len(joint_ids),np.float32)
        if len(indices):
            old_patches = np.stack([cv2.getRectSubPix(track['gray'],(11,11),
                tuple(joints[joint_ids[index],0])) for index in indices])
            new_patches = np.stack([cv2.getRectSubPix(gray,(11,11),
                tuple(local[index])) for index in indices])
            local_strength[indices] = patch_correlations(old_patches,new_patches,minimum_energy=100.)
            local_valid[indices] = local_strength[indices] >= .88
        joint_valid = np.zeros(21,dtype=bool)
        recovered_count = 0
        for index, joint_id in enumerate(joint_ids):
            if (joint_id not in (6,7,8,12,16,20) or not track['joint_valid'][joint_id]
                    or local_valid[index] and local_strength[index] >= .96):
                continue
            recovered = recover_local_joint(track['gray'], gray, joints[joint_id,0], predicted[joint_id])
            if recovered is not None:
                local[index] = recovered
                local_valid[index] = True
                recovered_count += 1
        joint_valid[joint_ids] = local_valid
        predicted[joint_ids[local_valid]] = local[local_valid]
        landmarks = predicted/[320,200]
        if not np.isfinite(landmarks).all() or (landmarks < -.05).any() or (landmarks > 1.05).any():
            return None
        retained = after[inliers.reshape(-1).astype(bool)].reshape(-1,1,2)
        # Replenish only inside the verified moving hand. Without renewal,
        # normal feature attrition makes long detector catch-up fail inevitably.
        if len(retained) < self.feature_limit:
            mask = np.zeros_like(gray)
            hull = cv2.convexHull(np.rint(predicted).astype(np.int32))
            cv2.fillConvexPoly(mask, hull, 255)
            for point in retained.reshape(-1,2):
                cv2.circle(mask, tuple(np.rint(point).astype(int)), 4, 0, -1)
            fresh = cv2.goodFeaturesToTrack(gray, self.feature_limit-len(retained), .01, 4, mask=mask)
            if fresh is not None:
                retained = np.concatenate((retained, fresh))
        return dict(track, sequence=sequence, gray=gray, features=retained,
                    landmarks=landmarks, inliers=int(inliers.sum()),
                    joint_valid=joint_valid,
                    local_patch_recovered_joints=recovered_count,
                    flow_kernel_ms=float(forward_kernel_ms+backward_kernel_ms),
                    ps_tracking_fallback=flow is not self.flow)

    def _catchup(self, track, record):
        advanced = self._advance(track, record)
        if advanced is not None:
            return advanced
        def ps_flow(before, after, points):
            return cv2.calcOpticalFlowPyrLK(before, after, points, None,
                winSize=(21,21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
        advanced = self._advance(track, record, flow=ps_flow)
        if advanced is not None:
            advanced['ps_catchup_steps'] = track.get('ps_catchup_steps', 0)+1
        return advanced

    def _catchup_detection(self, candidate, records, inference_ms):
        """Verify the latest frame directly, then retry through dense history.

        Adjacent images make LK's displacement assumption much more reliable
        than a uniformly sampled long jump.  The wall-clock budget prevents a
        delayed detector completion from monopolizing the live frame loop.
        A candidate is returned only when it has been verified at the latest
        record; callers never draw an intermediate or stale candidate.
        """
        budget_ms = min(self._CATCHUP_MAX_BUDGET_MS,
                        max(self._CATCHUP_MIN_BUDGET_MS, inference_ms*.15))
        started = time.monotonic()
        deadline = started+budget_ms/1000.
        attempted = 0
        adjacent_steps = 0
        max_sequence_gap = 0
        reason = 'reached_current'
        mode = 'already_current'
        remaining = records[1:]
        if remaining:
            mode = 'dense_adjacent'
            max_sequence_gap = records[-1][0]-candidate['sequence']
            attempted = 1
            direct = self._catchup(candidate,records[-1])
            if direct is not None and direct['sequence'] == records[-1][0]:
                candidate = direct
                mode = 'direct_verified'
                remaining = []
            # The direct attempt and dense retry share one deadline. A late
            # success is discarded just like a partial dense catch-up.
            if time.monotonic() >= deadline:
                candidate = None
                reason = 'time_budget'
                remaining = []
        for record in remaining:
            if attempted-1 >= self._MAX_CATCHUP_STEPS:
                reason = 'step_limit'
                candidate = None
                break
            if time.monotonic() >= deadline:
                reason = 'time_budget'
                candidate = None
                break
            gap = record[0]-candidate['sequence']
            max_sequence_gap = max(max_sequence_gap, gap)
            adjacent_steps += int(gap == 1)
            attempted += 1
            candidate = self._catchup(candidate,record)
            if time.monotonic() >= deadline:
                reason = 'time_budget'
                candidate = None
                break
            if candidate is None:
                reason = 'geometric_check_failed'
                break
        reached_current = candidate is not None and candidate['sequence'] == records[-1][0]
        if not reached_current:
            candidate = None
            if reason == 'reached_current':
                reason = 'current_frame_not_reached'
        self.catchup_diagnostics = dict(history_frames=len(self.history),
            history_capacity=self.history.maxlen, records=len(records), attempted_steps=attempted,
            adjacent_steps=adjacent_steps, max_sequence_gap=max_sequence_gap,
            elapsed_ms=(time.monotonic()-started)*1000, budget_ms=budget_ms,
            reached_current=reached_current, reason=reason, mode=mode)
        return candidate

    def _observe_motion(self, before, after, now):
        def snapshot(track):
            return dict(sequence=track['sequence'], landmarks=track['landmarks'].copy(),
                        joint_valid=track['joint_valid'].copy(),
                        ps_tracking_fallback=track.get('ps_tracking_fallback', False))
        if not self.motion_samples:
            self.motion_samples.append((now, snapshot(before)))
        while len(self.motion_samples) > 1 and now-self.motion_samples[0][0] > .12:
            self.motion_samples.popleft()
        if now-self.motion_samples[0][0] > .12:
            self.motion_samples.clear()
            self.motion_samples.append((now, snapshot(before)))
        reference_time, reference = self.motion_samples[0]
        immediate = classify_hand_motion(before, after, after['sequence'])
        cumulative = classify_hand_motion(reference, after, after['sequence'])
        cumulative['finger_change'] |= immediate['finger_change']
        cumulative['hand_shift'] |= immediate['hand_shift']
        cumulative['changing'] = cumulative['finger_change'] or cumulative['hand_shift']
        cumulative['comparison_span_ms'] = max(0., (now-reference_time)*1000)
        cumulative['comparison_sequence'] = reference['sequence']
        if (before.get('ps_tracking_fallback') or after.get('ps_tracking_fallback') or
                any(record.get('ps_tracking_fallback') for _,record in self.motion_samples)):
            cumulative['source'] = 'current_ps_joint_flow_recovery'
        self.motion_samples.append((now, snapshot(after)))
        return cumulative

    def update(self, rgba, pl_gray, options):
        started = time.monotonic()
        identity = (options['stream'],options['generation'])
        sequence = options['sequence']
        if identity != self.identity:
            with self.lock:
                self.epoch += 1
                self.identity = identity
                self.completed = None
                self.error = None
            self.history.clear()
            self.track = None
            self._model_proposal = None
            self._model_proposal_last_valid_at = float('-inf')
            self._model_proposal_identity = None
            self._model_proposal_epoch = None
            self.next_detection_at = 0.
            self.motion = {}
            self.motion_until = float('-inf')
            self.motion_event = {}
            self.motion_samples.clear()
            self.catchup_diagnostics = {}
            self.acceptance_reason = 'waiting_for_detection'
        gray = cv2.resize(pl_gray,(320,200),interpolation=cv2.INTER_AREA)
        self.history.append((sequence,gray))
        with self.lock:
            completed = self.completed
            self.completed = None
            error = self.error
        if completed is not None and completed[0] == identity and completed[5] == self.epoch:
            _, detected_sequence, hands, detected_at, inference_ms, _ = completed
            records = [r for r in self.history if r[0] >= detected_sequence]
            if records and records[0][0] == detected_sequence and started-detected_at < self.detector_ttl:
                candidate = self._initialize(records[0],hands[0],detected_at,inference_ms) if hands else None
                self.acceptance_reason = ('no_model_hand' if not hands else
                    'insufficient_tracking_features' if candidate is None else 'tracking')
                if candidate is not None:
                    candidate = self._catchup_detection(candidate,records,inference_ms)
                    if candidate is None:
                        self.acceptance_reason = 'detection_catchup_failed'
                # A failed catch-up from an old positive detection must not
                # discard an independently valid current track.
                if candidate is not None or (not hands and (self.track is None or
                        self.track['sequence'] <= detected_sequence)):
                    self.track = candidate
                    self.motion_samples.clear()
                if candidate is not None:
                    # Stable, individually verified LK joints need not compete
                    # with an uninterrupted stream of duplicate CNN forwards.
                    candidate['refreshed_at'] = started
                    self.next_detection_at = started + self.stable_refresh_seconds
                    self._model_proposal = np.asarray(hands[0]['landmarks'], np.float32).copy()
                    self._model_proposal_last_valid_at = started
                    self._model_proposal_identity = identity
                    self._model_proposal_epoch = self.epoch
            else:
                self.acceptance_reason = 'detection_outside_history_or_ttl'
        if self.track is not None:
            if started-self.track['detected_at'] > self.detector_ttl:
                self.track = None
                self.motion = {}
                self.motion_until = float('-inf')
                self.motion_event = {}
                self.motion_samples.clear()
            else:
                before = self.track
                self.track = self._catchup(self.track,(sequence,gray))
                if self.track is not None:
                    self.acceptance_reason = 'tracking'
                    self._model_proposal_last_valid_at = started
                    self.motion = self._observe_motion(before, self.track, started)
                    if self.motion['changing']:
                        self.motion_until = started+.25
                        self.motion_event = dict(self.motion)
                    elif started <= self.motion_until:
                        self.motion.update(changing=True,
                            finger_change=self.motion_event.get('finger_change', False),
                            hand_shift=self.motion_event.get('hand_shift', False),
                            transition_event_sequence=self.motion_event['sequence'])
                else:
                    self.acceptance_reason = 'current_tracking_failed'
                    self.motion = {}
                    self.motion_until = float('-inf')
                    self.motion_event = {}
                    self.motion_samples.clear()
        with self.lock:
            request_reason = ('reacquire' if self.track is None else
                'insufficient_fingertips' if int(np.count_nonzero(self.track['joint_valid'][[8,12,16,20]])) < 3 else
                'finger_motion' if self.motion.get('sequence') == sequence and self.motion.get('finger_change') is True else
                'anchor_refresh' if started-self.track.get('refreshed_at',self.track['detected_at']) >= self.anchor_refresh_seconds else
                'periodic_refresh' if started >= self.next_detection_at else 'stable_tracking')
            if not self.busy and not self.stopping and request_reason != 'stable_tracking':
                proposal = (self.track['landmarks'].copy() if self.track is not None else
                            self._recovery_proposal(identity, started))
                self.pending = (identity,sequence,rgba.copy(),started,proposal,self.epoch)
                self.busy = True
                self.detection_requests += 1
                self.event.set()
            scheduler = dict(request_reason=request_reason,inference_busy=self.busy,
                detection_requests=self.detection_requests,keypoint_mode=self.keypoint_mode,
                tracked_joint_ids=self.tracked_joint_ids.tolist(), feature_limit=self.feature_limit,
                next_refresh_ms=max(0.,(self.next_detection_at-started)*1000),
                anchor_refresh_seconds=self.anchor_refresh_seconds)
        hands = []
        if self.track is not None:
            track = self.track
            points = track['landmarks']
            hands = [{'landmarks':[dict(x=float(x),y=float(y),z=0.) for x,y in points],
                      'score':track['score'], 'landmarks_observed':sequence == track['detected_sequence'],
                      'detected_sequence':track['detected_sequence'], 'tracked_sequence':sequence,
                      'inference_ms':track['inference_ms'], 'inliers':track['inliers']}]
            hands[0]['joint_valid'] = track['joint_valid'].tolist()
            hands[0]['local_patch_recovered_joints'] = track.get('local_patch_recovered_joints',0)
            hands[0]['joint_tracking'] = ('pl_lk_ps_local_patch_fb' if hands[0]['local_patch_recovered_joints']
                                          else 'pl_lk_per_joint_fb_patch')
            if track.get('ps_tracking_fallback'):
                hands[0]['joint_tracking'] = 'ps_lk_per_joint_fb_patch'
            hands[0]['keypoint_source'] = 'neural' if sequence == track['detected_sequence'] else 'optical_flow'
            hands[0]['fresh_for_position_candidates'] = started-track['detected_at'] <= 1.5
            hands[0]['current_for_position_candidates'] = (
                sequence == track['sequence'] and started-track['detected_at'] <= 8. and
                track.get('flow_kernel_ms',0.) > 0)
            hands[0]['detection_mode'] = track['detection_mode']
            hands[0]['flow_kernel_ms'] = track.get('flow_kernel_ms',0.)
            hands[0]['detector_age_ms'] = (started-track['detected_at'])*1000
            hands[0]['ps_catchup_steps'] = track.get('ps_catchup_steps',0)
        return dict(hands=hands,status='tracking' if hands else 'error' if error else 'searching',error=error,
                    motion=self.motion if self.motion.get('sequence') == sequence else {},scheduler=scheduler,
                    inference_location='ps_pl' if getattr(self.model,'neural',None) is not None else 'ps_linux',
                    neural_backend=getattr(self.model,'neural_backend','ps_opencv_dnn'),
                    neural_pl_calls=getattr(getattr(self.model,'neural',None),'calls',0),
                    neural_pl_kernel_ms=getattr(getattr(self.model,'neural',None),'kernel_ms',0.),
                    neural_requant_calls=getattr(getattr(self.model,'neural',None),'requant_calls',0),
                    neural_requant_ms=getattr(getattr(self.model,'neural',None),'requant_ms',0.),
                    detector_completed=self.detector_completed,last_inference_ms=self.last_inference_ms,
                    detector_hand_count=self.detector_hand_count,
                    detector_diagnostics=self.detector_diagnostics,
                    catchup_diagnostics=self.catchup_diagnostics,
                    acceptance_reason=self.acceptance_reason,
                    tracking_backend=getattr(self.flow,'backend_name','unknown'),
                    stream_id=identity[0],generation=identity[1],sequence=sequence,
                    processing_ms=(time.monotonic()-started)*1000)

    def close(self):
        with self.lock:
            self.stopping = True
            self.pending = None
        self.event.set()
        neural = getattr(self.model,'neural',None)
        if neural is not None:
            neural.request_stop()
        self.worker.join(timeout=max(10,getattr(neural,'timeout',0)+2))
        if self.worker.is_alive():
            raise RuntimeError('PS hand inference worker did not stop')
        with self.lock:
            self.busy = False
