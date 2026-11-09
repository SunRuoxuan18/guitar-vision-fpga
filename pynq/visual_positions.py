"""Current-frame fingertip/visible-wire projections on PS; pressure is unknown."""
from bisect import bisect_right
import math

import cv2
import numpy as np


def validate_note_calibration(calibration):
    if calibration is None:
        return
    required = {'stream_id', 'generation', 'lattice_epoch', 'first_boundary_fret',
                'strings_reversed', 'frets_reversed', 'tuning'}
    if not isinstance(calibration, dict) or set(calibration) != required:
        raise ValueError('Invalid note calibration')
    if (not isinstance(calibration['stream_id'], str) or
            not 1 <= len(calibration['stream_id']) <= 80 or
            any(type(calibration[k]) is not int or calibration[k] < 0
                for k in ('generation', 'lattice_epoch', 'first_boundary_fret')) or
            calibration['first_boundary_fret'] > 24 or
            any(type(calibration[k]) is not bool for k in ('strings_reversed', 'frets_reversed')) or
            not isinstance(calibration['tuning'], list) or len(calibration['tuning']) != 6 or
            any(type(m) is not int or not 0 <= m <= 103 for m in calibration['tuning'])):
        raise ValueError('Invalid note calibration')


def _observations(analysis, tips):
    if tips is not None:
        return [(0, finger, tip, 'four_fingertip_heatmap', None)
                for finger, tip in tips.items()
                if finger in ('index', 'middle', 'ring', 'little')]
    observations = []
    for hand_index, hand in enumerate(analysis.get('hand_detection', {}).get('hands', [])):
        current_flow = (hand.get('current_for_position_candidates') is True and
            hand.get('keypoint_source') == 'optical_flow' and
            hand.get('joint_tracking') in ('pl_lk_per_joint_fb_patch', 'pl_lk_ps_local_patch_fb') and
            0 <= hand.get('detector_age_ms', float('inf')) <= 8000 and hand.get('flow_kernel_ms', 0) > 0)
        if (hand.get('tracked_sequence') != analysis.get('sequence') or
                not (hand.get('fresh_for_position_candidates') or current_flow) or hand.get('score', 0) < .65):
            continue
        landmarks, valid = hand.get('landmarks', []), hand.get('joint_valid', [])
        if len(landmarks) != 21 or len(valid) != 21:
            continue
        for finger, index in (('index', 8), ('middle', 12), ('ring', 16), ('pinky', 20)):
            if valid[index]:
                observations.append((hand_index, finger, landmarks[index],
                                     hand.get('keypoint_source'), hand.get('detector_age_ms')))
    return observations


def project_positions(analysis, tips=None, calibration=None):
    result = dict(status='unlocalized',positions=[],processing_location='ps_linux',
        absolute_frets_known=False,pressed_strings_known=False,
        diagnostics=dict(eligible_hands=0,valid_fingertips=0,outside_measured_lattice=0,
                         outside_visible_fret_intervals=0))
    if analysis.get('status') != 'tracking':
        return result
    strings,frets = analysis.get('strings',[]),analysis.get('frets',[])
    corners = analysis.get('corners')
    if len(strings) != 6 or not 5 <= len(frets) <= 25 or not corners or len(corners) != 4:
        return result
    tracked = (analysis.get('stage') == 'tracked_lattice' and
        0 <= analysis.get('lattice_age_ms',float('inf')) <= 600 and
        analysis.get('flow',{}).get('inlier_ratio',0) >= .7)
    lines = strings+frets
    if not all(line.get('observed') is True or line.get('observed') is False and tracked for line in lines):
        return result
    quad = np.float32([[p['x'],p['y']] for p in corners])
    if (not np.isfinite(quad).all() or not cv2.isContourConvex(quad) or
            abs(cv2.contourArea(quad)) < .0001):
        return result
    transform = cv2.getPerspectiveTransform(quad,np.float32([[0,0],[1,0],[1,1],[0,1]]))
    endpoints = np.float32([[[line[key]['x'],line[key]['y']] for key in ('start','end')] for line in lines])
    if not np.isfinite(endpoints).all() or (endpoints < 0).any() or (endpoints > 1).any():
        return result
    mapped = cv2.perspectiveTransform(endpoints.reshape(-1,1,2),transform).reshape(-1,2,2)
    result['status'] = 'no_fingertips_on_board'
    observations = _observations(analysis, tips)
    result['diagnostics']['eligible_hands'] = len({o[0] for o in observations})
    valid = []
    for observation in observations:
        tip = observation[2]
        if (not isinstance(tip, dict) or
                any(type(tip.get(k)) not in (int, float) or not math.isfinite(tip[k])
                    or not 0 <= tip[k] <= 1 for k in ('x', 'y'))):
            continue
        valid.append(observation)
    result['diagnostics']['valid_fingertips'] = len(valid)
    inside = []
    if valid:
        points = np.float32([[[o[2]['x'], o[2]['y']]] for o in valid])
        points = cv2.perspectiveTransform(points, transform).reshape(-1, 2)
    else:
        points = ()
    for observation, (x, y) in zip(valid, points):
        if not 0 < x < 1 or not -.03 <= y <= 1.03:
            result['diagnostics']['outside_measured_lattice'] += 1
            continue
        inside.append((observation, x, y))
    boundaries_by_tip = strings_by_tip = ()
    if inside:
        centers = mapped.mean(axis=1)
        string_lines = mapped[sorted(range(6), key=lambda i: centers[i, 1])]
        fret_lines = mapped[sorted(range(6, len(mapped)), key=lambda i: centers[i, 0])]

        def at(lines, axis, coordinates):
            span = (lines[:, 1, 1-axis]-lines[:, 0, 1-axis]).astype(np.float64)
            span[np.abs(span) <= 1e-6] = np.nan
            # Match scalar promotion on both NumPy 1.x (float64) and 2.x (float32).
            dtype = np.result_type(coordinates[0], 1.)
            numerator = ((coordinates[:, None]-lines[None, :, 0, 1-axis]) *
                         (lines[None, :, 1, axis]-lines[None, :, 0, axis]))
            return (lines[None, :, 0, axis] + numerator.astype(dtype)/span.astype(dtype)).tolist()

        coordinates = np.asarray([(x, y) for _, x, y in inside], dtype=np.float32)
        boundaries_by_tip = at(fret_lines, 0, coordinates[:, 1])
        strings_by_tip = at(string_lines, 1, coordinates[:, 0])
    for (observation, x, y), boundaries, ys in zip(inside, boundaries_by_tip, strings_by_tip):
        hand_index, finger, tip, source, neural_age = observation
        x, y = float(x), float(y)
        if (not all(math.isfinite(v) for v in boundaries)
                or not all(a < b for a, b in zip(boundaries, boundaries[1:]))):
            result['diagnostics']['outside_visible_fret_intervals'] += 1
            continue
        next_index = bisect_right(boundaries, x)
        if not 1 <= next_index < len(boundaries):
            result['diagnostics']['outside_visible_fret_intervals'] += 1
            continue
        within = (x-boundaries[next_index-1])/(boundaries[next_index]-boundaries[next_index-1])
        spacing = (ys[-1]-ys[0])/5
        if (not all(math.isfinite(v) for v in ys) or spacing <= .02
                or not all(a < b for a, b in zip(ys, ys[1:]))):
            continue
        for string,yline in enumerate(ys):
            distance = abs(y-yline)/spacing
            if distance <= .55:
                result['positions'].append(dict(x=float(tip['x']),y=float(tip['y']),finger=finger,
                    hand_index=hand_index,string_line=string+1,fret_interval=next_index,relative=True,
                    ambiguous=bool(distance > .35 or within < .07 or within > .93),
                    pressed=None,sounding=None,keypoint_source=source,
                    neural_age_ms=neural_age,
                    string_distance=float(distance), fret_fraction=float(within),
                    evidence='tracked_lines_projection' if tracked else 'observed_lines_projection'))
    if result['positions']:
        result['status'] = 'relative_positions'
    elif result['diagnostics']['eligible_hands'] == 0:
        result['reason'] = 'no_current_valid_hand'
    elif result['diagnostics']['valid_fingertips'] == 0:
        result['reason'] = 'no_current_valid_fingertips'
    elif result['diagnostics']['outside_measured_lattice'] == result['diagnostics']['valid_fingertips']:
        result['reason'] = 'hand_outside_measured_lattice'
    else:
        result['reason'] = 'no_unambiguous_visible_string_fret_cell'
    if calibration is not None:
        validate_note_calibration(calibration)
        identity = ('stream_id', 'generation', 'lattice_epoch')
        if any(analysis.get(k) != calibration[k] for k in identity):
            result['calibration_reason'] = 'calibration_identity_mismatch'
            return result
        first = calibration['first_boundary_fret']
        last = first + (1-len(frets) if calibration['frets_reversed'] else len(frets)-1)
        if not 0 <= last <= 24:
            result['calibration_reason'] = 'fret_numbers_out_of_range'
            return result
        result['absolute_frets_known'] = True
        result['mapping_source'] = 'user_wire_calibration'
        names = ('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B')
        for position in result['positions']:
            string = position['string_line'] if calibration['strings_reversed'] else 7-position['string_line']
            interval = position['fret_interval']
            fret = first+1-interval if calibration['frets_reversed'] else first+interval
            midi = calibration['tuning'][6-string]+fret
            position.update(string=string, fret=fret, midi_note=midi,
                note_name=names[midi % 12]+str(midi//12-1),
                frequency_hz=round(440.*2.**((midi-69)/12.), 3), relative=False,
                mapping_source='user_wire_calibration')
        if result['positions']:
            result['status'] = 'note_candidates'
    return result
