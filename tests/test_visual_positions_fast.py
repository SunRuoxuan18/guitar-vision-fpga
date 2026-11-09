"""Projection parity against the pre-optimization geometry and boundary rules."""

import copy
import math
from pathlib import Path
import sys
import unittest
from unittest import mock

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pynq'))
import visual_positions as current


def original_projection(analysis, tips=None, calibration=None, legacy_promotion=False):
    result = dict(status='unlocalized', positions=[], processing_location='ps_linux',
        absolute_frets_known=False, pressed_strings_known=False,
        diagnostics=dict(eligible_hands=0, valid_fingertips=0, outside_measured_lattice=0,
                         outside_visible_fret_intervals=0))
    if analysis.get('status') != 'tracking':
        return result
    strings, frets = analysis.get('strings', []), analysis.get('frets', [])
    corners = analysis.get('corners')
    if len(strings) != 6 or not 5 <= len(frets) <= 25 or not corners or len(corners) != 4:
        return result
    tracked = (analysis.get('stage') == 'tracked_lattice' and
        0 <= analysis.get('lattice_age_ms', float('inf')) <= 600 and
        analysis.get('flow', {}).get('inlier_ratio', 0) >= .7)
    lines = strings + frets
    if not all(line.get('observed') is True or line.get('observed') is False and tracked for line in lines):
        return result
    quad = np.float32([[p['x'], p['y']] for p in corners])
    if (not np.isfinite(quad).all() or not cv2.isContourConvex(quad) or
            abs(cv2.contourArea(quad)) < .0001):
        return result
    transform = cv2.getPerspectiveTransform(quad, np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]))
    endpoints = np.float32([[[line[key]['x'], line[key]['y']] for key in ('start', 'end')] for line in lines])
    if not np.isfinite(endpoints).all() or (endpoints < 0).any() or (endpoints > 1).any():
        return result
    mapped = cv2.perspectiveTransform(endpoints.reshape(-1, 1, 2), transform).reshape(-1, 2, 2)
    string_lines = sorted(mapped[:6], key=lambda line: line[:, 1].mean())
    fret_lines = sorted(mapped[6:], key=lambda line: line[:, 0].mean())

    def at(line, axis, coordinate):
        span = float(line[1, 1-axis] - line[0, 1-axis])
        if legacy_promotion:
            span = np.float64(span)
        return (float(line[0, axis] + (coordinate-line[0, 1-axis]) *
                      (line[1, axis]-line[0, axis]) / span)
                if abs(span) > 1e-6 else float('nan'))

    result['status'] = 'no_fingertips_on_board'
    observations = current._observations(analysis, tips)
    result['diagnostics']['eligible_hands'] = len({o[0] for o in observations})
    for hand_index, finger, tip, source, neural_age in observations:
        if (not isinstance(tip, dict) or
                any(type(tip.get(k)) not in (int, float) or not np.isfinite(tip[k])
                    or not 0 <= tip[k] <= 1 for k in ('x', 'y'))):
            continue
        xy = np.float32([[[tip['x'], tip['y']]]])
        result['diagnostics']['valid_fingertips'] += 1
        x, y = cv2.perspectiveTransform(xy, transform)[0, 0]
        if not 0 < x < 1 or not -.03 <= y <= 1.03:
            result['diagnostics']['outside_measured_lattice'] += 1
            continue
        boundaries = np.array([at(line, 0, y) for line in fret_lines])
        next_index = int(np.searchsorted(boundaries, x, side='right'))
        if (not np.isfinite(boundaries).all() or not 1 <= next_index < len(boundaries)
                or not (np.diff(boundaries) > 0).all()):
            result['diagnostics']['outside_visible_fret_intervals'] += 1
            continue
        within = (x-boundaries[next_index-1]) / (boundaries[next_index]-boundaries[next_index-1])
        ys = np.array([at(line, 1, x) for line in string_lines])
        spacing = (ys[-1]-ys[0]) / 5
        if not np.isfinite(ys).all() or spacing <= .02 or not (np.diff(ys) > 0).all():
            continue
        for string, yline in enumerate(ys):
            distance = abs(y-yline) / spacing
            if distance <= .55:
                result['positions'].append(dict(x=float(tip['x']), y=float(tip['y']), finger=finger,
                    hand_index=hand_index, string_line=string+1, fret_interval=next_index, relative=True,
                    ambiguous=bool(distance > .35 or within < .07 or within > .93),
                    pressed=None, sounding=None, keypoint_source=source, neural_age_ms=neural_age,
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
        current.validate_note_calibration(calibration)
        if any(analysis.get(k) != calibration[k] for k in ('stream_id', 'generation', 'lattice_epoch')):
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


def fixture(seed=0, frets=9, perspective=True):
    rng = np.random.default_rng(seed)
    quad = np.float32([[.08, .13], [.91, .06], [.85, .93], [.16, .81]])
    if perspective:
        quad += rng.uniform(-.025, .025, quad.shape).astype(np.float32)
    else:
        quad = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
    transform = cv2.getPerspectiveTransform(np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]), quad)

    def point(x, y):
        a, b = cv2.perspectiveTransform(np.float32([[[x, y]]]), transform)[0, 0]
        return dict(x=float(a), y=float(b))

    def line(a, b):
        return dict(start=point(*a), end=point(*b), observed=True)

    xs = .05 + .90 * (1-2.**(-np.arange(frets)/12)) / (1-2.**(-(frets-1)/12))
    slant = rng.uniform(-.025, .025) if perspective else 0.
    frame = dict(status='tracking', sequence=12, stage='lattice', stream_id='fast-fixture',
                 generation=3, lattice_epoch=4,
                 corners=[dict(x=float(x), y=float(y)) for x, y in quad],
                 strings=[line((.02, y), (.98, y+slant)) for y in np.linspace(.10, .90, 6)],
                 frets=[line((x, .02), (x+slant, .98)) for x in xs])
    tips = {name: point(rng.uniform(.07, .93), rng.uniform(.04, .96))
            for name in ('index', 'middle', 'ring', 'little')}
    calibration = dict(stream_id='fast-fixture', generation=3, lattice_epoch=4,
                       first_boundary_fret=0, strings_reversed=False, frets_reversed=False,
                       tuning=[40, 45, 50, 55, 59, 64])
    return frame, tips, calibration


def neighboring_values(value):
    middle = np.float32(value)
    return (float(np.nextafter(middle, np.float32(-math.inf))), float(middle),
            float(np.nextafter(middle, np.float32(math.inf))))


class ProjectionParityTests(unittest.TestCase):
    def compare(self, frame, tips, calibration=None):
        expected = original_projection(frame, tips, calibration)
        actual = current.project_positions(frame, tips, calibration)
        self.assertEqual(actual, expected)
        for position in actual['positions']:
            self.assertIsNone(position['pressed'])
            self.assertIsNone(position['sounding'])
        return actual

    def test_random_perspective_nonuniform_frets_and_calibrated_notes(self):
        for seed in range(64):
            count = (5, 9, 17, 25)[seed % 4]
            frame, tips, calibration = fixture(seed, count)
            for reverse_strings, reverse_frets in ((False, False), (True, False), (False, True), (True, True)):
                calibration.update(strings_reversed=reverse_strings, frets_reversed=reverse_frets,
                                   first_boundary_fret=count-1 if reverse_frets else 0)
                with self.subTest(seed=seed, strings=reverse_strings, frets=reverse_frets):
                    self.compare(frame, tips, calibration)

    def test_string_distance_and_ambiguity_threshold_neighbors(self):
        frame, _, calibration = fixture(perspective=False)
        for threshold in (.35, .55):
            for sign in (-1, 1):
                for y in neighboring_values(.42 + sign*threshold*.16):
                    with self.subTest(threshold=threshold, y=y):
                        self.compare(frame, dict(index=dict(x=.4, y=y)), calibration)

    def test_fret_fraction_and_exact_wire_boundary_neighbors(self):
        frame, _, calibration = fixture(frets=5, perspective=False)
        left, right = frame['frets'][1]['start']['x'], frame['frets'][2]['start']['x']
        for fraction in (0., .07, .93, 1.):
            for x in neighboring_values(left + fraction*(right-left)):
                with self.subTest(fraction=fraction, x=x):
                    self.compare(frame, dict(index=dict(x=x, y=.42)), calibration)

    def test_four_point_perspective_batch_preserves_threshold_neighbors(self):
        for seed in range(24):
            frame, _, calibration = fixture(seed, frets=5, perspective=False)
            warped, _, _ = fixture(seed)
            quad = np.float32([[p['x'], p['y']] for p in warped['corners']])
            transform = cv2.getPerspectiveTransform(np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]), quad)

            def warp(point):
                x, y = cv2.perspectiveTransform(np.float32([[[point['x'], point['y']]]]), transform)[0, 0]
                return dict(x=float(x), y=float(y))

            left, right = (frame['frets'][i]['start']['x'] for i in (1, 2))
            frame['corners'] = warped['corners']
            for wire in frame['strings']+frame['frets']:
                wire['start'], wire['end'] = warp(wire['start']), warp(wire['end'])
            for fraction in (.07, .93):
                for distance in (.35, .55):
                    for x in neighboring_values(left+fraction*(right-left)):
                        for y in neighboring_values(.42+distance*.16):
                            tip = warp(dict(x=x, y=y))
                            self.compare(frame, {name: dict(tip) for name in
                                                 ('index', 'middle', 'ring', 'little')}, calibration)

    def test_numpy_one_scalar_promotion_matches_explicit_double_reference(self):
        cases = [fixture(seed, (5, 9, 17, 25)[seed % 4]) for seed in range(24)]
        frame, _, calibration = fixture(frets=5, perspective=False)
        left, right = (frame['frets'][i]['start']['x'] for i in (1, 2))
        for fraction in (.07, .93):
            for distance in (.35, .55):
                for x in neighboring_values(left+fraction*(right-left)):
                    for y in neighboring_values(.42+distance*.16):
                        cases.append((frame, dict(index=dict(x=x, y=y)), calibration))
        for frame, tips, calibration in cases:
            expected = original_projection(frame, tips, calibration, legacy_promotion=True)
            with mock.patch.object(current.np, 'result_type', return_value=np.dtype(np.float64)):
                actual = current.project_positions(frame, tips, calibration)
            self.assertEqual(actual, expected)

    def test_mixed_invalid_tips_and_empty_or_searching_frames_keep_diagnostics(self):
        frame, tips, calibration = fixture()
        tips['index'] = dict(x=math.nan, y=.3)
        tips['ring'] = dict(x=False, y=.4)
        self.compare(frame, tips, calibration)
        self.compare(frame, {}, calibration)
        self.compare(frame, None, calibration)
        for change in (dict(status='searching'), dict(corners=None), dict(strings=[]), dict(frets=[])):
            self.compare(dict(frame, **change), tips, calibration)
        frame, tips, calibration = fixture()
        before = copy.deepcopy((frame, tips, calibration))
        self.compare(frame, tips, calibration)
        self.assertEqual((frame, tips, calibration), before)

    def test_board_margin_spacing_and_span_threshold_neighbors(self):
        frame, _, calibration = fixture(perspective=False)
        frame['corners'] = [dict(x=x, y=y) for x, y in ((0., .2), (1., .2), (1., .8), (0., .8))]
        for wire in frame['strings']+frame['frets']:
            for key in ('start', 'end'):
                wire[key]['y'] = .2 + .6*wire[key]['y']
        for y in (*neighboring_values(.2-.03*.6), *neighboring_values(.2+1.03*.6)):
            self.compare(frame, dict(index=dict(x=.4, y=y)), calibration)
        for x in (0., 1., *neighboring_values(0.), *neighboring_values(1.)):
            self.compare(frame, dict(index=dict(x=x, y=.5)), calibration)
        for span in neighboring_values(1e-6):
            changed, tips, _ = fixture(perspective=False)
            changed['frets'][2]['start']['y'] = 0.
            changed['frets'][2]['end']['y'] = span
            self.compare(changed, tips)
        for spacing in neighboring_values(.02):
            changed, _, _ = fixture(perspective=False)
            for i, wire in enumerate(changed['strings']):
                wire['start']['y'] = wire['end']['y'] = .25+i*spacing
            self.compare(changed, dict(index=dict(x=.4, y=.25)))

    def test_invalid_tips_geometry_and_calibration_keep_protocol(self):
        frame, tips, calibration = fixture()
        invalid = (None, {}, dict(x=.4), dict(x='bad', y=.4), dict(x=True, y=.4),
                   dict(x=np.float32(.4), y=.4), dict(x=math.nan, y=.4),
                   dict(x=.4, y=math.inf), dict(x=-.01, y=.4), dict(x=.4, y=1.01))
        for tip in invalid:
            self.compare(frame, dict(index=tip, unknown=dict(x=.4, y=.4)), calibration)
        for change in ('nan', 'range', 'duplicate_fret', 'crossed_strings', 'flat_fret', 'unobserved', 'quad'):
            changed = copy.deepcopy(frame)
            if change == 'nan':
                changed['frets'][0]['start']['x'] = math.nan
            elif change == 'range':
                changed['strings'][0]['start']['x'] = -1.
            elif change == 'duplicate_fret':
                changed['frets'][1] = copy.deepcopy(changed['frets'][0])
            elif change == 'crossed_strings':
                changed['strings'][0]['end'], changed['strings'][1]['end'] = (
                    changed['strings'][1]['end'], changed['strings'][0]['end'])
            elif change == 'flat_fret':
                changed['frets'][0]['end'] = copy.deepcopy(changed['frets'][0]['start'])
            elif change == 'unobserved':
                changed['strings'][0]['observed'] = False
            else:
                changed['corners'][2] = copy.deepcopy(changed['corners'][0])
            self.compare(changed, tips, calibration)
        for changes in (dict(stream_id='other'), dict(first_boundary_fret=24),
                        dict(frets_reversed=True, first_boundary_fret=0)):
            self.compare(frame, tips, dict(calibration, **changes))
        for changes in (dict(first_boundary_fret=True), dict(tuning=[40]*5)):
            with self.assertRaises(ValueError):
                current.project_positions(frame, tips, dict(calibration, **changes))

    def test_hand_observations_order_metadata_and_tracked_wire_guards(self):
        frame, tips, calibration = fixture()
        hands = []
        for i in range(2):
            landmarks = [dict(x=0., y=0.) for _ in range(21)]
            for index, tip in zip((8, 12, 16, 20), tips.values()):
                landmarks[index] = tip
            hands.append(dict(score=.9, tracked_sequence=12, landmarks=landmarks,
                joint_valid=[True]*21, fresh_for_position_candidates=True,
                keypoint_source='optical_flow', detector_age_ms=100.+i))
        frame['hand_detection'] = dict(hands=hands)
        self.compare(frame, None, calibration)
        frame.update(stage='tracked_lattice', lattice_age_ms=600., flow=dict(inlier_ratio=.7))
        for wire in frame['strings']+frame['frets']:
            wire['observed'] = False
        self.compare(frame, None, calibration)
        frame['lattice_age_ms'] = 600.001
        self.compare(frame, None, calibration)


if __name__ == '__main__':
    unittest.main()
