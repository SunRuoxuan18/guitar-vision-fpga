"""Project four-tip results through their source-frame measured fretboard."""
from collections import OrderedDict
from copy import deepcopy
import math

from visual_positions import project_positions


class FingertipNotes:
    def __init__(self, max_frames=64, max_age_ms=8000):
        self.frames = OrderedDict()
        self.identity = None
        self.max_frames = max_frames
        self.max_age_ms = max_age_ms

    def remember(self, analysis, calibration=None):
        if (type(analysis.get('sequence')) is not int or
                not isinstance(analysis.get('stream_id'), str) or
                type(analysis.get('generation')) is not int):
            return
        identity = (analysis.get('stream_id'), analysis.get('generation'))
        if identity != self.identity:
            self.frames.clear()
            self.identity = identity
        keys = ('status', 'stream_id', 'generation', 'sequence', 'lattice_epoch',
                'stage', 'lattice_age_ms', 'flow', 'strings', 'frets', 'corners')
        frame = {k: deepcopy(analysis[k]) for k in keys if k in analysis}
        self.frames[analysis['sequence']] = (frame, deepcopy(calibration))
        while len(self.frames) > self.max_frames:
            self.frames.popitem(last=False)

    def project(self, neural, analysis, tracking=None):
        output = dict(status='unavailable', sequence=analysis.get('sequence'),
            stream_id=analysis.get('stream_id'), generation=analysis.get('generation'),
            source_sequence=None, age_ms=None, projection=None,
            reason='no_neural_result', pressed_strings_known=False, sounding_strings_known=False)
        if isinstance(neural, dict) and (neural.get('status') == 'fault' or neural.get('fault')):
            return dict(output, reason='neural_fault')
        if isinstance(tracking, dict) and tracking.get('status') == 'tracking':
            if (all(tracking.get(k) == analysis.get(k) for k in ('stream_id','generation','sequence'))
                    and tracking.get('keypoint_source') in ('optical_flow', 'neural')
                    and type(tracking.get('detector_age_ms')) in (int,float)
                    and math.isfinite(tracking['detector_age_ms'])
                    and 0 <= tracking['detector_age_ms'] <= 4000):
                points = tracking.get('points') or {}
                tips = {name: dict(x=p['x']/640., y=p['y']/480.) for name,p in points.items()
                    if name in ('index','middle','ring','little') and isinstance(p,dict)
                    and p.get('visible') is not False
                    and all(type(p.get(k)) in (int,float) and math.isfinite(p[k]) for k in ('x','y'))
                    and 0 <= p['x'] < 640 and 0 <= p['y'] < 480}
                remembered = self.frames.get(analysis['sequence'])
                if remembered is not None and tips:
                    geometry, calibration = remembered
                    projection = project_positions(geometry, tips=tips, calibration=calibration)
                    for position in projection['positions']:
                        position['keypoint_source'] = tracking['keypoint_source']
                        position['neural_age_ms'] = tracking['detector_age_ms']
                    return dict(output, status='current', source_sequence=analysis['sequence'],
                        detector_sequence=tracking.get('source_sequence'), age_ms=0.,
                        projection=projection, reason='current_frame_independent_tip_flow',
                        lattice_epoch=geometry.get('lattice_epoch'))
        if not isinstance(neural, dict):
            return output
        source = neural if neural.get('status') == 'ok' else neural.get('last_result')
        if not isinstance(source, dict) or source.get('fault'):
            return output
        if any(source.get(k) != analysis.get(k) for k in ('stream_id', 'generation')):
            output['reason'] = 'neural_source_mismatch'
            return output
        sequence, age = source.get('sequence'), source.get('age_ms')
        if (type(sequence) is not int or type(age) not in (int, float) or
                not math.isfinite(age) or not 0 <= age <= self.max_age_ms or
                sequence > analysis['sequence']):
            output['reason'] = 'neural_result_expired_or_invalid'
            return output
        output.update(source_sequence=sequence, age_ms=age)
        remembered = self.frames.get(sequence)
        if remembered is None:
            output['reason'] = 'source_frame_geometry_unavailable'
            return output
        points = source.get('points')
        if not isinstance(points, dict):
            return output
        tips = {}
        for name in ('index', 'middle', 'ring', 'little'):
            p = points.get(name)
            if (isinstance(p, dict) and p.get('visible') is not False and
                    all(type(p.get(k)) in (int, float) and math.isfinite(p[k])
                        for k in ('x', 'y'))):
                tips[name] = dict(x=p['x']/640., y=p['y']/480.)
        geometry, calibration = remembered
        projection = project_positions(geometry, tips=tips, calibration=calibration)
        current = neural.get('status') == 'ok' and sequence == analysis['sequence']
        output.update(status='current' if current else 'historical', projection=projection,
            reason='source_frame_projection', lattice_epoch=geometry.get('lattice_epoch'))
        return output
