"""Board-side musical constraints with explicit relative-fret hypotheses.

An audio-inferred fret mapping is a hypothesis, never independent visual proof
of pressing or sounding a string. Association uses PS receipt time; capture
clocks are not synchronized by this module.
"""
import json
import math
import os
from pathlib import Path
import numpy as np
from audio_music import CHORD_TYPES, NOTE_NAMES, recognize_chord

TUNING = (40,45,50,55,59,64)  # Image line order is not yet string order.
_ALIGNMENTS = tuple((first, strings, frets) for strings in (False, True)
                    for frets in (False, True) for first in range(25))
_FIRST = np.array([a[0] for a in _ALIGNMENTS], dtype=np.int16)[:, None]
_STRINGS_REVERSED = np.array([a[1] for a in _ALIGNMENTS])[:, None]
_FRETS_REVERSED = np.array([a[2] for a in _ALIGNMENTS])[:, None]
MAX_ASSOCIATION_SKEW = .3


def confident_midis(audio):
    return {note['midi_note'] for note in audio.get('notes', [])
        if isinstance(note, dict) and type(note.get('midi_note')) is int
        and type(note.get('confidence')) in (int, float)
        and math.isfinite(note['confidence']) and note['confidence'] >= .6
        and not note.get('octave_ambiguous')}


def associated_observation(observed, visual, now):
    # Both stamps use PS monotonic time. Neither is sensor exposure time.
    return (type(observed) in (int, float) and math.isfinite(observed) and
            observed <= now and
            abs(observed-visual['received_monotonic']) <= MAX_ASSOCIATION_SKEW)


def publish_visual(path, analysis, received_at):
    if not path:
        return
    report = dict(received_monotonic=received_at, sequence=analysis.get('sequence'),
        stream=analysis.get('stream_id'), generation=analysis.get('generation'),
        lattice_epoch=analysis.get('lattice_epoch'),
        fret_count=len(analysis.get('frets',[])),
        positions=analysis.get('visual_positions',{}).get('positions',[]),
        position_reason=analysis.get('visual_positions',{}).get('reason'),
        hand_motion=analysis.get('hand_detection',{}).get('motion',{}),
        processing_completed_monotonic=analysis.get('processing_completed_monotonic'))
    # Only geometry needed for fusion crosses the atomic exchange.
    target = Path(path)
    temporary = target.with_name(target.name+'.'+str(os.getpid())+'.tmp')
    temporary.write_text(json.dumps(report,allow_nan=False,separators=(',',':')))
    os.replace(str(temporary),str(target))


def read_visual(path):
    if not path:
        return None
    try:
        with open(path,'rb') as handle:
            raw = handle.read(32769)
        result = json.loads(raw) if len(raw) <= 32768 else None
        return result if isinstance(result, dict) else None
    except (OSError,ValueError):
        return None


def visual_age(visual, now):
    """Bound both input age and result age, using only the shared PS clock."""
    if not isinstance(visual,dict):
        return None
    received = visual.get('received_monotonic')
    if type(received) not in (int,float) or not math.isfinite(received) or received > now:
        return None
    completed = visual.get('processing_completed_monotonic')
    if completed is None:
        return now-received if now-received <= .3 else None
    if (type(completed) not in (int,float) or not math.isfinite(completed) or
            not received <= completed <= now or now-completed > .3 or now-received > .75):
        return None
    return now-received


def _fuse(audio, visual, now):
    output = dict(status='audio_only',processing_location='ps_linux',
        chord=audio.get('stable_chord') or audio.get('chord'),
        matches=[],mapping_hypotheses=[],positions=[],
        synchronized=False,time_association='board_receipt_monotonic',
        pressed_strings_known=False,sounding_strings_known=False,
        voicing_midi_notes=sorted(confident_midis(audio)),
        absolute_frets_known=False)
    if not visual or not isinstance(visual,dict):
        output['reason'] = 'vision_unavailable'
        return output
    age = visual_age(visual,now)
    if age is None:
        output['reason'] = 'vision_stale'
        return output
    output['vision_age_ms'] = round(age*1000,2)
    output['vision_processing_age_ms'] = (round((now-visual['processing_completed_monotonic'])*1000,2)
        if visual.get('processing_completed_monotonic') is not None else None)
    output['visual_sequence'] = visual.get('sequence')
    output['position_reason'] = visual.get('position_reason')
    output['visual_identity'] = [visual.get('stream'), visual.get('generation'), visual.get('lattice_epoch')]
    if (audio.get('status') != 'recognized' or audio.get('stale') or
            (audio.get('fft') or {}).get('analysis_test_mode',0) != 0):
        output['reason'] = 'live_audio_unavailable'
        return output
    if 'observed_monotonic' in audio:
        if not associated_observation(audio['observed_monotonic'], visual, now):
            output['reason'] = 'audio_visual_time_mismatch'
            return output
        output['association_skew_ms'] = round(
            (visual['received_monotonic']-audio['observed_monotonic'])*1000, 2)
    notes = confident_midis(audio)
    positions = [p for p in visual.get('positions',[]) if not p.get('ambiguous')]
    positions = [p for p in positions if type(p.get('hand_index',0)) is int]
    output['positions'] = positions
    absolute = [p for p in positions if not p.get('relative') and type(p.get('midi_note')) is int]
    output['absolute_projected_positions'] = [dict(p,audio_supported=p['midi_note'] in notes) for p in absolute]
    output['matches'] = [dict(p,audio_supported=True) for p in absolute if p['midi_note'] in notes]
    if output['matches']:
        output.update(status='audio_visual_supported',reason='absolute_positions_match_current_audio')
        return output
    relative = [p for p in positions if p.get('relative') and
        type(p.get('string_line')) is int and 1 <= p['string_line'] <= 6 and
        type(p.get('fret_interval')) is int and 1 <= p['fret_interval'] <= 24]
    # At least two different fingers are needed to constrain any mapping.
    if len({(p.get('hand_index',0), p.get('finger')) for p in relative}) < 2 or not notes:
        output['reason'] = 'insufficient_current_positions'
        return output
    fret_count = visual.get('fret_count',0)
    if type(fret_count) is not int or not 5 <= fret_count <= 25:
        output['reason'] = 'lattice_unavailable'
        return output
    # Enumerate all 0..24-fret alignments together, rather than constructing
    # every Python position for each rejected alignment on every audio poll.
    relative = list({(p.get('hand_index',0),p.get('finger'),p['string_line'],p['fret_interval']):p
                     for p in relative}.values())
    line = np.array([p['string_line'] for p in relative], dtype=np.int16)[None, :]
    cell = np.array([p['fret_interval'] for p in relative], dtype=np.int16)[None, :]
    index = np.where(_STRINGS_REVERSED, 6-line, line-1)
    interval = np.where(_FRETS_REVERSED, fret_count-cell, cell)
    frets = _FIRST+interval
    valid = (interval >= 1) & (interval < fret_count) & (frets >= 1) & (frets <= 24)
    midis = np.asarray(TUNING, dtype=np.int16)[index]+frets
    supported = valid & np.isin(midis, list(notes))
    support = np.sum(supported, axis=1)
    # One hand/finger cannot supply two independent votes at adjacent cells.
    for row in np.flatnonzero(support >= 2):
        support[row] = len({(relative[col].get('hand_index',0), relative[col].get('finger'))
                            for col in np.flatnonzero(supported[row])})
    hypotheses = []
    best = int(support.max()) if len(support) else 0
    winners = np.flatnonzero(support == best) if best >= 2 else []
    for row in winners:
        first_fret, reversed_strings, reversed_frets = _ALIGNMENTS[row]
        mapped = [dict(relative[col], string=int(6-index[row,col]), fret=int(frets[row,col]),
                    midi_note=int(midis[row,col]), audio_supported=bool(supported[row,col]), relative=False,
                    mapping_source='audio_constrained_hypothesis', pressed=None, sounding=None)
                  for col in np.flatnonzero(valid[row])]
        hypotheses.append(dict(first_fret=first_fret,reversed_strings=reversed_strings,
            reversed_frets=reversed_frets,support=int(support[row]),positions=mapped))
    hypotheses.sort(key=lambda h:-h['support'])
    if hypotheses:
        best = hypotheses[0]['support']
        winners = [h for h in hypotheses if h['support'] == best]
        output.update(status='mapping_ambiguous' if len(winners)>1 else 'mapping_candidate',
            reason='relative_positions_constrain_audio_mapping',mapping_hypotheses=winners[:8],
            mapping_alternatives=len(winners),absolute_frets_known=False)
        if len(winners) == 1:
            candidate = winners[0]['positions']
            output['hand_fret_span'] = [min(p['fret'] for p in candidate), max(p['fret'] for p in candidate)]
    else:
        output['reason'] = 'no_supported_mapping'
    return output


def attach_joint_decision(result):
    """A joint verdict requires current position support, not an audio echo."""
    decision = dict(status='audio_only', chord=None, visual_participation=False,
                    complete_fingering_known=False, independent_visual_anchor=False,
                    pressed_strings_known=False, sounding_strings_known=False)
    chord = result.get('chord') or {}
    positions = (result.get('aligned_positions') or result.get('absolute_projected_positions')
                 or result.get('matches') or [])
    supported = [p for p in positions if p.get('audio_supported') is True]
    distinct = {(p.get('hand_index',0),p.get('finger')) for p in supported}
    projected = {(p.get('hand_index',0),p.get('finger')) for p in positions}
    if projected-distinct:
        decision.update(status='visual_audio_conflict',conflicting_fingers=len(projected-distinct))
        result['joint_decision'] = decision
        return result
    if len(distinct) >= 2 and chord.get('status') == 'recognized':
        tones = set(result.get('voicing_midi_notes', []))
        intervals = next((steps for quality,_,steps in CHORD_TYPES if quality == chord.get('quality')), None)
        if intervals is not None and chord.get('root') in NOTE_NAMES:
            root = NOTE_NAMES.index(chord['root'])
            expected = {(root+step) % 12 for step in intervals}
            if {m % 12 for m in tones} != expected:
                decision['status'] = 'awaiting_audio_transition'
            else:
                decision.update(status='audio_visual_consistent', chord=chord,
                                visual_participation=True, supported_fingers=len(distinct),
                                mapping_source=result.get('mapping_source','absolute_position_projection'))
    if decision['visual_participation']:
        # A subset of finger projections is useful, but is not all sounding strings.
        notes = [dict(midi_note=p['midi_note'],confidence=.7) for p in supported]
        decision['projected_notes_chord_candidate'] = recognize_chord(notes)
    result['joint_decision'] = decision
    return result


def fuse(audio, visual, now):
    return attach_joint_decision(_fuse(audio, visual, now))


def visual_transition(visual, now, audio_observed_at=None):
    """Current PL motion can shorten confirmation, never supply missing tones."""
    if not isinstance(visual, dict):
        return False
    motion = visual.get('hand_motion') or {}
    return (visual_age(visual,now) is not None and
            (audio_observed_at is None or associated_observation(audio_observed_at,visual,now)) and
            motion.get('changing') is True and motion.get('sequence') == visual.get('sequence') and
            motion.get('source') == 'current_pl_joint_flow')


class FretMappingTracker:
    """Learn a tentative grid alignment across different observed harmonies.

    Alignment comes from audio and visual positions together. It is not an
    independent nut detector or evidence of finger pressure/sounding strings.
    """
    def __init__(self):
        self.key = None
        self.candidates = None
        self.harmonies = set()
        self.anchor = None
        self.last_pair = None
        self.last_support = float('-inf')

    @staticmethod
    def identity(visual):
        if not isinstance(visual, dict) or type(visual.get('lattice_epoch')) is not int:
            return None
        return (visual.get('stream'), visual.get('generation'),
                visual['lattice_epoch'], visual.get('fret_count'))

    def update(self, audio, visual, now):
        key = self.identity(visual)
        if key != self.key or now-self.last_support > 5:
            self.key, self.candidates, self.anchor = key, None, None
            self.harmonies, self.last_pair = set(), None
        result = fuse(audio, visual, now)
        hypotheses = result.get('mapping_hypotheses', [])
        chord = audio.get('stable_chord') or audio.get('chord') or {}
        pair = (audio.get('sequence'), visual.get('sequence')) if visual else None
        if (key is not None and pair != self.last_pair and hypotheses and
                len(hypotheses) == result.get('mapping_alternatives') and
                chord.get('status') == 'recognized' and chord.get('confidence',0) >= .8):
            candidates = {(h['first_fret'], h['reversed_strings'], h['reversed_frets']) for h in hypotheses}
            common = candidates if self.candidates is None else self.candidates & candidates
            if not common:
                # A contradiction discards the old alignment, never drags it
                # onto a new fret window or a new hand configuration.
                self.harmonies, self.anchor = set(), None
                common = candidates
            self.candidates = common
            # Equivalent names (sus2/sus4, dim7 rotations) share one note set;
            # naming ambiguity does not erase measured absolute pitches.
            self.harmonies.add(tuple(sorted({m % 12 for m in confident_midis(audio)})))
            self.last_pair, self.last_support = pair, now
            if len(common) == 1 and len(self.harmonies) >= 2:
                self.anchor = (key, next(iter(common)), now)
        return self.apply_anchor(result, visual, now)

    def apply_anchor(self, result, visual, now):
        anchor = self.anchor
        if (anchor is None or self.identity(visual) != anchor[0] or
                not 0 <= now-anchor[2] <= 5 or not result.get('mapping_hypotheses')):
            return result
        for hypothesis in result['mapping_hypotheses']:
            signature = (hypothesis['first_fret'], hypothesis['reversed_strings'], hypothesis['reversed_frets'])
            if signature == anchor[1]:
                result.update(status='audio_visual_aligned',
                    reason='cross_harmony_audio_visual_grid_alignment',
                    aligned_positions=hypothesis['positions'],
                    mapping_source='audio_visual_cross_harmony_hypothesis',
                    independent_visual_anchor=False, absolute_frets_known=False,
                    hand_fret_span=[min(p['fret'] for p in hypothesis['positions']),
                                    max(p['fret'] for p in hypothesis['positions'])])
                break
        return attach_joint_decision(result)
