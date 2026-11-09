"""Causal chord hysteresis, bounded sustain holds and modal compatibility."""

from collections import deque
import math

from audio_music import CHORD_TYPES, NOTE_NAMES, recognize_chord


MODES = (
    ("major", (0, 2, 4, 5, 7, 9, 11)),
    ("dorian", (0, 2, 3, 5, 7, 9, 10)),
    ("phrygian", (0, 1, 3, 5, 7, 8, 10)),
    ("lydian", (0, 2, 4, 6, 7, 9, 11)),
    ("mixolydian", (0, 2, 4, 5, 7, 9, 10)),
    ("natural_minor", (0, 2, 3, 5, 7, 8, 10)),
    ("locrian", (0, 1, 3, 5, 6, 8, 10)),
)
SCALE_LIMIT = ("Recent pitch classes rank compatible collections, not a proven key. "
               "All seven modes sharing one collection remain alternatives; "
               "tonic requires musical context. Scores are not probabilities.")


class MusicTracker:
    """Count newly observed pitch-class events, never fabricated dropped frames.

    A held chord supplies one event, rather than dominating the key estimate
    through repeated polling. Silence clears chord confirmation immediately.
    Recent melody history survives a short rest and expires by monotonic time.
    """

    def __init__(self, window_seconds=12.0, confirmation_gap=2.0,
                 hold_seconds=0.75, sustain_seconds=1.8, switch_seconds=0.22):
        if not math.isfinite(window_seconds) or window_seconds <= 0:
            raise ValueError("window_seconds must be positive and finite")
        if not math.isfinite(confirmation_gap) or confirmation_gap <= 0:
            raise ValueError("confirmation_gap must be positive and finite")
        for value in (hold_seconds, sustain_seconds, switch_seconds):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Tracking durations must be positive and finite")
        self.window_seconds = float(window_seconds)
        self.confirmation_gap = float(confirmation_gap)
        self.hold_seconds = float(hold_seconds)
        self.sustain_seconds = float(sustain_seconds)
        self.switch_seconds = float(switch_seconds)
        grouped = {}
        for root in range(12):
            for name, intervals in MODES:
                pitches = tuple(sorted((root + step) % 12 for step in intervals))
                grouped.setdefault(pitches, []).append({"tonic": NOTE_NAMES[root], "scale": name})
        self._collections = grouped
        order = {name: index for index, name in enumerate(
            ("major", "natural_minor", "dorian", "phrygian", "lydian", "mixolydian", "locrian"))}
        for interpretations in self._collections.values():
            interpretations.sort(key=lambda item: order[item["scale"]])
        self.reset()

    def reset(self):
        self._events = deque()
        self._last_sequence = None
        self._last_now = None
        self._context = None
        self._pitch_signature = ()
        self._clear_chord()

    def _clear_chord(self):
        self._pending_label = None
        self._confirmations = 0
        self._pending_since = None
        self._pending_attack = False
        self._pending_required = 2
        self._pending_duration = 0.0
        self._chord_at = None
        self._supported_at = None
        self._stable_rms = None
        self._previous_rms = None
        self._required = 2
        self._tracking_status = "none"
        self._hold_reason = None
        self._stable = recognize_chord([])

    def _clear_pending(self):
        self._pending_label = None
        self._pending_since = None
        self._confirmations = 0
        self._pending_attack = False

    @staticmethod
    def _rms(result):
        value = (result.get("spectrum") or {}).get("rms")
        if isinstance(value, (int, float)) and math.isfinite(value) and value > 0:
            return float(value)
        return None

    def _track_chord(self, result, raw, now):
        rms = self._rms(result)
        attack = (rms is not None and self._previous_rms is not None and
                  rms >= max(0.002, self._previous_rms * 1.7))
        self._previous_rms = rms
        self._chord_at = now
        self._hold_reason = None
        stable = self._stable.get("status") == "recognized"
        confidence = float(raw.get("confidence", 0.0))
        reliable = (raw.get("status") == "recognized" and
                    math.isfinite(confidence) and confidence >= 0.6)
        same = reliable and raw.get("label") == self._stable.get("label")
        if stable and same:
            self._stable = dict(raw)
            self._supported_at = now
            if rms is not None:
                self._stable_rms = max(rms, self._stable_rms or 0.0)
            self._clear_pending()
            self._tracking_status = "confirmed"
            return

        pitches = {int(note["midi_note"]) % 12 for note in result.get("notes", [])
                   if float(note.get("confidence", 0.0)) >= 0.6}
        stable_pitches = set()
        if stable:
            root = NOTE_NAMES.index(self._stable["root"])
            intervals = next(intervals for quality, _, intervals in CHORD_TYPES
                             if quality == self._stable["quality"])
            stable_pitches = {(root + interval) % 12 for interval in intervals}
        # Equivalent chord names lower the naming score, even when every
        # observed tone is reliable. Keep the aliases while allowing a change.
        strong_transition = (result.get('visual_transition') is True and reliable and
            (confidence >= .8 or raw.get('ambiguous') is True and bool(result.get('notes')) and
             all(note.get('confidence',0) >= .8 for note in result['notes'])))
        decaying_subset = (stable and bool(pitches) and pitches < stable_pitches and
                           rms is not None and self._stable_rms is not None and
                           rms < self._stable_rms * 0.7 and not attack and
                           not (raw.get("label") == self._pending_label and self._pending_attack) and
                           not strong_transition)
        same_harmony = (stable and reliable and
                        (raw.get("root"), raw.get("quality")) ==
                        (self._stable.get("root"), self._stable.get("quality")))

        if decaying_subset or not reliable:
            self._clear_pending()
            limit = self.sustain_seconds if decaying_subset else self.hold_seconds
            if stable and now - self._supported_at <= limit:
                self._tracking_status = "holding"
                self._hold_reason = "decaying_partial_chord" if decaying_subset else "uncertain_frame"
            else:
                self._clear_chord()
            return

        label = raw["label"]
        rapid_change = (stable and not same_harmony and not decaying_subset and
            strong_transition and
            bool(pitches != stable_pitches))
        if label != self._pending_label:
            self._pending_label = label
            self._pending_since = now
            self._confirmations = 0
            self._pending_attack = attack
            self._pending_required = 4 if same_harmony else 3 if stable else 2
            self._pending_duration = (max(0.5, self.switch_seconds) if same_harmony else
                                      self.switch_seconds if stable else 0)
        if rapid_change:
            # Current visual motion only lowers confirmation delay; it cannot
            # invent notes, bypass two independent audio frames, or label pressure.
            self._pending_required = min(self._pending_required,2)
            self._pending_duration = min(self._pending_duration,.08)
        self._confirmations += 1
        self._required = self._pending_required
        if self._confirmations >= self._required and now - self._pending_since >= self._pending_duration:
            self._stable = dict(raw)
            self._supported_at = now
            self._stable_rms = rms
            self._tracking_status = "confirmed"
        elif stable and now - self._supported_at <= self.hold_seconds:
            self._tracking_status = "switching"
            self._hold_reason = "inversion_confirmation" if same_harmony else "new_chord_confirmation"
        else:
            # Preserve candidate votes while expiring the unsupported old label.
            self._stable = recognize_chord([])
            self._tracking_status = "pending"

    def _scales(self, now, active):
        while self._events and now - self._events[0][0] > self.window_seconds:
            self._events.popleft()
        weights = [0.0] * 12
        for timestamp, event in self._events:
            decay = math.exp(-(now - timestamp) / (self.window_seconds / 2))
            for pitch, weight in event.items():
                weights[pitch] += weight * decay
        total = sum(weights)
        observed = [pitch for pitch, weight in enumerate(weights) if weight > 0]
        report = {"status": "insufficient", "observed_pitch_classes":
                  [NOTE_NAMES[pitch] for pitch in observed], "candidates": [], "groups": [],
                  "window_seconds": self.window_seconds, "event_count": len(self._events),
                  "history_active": active, "limitation": SCALE_LIMIT}
        if len(observed) < 3 or total <= 0:
            return report
        groups = []
        for pitches, interpretations in self._collections.items():
            inside = sum(weights[pitch] for pitch in pitches) / total
            coverage = sum(pitch in observed for pitch in pitches) / 7.0
            # Outliers reduce compatibility; they never silently establish a key.
            if inside < 0.88:
                continue
            score = inside * (0.65 + 0.35 * coverage)
            groups.append({"pitch_classes": [NOTE_NAMES[pitch] for pitch in pitches],
                           "coverage": round(coverage, 4), "score": round(score, 4),
                           "outside_weight": round(1.0 - inside, 4),
                           "modes": [dict(item) for item in interpretations]})
        groups.sort(key=lambda item: (-item["score"], item["pitch_classes"]))
        report["compatible_group_count"] = len(groups)
        report["groups"] = groups[:3]
        for group in report["groups"]:
            for mode in group["modes"]:
                report["candidates"].append(dict(mode, coverage=group["coverage"],
                                                  score=group["score"], ambiguous=True))
        report["status"] = "ambiguous" if groups else "mixed_or_chromatic"
        return report

    def update(self, result, now, context=None):
        if not math.isfinite(now):
            raise ValueError("now must be finite monotonic seconds")
        sequence = result.get("sequence")
        reset_needed = (self._last_now is not None and
                        (now < self._last_now or context != self._context or
                         now - self._last_now > self.confirmation_gap))
        if reset_needed:
            self.reset()
        self._context = context
        duplicate = sequence is not None and sequence == self._last_sequence
        raw = result.get("chord") or recognize_chord([])
        active = (result.get("status") in ("recognized", "unknown") and
                  bool(result.get("signal_active")))
        if not active:
            self._clear_chord()
            self._pitch_signature = ()
        elif not duplicate:
            if self._chord_at is not None and now - self._chord_at > self.confirmation_gap:
                self._clear_chord()
            self._track_chord(result, raw, float(now))
            event = {}
            for note in result.get("notes", []):
                confidence = float(note.get("confidence", 0.0))
                strength = float(note.get("relative_strength", 1.0))
                if not (math.isfinite(confidence) and math.isfinite(strength)):
                    continue
                if confidence < 0.65 or strength < 0.15:
                    continue
                pitch = int(note["midi_note"]) % 12
                event[pitch] = max(event.get(pitch, 0.0), confidence * min(1.0, strength))
            signature = tuple(sorted(event))
            if signature and signature != self._pitch_signature:
                self._events.append((float(now), event))
            self._pitch_signature = signature
        if duplicate and self._chord_at is not None and now - self._chord_at > self.confirmation_gap:
            self._clear_chord()
        elif (duplicate and self._tracking_status in ("holding", "switching") and
              self._supported_at is not None and now - self._supported_at >
              (self.sustain_seconds if self._hold_reason == "decaying_partial_chord" else self.hold_seconds)):
            self._stable = recognize_chord([])
            self._tracking_status = "pending" if self._pending_label else "none"
        self._last_sequence = sequence
        self._last_now = float(now)
        return dict(result, raw_chord=dict(raw), stable_chord=dict(self._stable),
                    tracking={"chord_status": self._tracking_status, "confirmations": self._confirmations,
                              "required_observations": self._required, "duplicate_sequence": duplicate,
                              "confirmation_gap_seconds": self.confirmation_gap,
                              "candidate_label": self._pending_label,
                              "hold_reason": self._hold_reason,
                              "hold_seconds": self.hold_seconds, "sustain_seconds": self.sustain_seconds,
                              "switch_seconds": self.switch_seconds,
                              "support_age_seconds": None if self._supported_at is None else
                              round(float(now) - self._supported_at, 4)},
                    scale_evidence=self._scales(float(now), active))
