"""Guitar-oriented polyphonic inference from one-sided FFT power spectra.

NumPy is the only dependency. Hardware access belongs to audio_fft.py.
"""

from dataclasses import dataclass
from typing import Dict, Optional
import os

import numpy as np


NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
CHORD_TYPES = (
    ("major", "", (0, 4, 7)),
    ("minor", "m", (0, 3, 7)),
    ("diminished", "dim", (0, 3, 6)),
    ("augmented", "aug", (0, 4, 8)),
    ("dominant_seventh", "7", (0, 4, 7, 10)),
    ("major_seventh", "maj7", (0, 4, 7, 11)),
    ("minor_seventh", "m7", (0, 3, 7, 10)),
    ("minor_major_seventh", "m(maj7)", (0, 3, 7, 11)),
    ("half_diminished_seventh", "m7b5", (0, 3, 6, 10)),
    ("diminished_seventh", "dim7", (0, 3, 6, 9)),
    ("suspended_second", "sus2", (0, 2, 7)),
    ("suspended_fourth", "sus4", (0, 5, 7)),
    ("dominant_seventh_sus2", "7sus2", (0, 2, 7, 10)),
    ("dominant_seventh_sus4", "7sus4", (0, 5, 7, 10)),
    ("power", "5", (0, 7)),
    ("major_sixth", "6", (0, 4, 7, 9)),
    ("minor_sixth", "m6", (0, 3, 7, 9)),
    ("added_ninth", "add9", (0, 4, 7, 2)),
    ("minor_added_ninth", "m(add9)", (0, 3, 7, 2)),
    ("dominant_ninth", "9", (0, 4, 7, 10, 2)),
    ("major_ninth", "maj9", (0, 4, 7, 11, 2)),
    ("minor_ninth", "m9", (0, 3, 7, 10, 2)),
    ("added_eleventh", "add11", (0, 4, 7, 5)),
    ("minor_added_eleventh", "m(add11)", (0, 3, 7, 5)),
    ("sixth_added_ninth", "6/9", (0, 4, 7, 9, 2)),
    ("minor_sixth_added_ninth", "m6/9", (0, 3, 7, 9, 2)),
    ("dominant_seventh_flat_ninth", "7b9", (0, 4, 7, 10, 1)),
    ("dominant_seventh_sharp_ninth", "7#9", (0, 4, 7, 10, 3)),
    ("dominant_seventh_flat_fifth", "7b5", (0, 4, 6, 10)),
    ("dominant_seventh_sharp_fifth", "7#5", (0, 4, 8, 10)),
    ("dominant_ninth_suspended_fourth", "9sus4", (0, 5, 7, 10, 2)),
    ("dominant_eleventh", "11", (0, 4, 7, 10, 2, 5)),
    ("major_eleventh", "maj11", (0, 4, 7, 11, 2, 5)),
    ("minor_eleventh", "m11", (0, 3, 7, 10, 2, 5)),
    ("dominant_ninth_sharp_eleventh", "9#11", (0, 4, 7, 10, 2, 6)),
    ("dominant_seventh_added_thirteenth", "7add13", (0, 4, 7, 10, 9)),
)
# Index pitch-class sets once; naming stays independent of register and voicing.
_CHORD_INDEX = {}
for _root in range(12):
    for _quality, _suffix, _intervals in CHORD_TYPES:
        _mask = sum(1 << ((_root+step) % 12) for step in _intervals)
        _CHORD_INDEX.setdefault(_mask, []).append((_root, _quality, _suffix, _intervals))
LIMITATIONS = (
    "A single 4096-sample frame at 48 kHz spans 85.3 ms; its bins are 11.72 Hz apart.",
    "Confidence is a fit heuristic, not a calibrated probability.",
    "Exact octave doublings and missing partials can be indistinguishable from harmonics.",
    "Distortion, transients, inharmonic strings, weak notes and other instruments reduce reliability.",
    "Chord names describe complete observed pitch-class sets; key and root intent can be ambiguous.",
)


def midi_frequency(midi_note: int, cents: float = 0.0) -> float:
    return float(440.0 * 2.0 ** ((midi_note - 69 + cents / 100.0) / 12.0))


def note_name(midi_note: int) -> str:
    return "{}{}".format(NOTE_NAMES[midi_note % 12], midi_note // 12 - 1)


@dataclass(frozen=True)
class MusicConfig:
    sample_rate: float = 48000.0
    fft_size: int = 4096
    min_midi: int = 40
    max_midi: int = 88
    max_notes: int = 6
    max_harmonics: int = 10
    max_frequency: float = 8000.0
    window: str = "hann"
    # Hardware bins represent squared int16 FFT magnitudes divided by power_scale.
    # Normalized PC spectra already include the N**2 divisor and use no scale.
    power_scale: Optional[float] = None
    minimum_rms: float = 0.001
    # Units are the supplied FFT power units, which differ between PCM and PL.
    silence_power: float = 1e-10
    minimum_explained: float = 0.55
    minimum_note_gain: float = 0.025
    minimum_relative_strength: float = 0.12
    infer_scales: bool = False

    def __post_init__(self):
        if self.fft_size < 64 or self.fft_size % 2 or self.sample_rate <= 0:
            raise ValueError("An even FFT size >= 64 and positive sample rate are required")
        if not 0 <= self.min_midi <= self.max_midi <= 127:
            raise ValueError("Invalid MIDI search range")
        if not 1 <= self.max_notes <= 12 or not 1 <= self.max_harmonics <= 32:
            raise ValueError("Invalid note or harmonic limit")
        if not 0 < self.max_frequency <= self.sample_rate / 2:
            raise ValueError("Invalid analysis frequency limit")
        if self.window not in ("hann", "rectangular"):
            raise ValueError("window must be hann or rectangular")
        if self.power_scale is not None and (not np.isfinite(self.power_scale) or self.power_scale <= 0):
            raise ValueError("power_scale must be positive and finite")
        if not np.isfinite(self.minimum_rms) or not 0 <= self.minimum_rms <= 1:
            raise ValueError("minimum_rms must be normalized to full-scale PCM in [0, 1]")
        if not np.isfinite(self.silence_power) or self.silence_power < 0:
            raise ValueError("silence_power must be finite and nonnegative")
        for value in (self.minimum_explained, self.minimum_note_gain,
                      self.minimum_relative_strength):
            if not 0 < value <= 1:
                raise ValueError("Fit thresholds must be in (0, 1]")


def spectrum_from_samples(samples, config: Optional[MusicConfig] = None):
    """Normalized power in the configured window; omit Nyquist to match PL."""
    config = config or MusicConfig()
    samples = np.asarray(samples, dtype=np.float64)
    if samples.shape != (config.fft_size,) or not np.all(np.isfinite(samples)):
        raise ValueError("Expected one finite, real FFT-length PCM frame")
    window = np.hanning(config.fft_size) if config.window == "hann" else 1.0
    windowed = (samples - np.mean(samples)) * window
    return (np.abs(np.fft.rfft(windowed)) ** 2 / config.fft_size ** 2)[:-1]


def recognize_chord(notes) -> Dict:
    """Name complete pitch-class sets, preserving equivalent interpretations."""
    unknown = {"status": "unknown", "label": None, "confidence": 0.0,
               "root": None, "quality": None, "bass": None,
               "inversion": None, "ambiguous": False, "alternatives": []}
    if len(notes) < 2:
        return unknown
    pitches = {int(note["midi_note"]) % 12 for note in notes}
    bass = min(int(note["midi_note"]) for note in notes) % 12
    candidates = []
    mask = sum(1 << pitch for pitch in pitches)
    for root, quality, suffix, intervals in _CHORD_INDEX.get(mask, ()):
        inversion = intervals.index((bass - root) % 12)
        label = NOTE_NAMES[root] + suffix
        if bass != root:
            label += "/" + NOTE_NAMES[bass]
        candidates.append({"label": label, "root": NOTE_NAMES[root],
                           "quality": quality, "bass": NOTE_NAMES[bass],
                           "inversion": inversion})
    if not candidates:
        unknown["bass"] = NOTE_NAMES[bass]
        return unknown
    # Bass is only a naming preference; symmetrical/suspended alternatives remain visible.
    candidates.sort(key=lambda candidate: (candidate["inversion"] != 0,
                                            candidate["inversion"], candidate["label"]))
    confidence = float(np.mean([note.get("confidence", 0.0) for note in notes]))
    if len(candidates) > 1:
        confidence *= 0.85
    return dict(candidates[0], status="recognized", confidence=round(confidence, 4),
                ambiguous=len(candidates) > 1, alternatives=candidates[1:])


def scale_evidence(notes) -> Dict:
    pitches = {int(note["midi_note"]) % 12 for note in notes}
    candidates = []
    if len(pitches) >= 3:
        for root in range(12):
            for name, intervals in (("major", (0, 2, 4, 5, 7, 9, 11)),
                                    ("natural_minor", (0, 2, 3, 5, 7, 8, 10))):
                if pitches <= {(root + step) % 12 for step in intervals}:
                    candidates.append({"tonic": NOTE_NAMES[root], "scale": name,
                                       "coverage": round(len(pitches) / 7.0, 4)})
    return {"status": "ambiguous" if len(candidates) > 1 else
            "compatible" if candidates else "insufficient",
            "observed_pitch_classes": [NOTE_NAMES[pc] for pc in sorted(pitches)],
            "candidates": candidates,
            "limitation": "One frame supplies scale compatibility, not evidence of a tonal key."}


class MusicRecognizer:
    """Nonnegative harmonic-template pursuit with joint refitting.

    Each candidate explains its fundamental and up to ten harmonics together.
    Envelope and weak/missing-fundamental variants accommodate plucked strings.
    Joint nonnegative fits and a marginal-gain threshold suppress overtone echoes.
    """

    def __init__(self, config: Optional[MusicConfig] = None):
        self.config = config or MusicConfig()
        self._templates = None
        self._metadata = []
        self._octave_uncertain = set()
        self._native = None
        if os.environ.get("FRETLENS_AUDIO_NATIVE") == "1":
            from audio_native import AudioNative
            self._native = AudioNative()
        self._bin_hz = self.config.sample_rate / self.config.fft_size
        frequencies = np.arange(self.config.fft_size // 2) * self._bin_hz
        self._band = (frequencies >= 35.0) & (frequencies <= self.config.max_frequency)

    def _build_templates(self):
        config = self.config
        templates = []
        for midi in range(config.min_midi, config.max_midi + 1):
            for cents in (-25.0, 0.0, 25.0):
                frequency = midi_frequency(midi, cents)
                for decay in (0.7, 1.1, 1.7, 8.0):
                    for fundamental in (1.0, 0.25, 0.0):
                        if decay == 8.0 and fundamental != 1.0:
                            continue
                        template = np.zeros(config.fft_size // 2, dtype=np.float32)
                        for harmonic in range(1, config.max_harmonics + 1):
                            center = harmonic * frequency / self._bin_hz
                            if harmonic * frequency > config.max_frequency:
                                break
                            bins = np.arange(max(1, int(center) - 4),
                                             min(len(template), int(center) + 6))
                            offset = bins - center
                            # Continuous Hann main lobe and nearby sidelobes.
                            if config.window == "hann":
                                lobe = np.abs(np.sinc(offset) + 0.5 * np.sinc(offset - 1)
                                              + 0.5 * np.sinc(offset + 1))
                            else:
                                lobe = np.abs(np.sinc(offset))
                            amplitude = harmonic ** -decay
                            if harmonic == 1:
                                amplitude *= fundamental
                            template[bins] += amplitude * lobe
                        template[~self._band] = 0.0
                        norm = float(np.linalg.norm(template))
                        if norm == 0:
                            continue
                        templates.append(template / norm)
                        self._metadata.append((midi, cents, decay, fundamental))
        self._templates = np.asarray(templates, dtype=np.float32)
        self._midi_indices = np.asarray([item[0] for item in self._metadata])
        self._variants = {midi: np.flatnonzero(self._midi_indices == midi)
                         for midi in range(config.min_midi, config.max_midi+1)}
        self._full_fundamental = np.asarray([item[3] == 1.0 for item in self._metadata])
        # Missing fundamentals are possible, but mixtures often create false subharmonics.
        self._priors = np.asarray([1.0 if item[3] == 1.0 else
                                   0.94 if item[3] else 0.80
                                   for item in self._metadata], dtype=np.float32)
        # Keep every nonzero bin, including sidelobes; padding has zero weight.
        nonzero_bins = [np.flatnonzero(template) for template in self._templates]
        width = max((len(bins) for bins in nonzero_bins), default=0)
        self._template_bins = np.zeros((len(self._templates), width), dtype=np.intp)
        self._template_values = np.zeros(self._template_bins.shape, dtype=np.float32)
        for row, bins in enumerate(nonzero_bins):
            self._template_bins[row, :len(bins)] = bins
            self._template_values[row, :len(bins)] = self._templates[row, bins]
        if self._native is not None:
            self._native.prepare(self._template_bins, self._template_values, self.config.fft_size // 2)

    def _correlate(self, spectrum, indices=None):
        """Dot products over cached nonzero bins, with NumPy's input precision."""
        if self._native is not None:
            result = self._native.correlate(spectrum, indices)
            if result is not None:
                return result
        bins, values = self._template_bins, self._template_values
        if indices is not None:
            bins, values = bins[indices], values[indices]
        products = np.asarray(spectrum[bins], dtype=np.result_type(values, spectrum))
        products *= values
        return np.sum(products, axis=1)

    def _fit(self, design, spectrum):
        if self._native is not None:
            return self._native.fit(design, spectrum)
        return self._fit_python(design, spectrum)

    @staticmethod
    def _fit_python(design, spectrum):
        """Small coordinate-descent NNLS; no SciPy dependency or negative sources."""
        gram = design @ design.T
        target = design @ spectrum
        weights = np.zeros(len(design), dtype=np.float32)
        for _ in range(60):
            previous = weights.copy()
            for index in range(len(weights)):
                weights[index] = max(0.0, weights[index] +
                                     (target[index] - gram[index] @ weights) /
                                     max(float(gram[index, index]), 1e-12))
            if float(np.max(np.abs(weights - previous))) < 1e-5:
                break
        return weights, spectrum - weights @ design

    def _pursue(self, spectrum):
        total = float(spectrum @ spectrum)
        self._octave_uncertain = set()
        # Envelope refinement revisits identical designs within one frame.
        # Cache only this frame: a new spectrum always receives a new fit.
        fitted_designs = {}

        def fit(indices):
            key = tuple(indices)
            if key not in fitted_designs:
                fitted_designs[key] = self._fit(self._templates[list(key)], spectrum)
            return fitted_designs[key]

        selected = []
        residual = spectrum.copy()
        weights = np.empty(0, dtype=np.float32)
        excluded = np.zeros(len(self._templates), dtype=bool)
        previous_error = total
        for _ in range(self.config.max_notes):
            correlation = self._correlate(residual)
            correlation *= self._priors
            correlation[excluded] = -np.inf
            index = int(np.argmax(correlation))
            if not np.isfinite(correlation[index]) or correlation[index] <= 0:
                break
            trial = selected + [index]
            trial_weights, trial_residual = fit(trial)
            error = float(trial_residual @ trial_residual)
            gain = (previous_error - error) / max(total, 1e-12)
            if gain < self.config.minimum_note_gain:
                break
            selected, weights, residual = trial, trial_weights, trial_residual
            previous_error = error
            excluded |= self._midi_indices == self._metadata[index][0]
            # Refit each selected envelope to the mixture left by all other notes.
            for position, current in enumerate(selected):
                others = weights @ self._templates[selected] - weights[position] * self._templates[current]
                candidates = self._variants[self._metadata[current][0]]
                best = int(candidates[np.argmax(self._correlate(spectrum - others, candidates)
                                                * self._priors[candidates])])
                selected[position] = best
            weights, residual = fit(selected)
            previous_error = float(residual @ residual)
        # Early greedy choices can pool overlapping partials into a false lower note.
        # Replace those choices against the fully fitted mixture, then prune by unique gain.
        for _ in range(3):
            changed = False
            for position, current in enumerate(selected):
                others = weights @ self._templates[selected] - weights[position] * self._templates[current]
                scores = self._correlate(spectrum - others) * self._priors
                other_midis = {self._metadata[item][0] for slot, item in enumerate(selected)
                               if slot != position}
                scores[np.isin(self._midi_indices, list(other_midis))] = -np.inf
                best = int(np.argmax(scores))
                if best != current:
                    trial = list(selected)
                    trial[position] = best
                    trial_weights, trial_residual = fit(trial)
                    if float(trial_residual @ trial_residual) < float(residual @ residual) - 1e-6:
                        selected, weights, residual = trial, trial_weights, trial_residual
                        changed = True
            if not changed:
                break
        def refit_envelopes(indices):
            indices = list(indices)
            fitted, remainder = fit(indices)
            for _ in range(3):
                changed = False
                for slot, current in enumerate(indices):
                    others = fitted @ self._templates[indices] - fitted[slot] * self._templates[current]
                    candidates = self._variants[self._metadata[current][0]]
                    best = int(candidates[np.argmax(self._correlate(spectrum - others, candidates))])
                    trial = list(indices)
                    trial[slot] = best
                    trial_weights, trial_residual = fit(trial)
                    if float(trial_residual @ trial_residual) < float(remainder @ remainder):
                        indices, fitted, remainder = trial, trial_weights, trial_residual
                        changed = True
                if not changed:
                    break
            return indices, fitted, remainder

        while len(selected) > 1:
            losses = []
            trials = []
            for position in range(len(selected)):
                trial = selected[:position] + selected[position + 1:]
                trial, trial_weights, trial_residual = refit_envelopes(trial)
                trials.append((trial, trial_weights, trial_residual))
                losses.append(float(trial_residual @ trial_residual - residual @ residual) / total)
            weakest = int(np.argmin(losses))
            if losses[weakest] >= self.config.minimum_note_gain:
                break
            selected, weights, residual = trials[weakest]
        if len(weights):
            keep = weights >= float(np.max(weights)) * self.config.minimum_relative_strength
            selected = [index for index, accepted in zip(selected, keep) if accepted]
            weights, residual = fit(selected)
        # Two real fundamentals can be pooled into harmonics 2/3 of a false
        # lower note. Test one bounded split only when that bass has no measured
        # fundamental. Both recovered notes must contribute independently.
        for position, current in enumerate(list(selected)):
            midi, cents, _, fundamental = self._metadata[current]
            if fundamental == 1.0 or len(selected) >= self.config.max_notes:
                continue
            center = midi_frequency(midi,cents)/self._bin_hz
            if float(np.max(spectrum[max(1,int(center)-1):int(center)+3])) >= .08:
                continue
            others = selected[:position]+selected[position+1:]
            occupied = {self._metadata[i][0] for i in others}
            targets = (midi+12,midi+19)
            if any(p in occupied or p not in self._variants for p in targets):
                continue
            remainder = spectrum-weights @ self._templates[selected]+weights[position]*self._templates[current]
            replacements = []
            for pitch in targets:
                candidates = self._variants[pitch]
                candidates = candidates[self._full_fundamental[candidates]]
                replacements.append(int(candidates[np.argmax(self._correlate(remainder,candidates))]))
            trial, trial_weights, trial_residual = refit_envelopes(others+replacements)
            error = float(trial_residual @ trial_residual)
            if (float(residual @ residual)-error)/total < self.config.minimum_note_gain:
                continue
            losses = []
            for slot in (-2,-1):
                reduced = trial[:len(trial)+slot]+trial[len(trial)+slot+1:]
                _, _, reduced_residual = refit_envelopes(reduced)
                losses.append((float(reduced_residual @ reduced_residual)-error)/total)
            if min(losses) >= self.config.minimum_note_gain:
                selected, weights, residual = trial, trial_weights, trial_residual
                break
        # A missing fundamental can pool other notes' partials into a false
        # low octave. Prefer an explicit upper fundamental when its additional
        # fitting error is below half the minimum accepted note contribution.
        full_fundamental = self._full_fundamental
        for position, current in enumerate(selected):
            midi, _, _, fundamental = self._metadata[current]
            if fundamental != 0.0 or any(self._metadata[item][0] == midi + 12 for item in selected):
                continue
            candidates = np.flatnonzero((self._midi_indices == midi + 12) & full_fundamental)
            best = None
            for candidate in candidates:
                trial = list(selected)
                trial[position] = int(candidate)
                trial_weights, trial_residual = fit(trial)
                error = float(trial_residual @ trial_residual)
                if best is None or error < best[0]:
                    best = (error, trial, trial_weights, trial_residual)
            if best is not None and (best[0] - float(residual @ residual)) / max(total, 1e-12) <= self.config.minimum_note_gain / 2:
                selected, weights, residual = best[1:]
                self._octave_uncertain.add(midi + 12)
        explained = max(0.0, 1.0 - float(residual @ residual) / max(total, 1e-12))
        return selected, weights, residual, explained

    def analyze(self, power, sequence: Optional[int] = None) -> Dict:
        power = np.asarray(power, dtype=np.float64)
        if power.shape != (self.config.fft_size // 2,):
            raise ValueError("Expected exactly fft_size/2 power bins, with DC and without Nyquist")
        if not np.all(np.isfinite(power)) or np.any(power < 0):
            raise ValueError("Power bins must be finite and nonnegative")
        energy = float(np.sum(power[1:]))
        window_gain = (3.0 / 8.0 * (self.config.fft_size - 1) / self.config.fft_size
                       if self.config.window == "hann" else 1.0)
        denominator = window_gain
        scale = 1.0
        if self.config.power_scale is not None:
            scale = self.config.power_scale
            denominator *= self.config.fft_size ** 2 * 32768.0 ** 2
        rms_squared = (2.0 * energy + float(power[0])) * scale / denominator
        rms = float(np.sqrt(max(0.0, rms_squared)))
        # Integer truncation loses less than one power unit per retained bin.
        quantization_bound = (2 * len(power) - 1) * scale / denominator if self.config.power_scale is not None else 0.0
        state = {"status": "silence", "signal_active": False, "confidence": 0.0,
                 "sequence": sequence, "notes": [], "chord": recognize_chord([]),
                 "spectrum": {"power": energy, "noise_floor": 0.0,
                              "rms": round(rms, 9), "rms_pcm16": round(rms * 32768.0, 4),
                              "rms_dbfs": round(float(20.0 * np.log10(rms)), 3) if rms > 0 else None,
                              "rms_quantization_upper": round(float(np.sqrt(rms_squared + quantization_bound)), 9),
                              "minimum_rms": self.config.minimum_rms,
                              "rms_method": "Parseval/window-power corrected; Nyquist omitted; stationary signal estimate",
                              "explained_fraction": 0.0},
                 "scale_evidence": None, "limitations": list(LIMITATIONS)}
        if energy <= self.config.silence_power:
            state["reason"] = "below_configured_silence_floor"
            return state
        if rms < self.config.minimum_rms:
            state["reason"] = "below_absolute_rms_gate"
            return state
        state.update(status="unknown", signal_active=True, reason="no_supported_note_model")
        floor = float(np.median(power[1:]))
        state["spectrum"]["noise_floor"] = floor
        magnitude = np.sqrt(np.maximum(power - 2.0 * floor, 0.0))
        magnitude[~self._band] = 0.0
        maximum = float(np.max(magnitude))
        if maximum <= 0:
            return state
        magnitude = np.asarray(magnitude / maximum, dtype=np.float32)
        if self._templates is None:
            self._build_templates()
        selected, weights, residual, explained = self._pursue(magnitude)
        state["spectrum"]["explained_fraction"] = round(explained, 4)
        if explained < self.config.minimum_explained or not selected:
            state["reason"] = "weak_harmonic_fit_or_broadband_noise"
            return state
        notes = []
        for index, weight in zip(selected, weights):
            midi, cents, decay, fundamental = self._metadata[index]
            own = self._templates[index] * weight
            isolated = np.maximum(residual + own, 0.0)
            support = []
            for harmonic in range(1, self.config.max_harmonics + 1):
                frequency = midi_frequency(midi, cents) * harmonic
                if frequency > self.config.max_frequency:
                    break
                center = frequency / self._bin_hz
                start, stop = max(1, int(center) - 1), min(len(isolated), int(center) + 3)
                level = float(np.max(isolated[start:stop]))
                if level >= max(float(np.max(own)) * 0.07, 0.025):
                    support.append(harmonic)
            if not support:
                continue
            cosine = float((own @ isolated) / max(float(np.linalg.norm(own) *
                                                       np.linalg.norm(isolated)), 1e-12))
            confidence = min(0.99, 0.35 + 0.45 * cosine + 0.2 * min(len(support) / 3.0, 1.0))
            octave_ambiguous = (midi in self._octave_uncertain or len(support) == 1 or
                                all(harmonic % 2 == 0 for harmonic in support))
            if octave_ambiguous:
                confidence = min(confidence, 0.72)
            notes.append({"midi_note": int(midi), "note_name": note_name(midi),
                          "pitch_class": NOTE_NAMES[midi % 12],
                          "frequency_hz": round(midi_frequency(midi, cents), 3),
                          "cents": cents, "confidence": round(confidence, 4),
                          "relative_strength": round(float(weight / np.max(weights)), 4),
                          "harmonic_support": support, "octave_ambiguous": octave_ambiguous})
        notes.sort(key=lambda note: note["midi_note"])
        if not notes:
            return state
        confidence = explained * float(np.mean([note["confidence"] for note in notes]))
        state.update(status="recognized", confidence=round(confidence, 4), notes=notes,
                     chord=recognize_chord(notes), reason="supported_harmonic_model")
        if self.config.infer_scales:
            state["scale_evidence"] = scale_evidence(notes)
        return state
