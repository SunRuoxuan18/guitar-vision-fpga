"""Deterministic real-waveform recognition and fake-MMIO/API contract tests."""

import json
from ctypes import c_uint32
import mmap
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audio_fft import AUD4_REGISTERS, AudioFFT, DevMemMMIO, FFTRegisterMap, FFTSnapshot
from audio_music import (MusicConfig, MusicRecognizer, midi_frequency,
                         recognize_chord, scale_evidence, spectrum_from_samples)
from audio_music_server import MusicState, make_server, settle_analysis_source
from test_board_music import run_smoke


def synthesize(notes, decay=1.1, fundamental=1.0, harmonics=10, noise=0.0,
               cents=0.0, strengths=None, seed=1729, config=None):
    rng = np.random.default_rng(seed)
    time = np.arange(4096) / 48000.0
    samples = np.zeros(4096)
    strengths = strengths if strengths is not None else [1.0] * len(notes)
    for midi, strength in zip(notes, strengths):
        for harmonic in range(1, harmonics + 1):
            amplitude = 0.15 * strength / harmonic ** decay
            if harmonic == 1:
                amplitude *= fundamental
            samples += amplitude * np.sin(2 * np.pi * midi_frequency(midi, cents) * harmonic * time
                                          + rng.uniform(-np.pi, np.pi))
    samples += rng.normal(0.0, noise, len(samples))
    return spectrum_from_samples(samples, config)


class RecognitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recognizer = MusicRecognizer()

    def assert_notes(self, expected, **kwargs):
        result = self.recognizer.analyze(synthesize(expected, **kwargs), sequence=123)
        self.assertEqual(result["status"], "recognized", result)
        actual = [note["midi_note"] for note in result["notes"]]
        self.assertEqual(actual, sorted(expected), result)
        self.assertEqual(result["sequence"], 123)
        self.assertGreater(result["confidence"], 0.45)
        json.dumps(result, allow_nan=False)
        return result

    def test_mono_guitar_range(self):
        for midi in (40, 45, 50, 55, 59, 64, 76, 88):
            with self.subTest(midi=midi):
                self.assert_notes([midi])

    def test_per_frame_fit_cache_never_reuses_another_spectrum(self):
        recognizer = self.recognizer
        signatures = []
        original_fit = recognizer._fit

        def recording_fit(design, spectrum):
            signature = design.tobytes()
            self.assertNotIn(signature, signatures)
            signatures.append(signature)
            return original_fit(design, spectrum)

        for expected in ([60, 64, 67], [57, 60, 64], [60, 64, 67]):
            signatures.clear()
            with patch.object(recognizer, "_fit", side_effect=recording_fit):
                result = recognizer.analyze(synthesize(expected, harmonics=1))
            self.assertEqual([note["midi_note"] for note in result["notes"]], expected)
            self.assertGreater(len(signatures), 0)

    def test_harmonics_do_not_become_notes(self):
        for midi in (40, 48, 57, 64):
            with self.subTest(midi=midi):
                self.assert_notes([midi], decay=0.7, fundamental=0.25)

    def test_missing_fundamental_uses_odd_partials(self):
        result = self.assert_notes([45], fundamental=0.0, decay=0.7)
        self.assertIn(3, result["notes"][0]["harmonic_support"])

    def test_major_triad_and_inversions(self):
        for notes, label, inversion in (((48, 52, 55), "C", 0),
                                        ((52, 55, 60), "C/E", 1),
                                        ((55, 60, 64), "C/G", 2)):
            with self.subTest(notes=notes):
                result = self.assert_notes(list(notes))
                self.assertEqual(result["chord"]["label"], label)
                self.assertEqual(result["chord"]["inversion"], inversion)

    def test_minor_triad(self):
        self.assertEqual(self.assert_notes([45, 48, 52])["chord"]["label"], "Am")

    def test_pl_pure_tone_triad_voicings(self):
        for notes, label in (([60, 64, 67], "C"), ([57, 60, 64], "Am")):
            with self.subTest(label=label):
                result = self.assert_notes(notes, harmonics=1)
                self.assertEqual(result["chord"]["label"], label)
                self.assertTrue(all(note["octave_ambiguous"] for note in result["notes"]))

    def test_pure_tone_hardware_quantization(self):
        config = MusicConfig(power_scale=1048576.0)
        recognizer = MusicRecognizer(config)
        for notes, label in (([60, 64, 67], "C"), ([57, 60, 64], "Am")):
            power = np.asarray(synthesize(notes, harmonics=1) * 4096 ** 2 * 32768 ** 2 / 1048576,
                               dtype=np.uint32)
            result = recognizer.analyze(power)
            self.assertEqual(result["chord"]["label"], label, result)

    def test_sevenths(self):
        for notes, label in (((48, 52, 55, 58), "C7"),
                             ((48, 52, 55, 59), "Cmaj7"),
                             ((50, 53, 57, 60), "Dm7"),
                             ((52, 55, 58, 62), "Em7b5")):
            with self.subTest(label=label):
                self.assertEqual(self.assert_notes(list(notes))["chord"]["label"], label)

    def test_extended_guitar_chords(self):
        for notes, label in (((48, 52, 55, 62), "Cadd9"),
                             ((48, 51, 55, 62), "Cm(add9)"),
                             ((48, 52, 55, 58, 62), "C9"),
                             ((48, 52, 55, 59, 62), "Cmaj9"),
                             ((48, 51, 55, 58, 62), "Cm9"),
                             ((48, 52, 55, 57), "C6"),
                             ((48, 51, 55, 57), "Cm6")):
            with self.subTest(label=label):
                result = self.assert_notes(list(notes))
                self.assertEqual(result["chord"]["label"], label)
                if label == "C6":
                    self.assertTrue(result["chord"]["ambiguous"])
                    self.assertIn("Am7/C", [item["label"] for item in result["chord"]["alternatives"]])
                if label == "Cmaj9":
                    self.assertTrue(next(note for note in result["notes"] if note["midi_note"] == 55)["octave_ambiguous"])
        inversion = recognize_chord([{"midi_note": midi} for midi in (52, 55, 58, 60, 62)])
        self.assertEqual(inversion["label"], "C9/E")
        self.assertEqual(inversion["inversion"], 1)

    def test_suspended_and_power(self):
        for notes, label in (((48, 53, 55), "Csus4"),
                             ((48, 50, 55), "Csus2"),
                             ((40, 47), "E5")):
            with self.subTest(label=label):
                self.assertEqual(self.assert_notes(list(notes))["chord"]["label"], label)

    def test_six_string_voicing(self):
        result = self.recognizer.analyze(synthesize([40, 47, 52, 56, 59, 64]))
        self.assertEqual(result["status"], "recognized", result)
        self.assertEqual(result["chord"]["label"], "E", result)
        self.assertEqual({note["pitch_class"] for note in result["notes"]}, {"E", "G#", "B"})

    def test_high_add11_doubling_does_not_pool_two_notes_into_a_false_bass(self):
        for transpose in (-2, 0, 2):
            notes = [p+transpose for p in (60,65,67,72,76)]
            result = self.recognizer.analyze(synthesize(notes, harmonics=8, seed=672))
            with self.subTest(transpose=transpose):
                self.assertEqual({n['midi_note'] % 12 for n in result['notes']}, {p % 12 for p in notes})
                self.assertNotIn(48+transpose, [n['midi_note'] for n in result['notes']])

    def test_detuned_and_unequal_mixture(self):
        self.assert_notes([48, 52, 55], cents=17.0, strengths=[1.0, 0.7, 0.5], noise=0.002)

    def test_silence_and_low_energy(self):
        for power in (np.zeros(2048), synthesize([57]) * 1e-12):
            result = self.recognizer.analyze(power)
            self.assertEqual(result["status"], "silence")
            self.assertFalse(result["signal_active"])
            self.assertEqual(result["notes"], [])

    def test_noise_and_out_of_band_are_unknown(self):
        out_of_band = np.zeros(2048)
        out_of_band[1000] = 1.0
        for power in (synthesize([], noise=0.15), np.ones(2048), out_of_band):
            result = self.recognizer.analyze(power)
            self.assertEqual(result["status"], "unknown", result)
            self.assertTrue(result["signal_active"])
            self.assertEqual(result["notes"], [])

    def test_single_partial_octave_ambiguity(self):
        result = self.assert_notes([57], harmonics=1)
        self.assertTrue(result["notes"][0]["octave_ambiguous"])
        self.assertLessEqual(result["notes"][0]["confidence"], 0.72)

    def test_octave_doubling_is_not_overconfident(self):
        result = self.recognizer.analyze(synthesize([48, 60]))
        # At this resolution a strong second partial and a doubled octave overlap.
        self.assertEqual(result["status"], "recognized", result)
        self.assertTrue({note["midi_note"] for note in result["notes"]} <= {48, 60}, result)
        self.assertEqual(result["chord"]["status"], "unknown")
        self.assertTrue(any("octave" in limit for limit in result["limitations"]))

    def test_invalid_spectra(self):
        for power in (np.zeros(2049), np.full(2048, -1), np.full(2048, np.nan), np.full(2048, np.inf)):
            with self.assertRaises(ValueError):
                self.recognizer.analyze(power)

    def test_pitch_class_ambiguity_and_unknown(self):
        notes = [{"midi_note": midi, "confidence": 0.9} for midi in (48, 50, 55)]
        result = recognize_chord(notes)
        self.assertEqual(result["label"], "Csus2")
        self.assertTrue(result["ambiguous"])
        self.assertTrue(any(item["root"] == "G" for item in result["alternatives"]))
        self.assertEqual(recognize_chord([{"midi_note": midi} for midi in (48, 49, 55)])["status"], "unknown")
        diminished = recognize_chord([{"midi_note": midi} for midi in (48, 51, 54, 57)])
        self.assertTrue(diminished["ambiguous"])
        self.assertEqual(len(diminished["alternatives"]), 3)

    def test_scale_evidence_stays_ambiguous(self):
        result = scale_evidence([{"midi_note": midi} for midi in (48, 52, 55)])
        self.assertEqual(result["status"], "ambiguous")
        self.assertGreater(len(result["candidates"]), 1)
        self.assertEqual(scale_evidence([])["status"], "insufficient")

    def test_all_chord_qualities_and_seventh_inversion(self):
        for midis, quality in (((48, 51, 54), "diminished"),
                               ((48, 52, 56), "augmented"),
                               ((48, 51, 55, 59), "minor_major_seventh"),
                               ((48, 51, 54, 57), "diminished_seventh"),
                               ((48, 50, 55, 58), "dominant_seventh_sus2"),
                               ((48, 53, 55, 58), "dominant_seventh_sus4")):
            with self.subTest(quality=quality):
                chord = recognize_chord([{"midi_note": midi, "confidence": 0.8} for midi in midis])
                self.assertEqual(chord["quality"], quality)
        self.assertEqual(self.assert_notes([52, 55, 58, 60])["chord"]["label"], "C7/E")

    def test_rectangular_fft_and_uint32_power(self):
        config = MusicConfig(window="rectangular", silence_power=1.0)
        recognizer = MusicRecognizer(config)
        power = np.asarray(synthesize([48, 52, 55], config=config) * 2 ** 26, dtype=np.uint32)
        result = recognizer.analyze(power)
        self.assertEqual(result["status"], "recognized", result)
        self.assertEqual(result["chord"]["label"], "C", result)

    def test_scale_option_and_pcm_validation(self):
        result = MusicRecognizer(MusicConfig(infer_scales=True)).analyze(synthesize([48, 52, 55]))
        self.assertEqual(result["scale_evidence"]["status"], "ambiguous")
        for samples in (np.zeros(20), np.full(4096, np.inf)):
            with self.assertRaises(ValueError):
                spectrum_from_samples(samples)

    def test_absolute_rms_gate_hann_q15_shift20(self):
        # Reproduce PL PCM/window quantization and unsigned FFT-power truncation.
        time = np.arange(4096) / 48000.0
        window = np.rint(32767 * np.hanning(4096)).astype(np.int64)
        config = MusicConfig(power_scale=1048576.0, minimum_rms=0.003)
        recognizer = MusicRecognizer(config)
        for expected_rms in (0.001, 0.01):
            samples = np.rint(expected_rms * np.sqrt(2) * 32768 *
                              np.sin(2 * np.pi * midi_frequency(57) * time)).astype(np.int64)
            windowed = (samples * window) >> 15
            power = np.asarray(np.abs(np.fft.rfft(windowed))[:-1] ** 2 / 1048576, dtype=np.uint32)
            result = recognizer.analyze(power)
            self.assertAlmostEqual(result["spectrum"]["rms"], expected_rms, delta=0.00004)
            if expected_rms < 0.003:
                self.assertEqual(result["status"], "silence")
                self.assertEqual(result["reason"], "below_absolute_rms_gate")
                self.assertEqual(result["notes"], [])
            else:
                self.assertEqual(result["status"], "recognized", result)
                self.assertEqual([note["midi_note"] for note in result["notes"]], [57])
                self.assertAlmostEqual(result["spectrum"]["rms_dbfs"], -40.0, delta=0.05)

    def test_hardware_scale_wrong_value_rejected_by_gate(self):
        power = np.zeros(2048, dtype=np.uint32)
        power[19] = 100
        result = MusicRecognizer(MusicConfig(power_scale=1048576.0)).analyze(power)
        self.assertEqual(result["status"], "silence")
        self.assertLess(result["spectrum"]["rms"], 0.001)


class SparseCorrelationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recognizers = [MusicRecognizer(),
                           MusicRecognizer(MusicConfig(window="rectangular",
                                                       max_harmonics=32))]
        for recognizer in cls.recognizers:
            recognizer._build_templates()

    def test_cache_preserves_every_nonzero_template_value(self):
        for recognizer in self.recognizers:
            bins, values = recognizer._template_bins, recognizer._template_values
            with self.subTest(config=recognizer.config):
                self.assertEqual(values.dtype, recognizer._templates.dtype)
                self.assertEqual(bins.shape, values.shape)
                self.assertLess(bins.shape[1], recognizer._templates.shape[1])
                for template, row_bins, row_values in zip(recognizer._templates, bins, values):
                    active = row_values != 0
                    np.testing.assert_array_equal(row_bins[active], np.flatnonzero(template))
                    np.testing.assert_array_equal(row_values[active], template[row_bins[active]])
                    self.assertTrue(np.all(row_bins[~active] == 0))

    def test_correlations_match_dense_for_random_residuals_and_subsets(self):
        rng = np.random.default_rng(925)
        for recognizer in self.recognizers:
            templates = recognizer._templates
            subsets = (None, rng.integers(len(templates), size=37),
                       [len(templates) - 1, 0, 0], [])
            for dtype in (np.float32, np.float64):
                tolerance = 2e-6 if dtype == np.float32 else 2e-14
                for scale in (1e-3, 1.0, 1e3):
                    residual = (rng.standard_normal(templates.shape[1]) * scale).astype(dtype)
                    for subset in subsets:
                        with self.subTest(config=recognizer.config, dtype=dtype,
                                          scale=scale, subset=subset):
                            design = templates if subset is None else templates[subset]
                            expected = design @ residual
                            actual = recognizer._correlate(residual, subset)
                            self.assertEqual(actual.dtype, expected.dtype)
                            np.testing.assert_allclose(actual, expected, rtol=tolerance,
                                                       atol=tolerance * scale)

    def test_recognition_matches_dense_correlations(self):
        recognizer = self.recognizers[0]

        def dense_correlate(residual, indices=None):
            design = recognizer._templates if indices is None else recognizer._templates[indices]
            return design @ residual

        cases = (([69], {"harmonics": 1}),
                 ([60, 64, 67], {"harmonics": 1}),
                 ([57, 60, 64], {"harmonics": 1}),
                 ([45], {"fundamental": 0.0, "decay": 0.7}),
                 ([48, 52, 55], {"cents": 17.0, "strengths": [1.0, 0.7, 0.5],
                                 "noise": 0.002}),
                 ([40, 47, 52, 56, 59, 64], {}),
                 ([], {"noise": 0.15}))
        for notes, options in cases:
            with self.subTest(notes=notes, options=options):
                power = synthesize(notes, **options)
                actual = recognizer.analyze(power, sequence=925)
                with patch.object(recognizer, "_correlate", side_effect=dense_correlate):
                    expected = recognizer.analyze(power, sequence=925)
                self.assertEqual(actual, expected)


REGISTERS = FFTRegisterMap(control_offset=0, sequence_offset=4, status_offset=8,
                           freeze_mask=2, freeze_ack_mask=4, ready_mask=1,
                           power_base_offset=0x100)


class FakeMMIO:
    def __init__(self, corrupt_reads=0, acknowledge=True, fail_bin=None):
        self.control = 0x80
        self.sequence = 0xFFFFFFFF
        self.corrupt_reads = corrupt_reads
        self.acknowledge = acknowledge
        self.fail_bin = fail_bin
        self.index = 0
        self.reads = 0

    def write(self, offset, value):
        if offset == 0:
            self.control = value
        elif offset == 0x10:
            self.index = value
        else:
            raise AssertionError("Unexpected write")

    def read(self, offset):
        if offset == 0:
            return self.control
        if offset == 4:
            return self.sequence
        if offset == 8:
            return 1 | (4 if self.control & 2 and self.acknowledge else 0)
        index = self.index if offset == 0x14 else (offset - 0x100) // 4
        if index == self.fail_bin:
            raise OSError("bus error")
        if index == 2047 and self.corrupt_reads:
            self.sequence = (self.sequence + 1) & 0xFFFFFFFF
            self.corrupt_reads -= 1
        self.reads += 1
        return 0xFFFFFFFF if index == 10 else index


class DriverTests(unittest.TestCase):
    def test_test_mode_change_drains_busy_and_preserves_bits(self):
        class ModeMMIO:
            def __init__(self):
                self.control = 0x80
                self.busy_reads = 0
                self.writes = []

            def read(self, offset):
                if offset == 0x0C:
                    return self.control
                if offset == 4:
                    self.busy_reads += 1
                    # Source selection is allowed even before the first valid frame.
                    return 4 | int(self.busy_reads < 3)
                raise AssertionError("Unexpected read")

            def write(self, offset, value):
                if offset != 0x0C:
                    raise AssertionError("Unexpected write")
                if (value & 6) != (self.control & 6):
                    if not self.control & 1 or self.busy_reads < 3:
                        raise AssertionError("Source changed before frozen idle")
                self.control = value
                self.writes.append(value)

        registers = FFTRegisterMap(control_offset=0x0C, status_offset=4, sequence_offset=8,
                                   freeze_mask=1, busy_mask=1, ready_mask=2,
                                   freeze_ack_mask=4, power_base_offset=0x1000)
        mmio = ModeMMIO()
        driver = AudioFFT(mmio=mmio, register_map=registers)
        with patch("audio_fft.time.sleep") as delay:
            driver.set_test_mode(2)
        self.assertEqual(mmio.writes, [0x81, 0x85, 0x84])
        self.assertIn(unittest.mock.call(0.0001), delay.call_args_list)
        self.assertEqual(mmio.control, 0x84)
        # A caller-owned freeze remains asserted after a later mode switch.
        mmio.control |= 1
        driver.set_test_mode(3)
        self.assertEqual(mmio.control, 0x87)
        for mode in (-1, 4, 1.5, True):
            with self.assertRaises(ValueError):
                driver.set_test_mode(mode)

    def test_devmem_bounds_unsigned_and_close_without_board_access(self):
        mapping = mmap.mmap(-1, 0x4004)
        with patch("audio_fft.os.open", return_value=44) as opened, \
                patch("audio_fft.os.close") as closed, \
                patch("audio_fft.mmap.mmap", return_value=mapping) as mapped:
            mmio = DevMemMMIO(0x43C00004, 0x4000)
        opened.assert_called_once()
        closed.assert_called_once_with(44)
        self.assertEqual(mapped.call_args.kwargs["offset"], 0x43C00000)
        mmio.write(0x100, 0xFFFFFFFF)
        self.assertEqual(mmio.read(0x100), 0xFFFFFFFF)
        for offset in (-4, 2, 0x4000):
            with self.assertRaises(ValueError):
                mmio.read(offset)
        mmio.close()
        mmio.close()
        self.assertTrue(mapping.closed)
        with self.assertRaises(RuntimeError):
            mmio.read(0)

    def test_devmem_control_uses_aligned_native_words_without_byte_packing(self):
        mapping = mmap.mmap(-1, 0x4004)
        with patch("audio_fft.os.open", return_value=44), \
                patch("audio_fft.os.close"), \
                patch("audio_fft.mmap.mmap", return_value=mapping):
            mmio = DevMemMMIO(0x43C00004, 0x4000)
        # Host RAM accepts byte accesses, so explicitly guard the path that faulted on ARM.
        with patch("struct.pack_into", side_effect=AssertionError("Byte-packed MMIO write")), \
                patch("struct.unpack_from", side_effect=AssertionError("Byte-packed MMIO read")), \
                patch("audio_fft.c_uint32", wraps=c_uint32) as word_type:
            mmio.write(0x08, 0x11223344)
            mmio.write(0x0C, 0x85)
            mmio.write(0x10, 0xFFFFFFFF)
            self.assertEqual(mmio.read(0x0C), 0x85)
            self.assertEqual(mmio.read(0x08), 0x11223344)
            self.assertEqual(mmio.read(0x10), 0xFFFFFFFF)
        self.assertEqual([call.args[1] for call in word_type.from_buffer.call_args_list],
                         [0x0C, 0x10, 0x14, 0x10, 0x0C, 0x14])
        self.assertTrue(all(call.args[0] is mapping for call in word_type.from_buffer.call_args_list))
        mmio.close()
        self.assertTrue(mapping.closed)

    def test_coherent_unsigned_and_control_preservation(self):
        mmio = FakeMMIO()
        result = AudioFFT(mmio=mmio, register_map=REGISTERS).read_snapshot()
        self.assertEqual(result.sequence, 0xFFFFFFFF)
        self.assertEqual(len(result.power), 2048)
        self.assertEqual(result.power.dtype, np.dtype("uint32"))
        self.assertEqual(int(result.power[10]), 0xFFFFFFFF)
        self.assertEqual(mmio.control, 0x80)
        self.assertFalse(result.power.flags.writeable)

    def test_sequence_retry_and_wrap(self):
        mmio = FakeMMIO(corrupt_reads=1)
        result = AudioFFT(mmio=mmio, register_map=REGISTERS).read_snapshot()
        self.assertEqual(result.sequence, 0)
        self.assertEqual(mmio.reads, 4096)
        self.assertEqual(mmio.control, 0x80)

    def test_incoherent_frame_discarded(self):
        mmio = FakeMMIO(corrupt_reads=3)
        with self.assertRaises(RuntimeError):
            AudioFFT(mmio=mmio, register_map=REGISTERS).read_snapshot()
        self.assertEqual(mmio.control, 0x80)

    def test_timeout_and_bus_error_release_freeze(self):
        for mmio, exception in ((FakeMMIO(acknowledge=False), TimeoutError),
                                (FakeMMIO(fail_bin=20), OSError)):
            with self.subTest(exception=exception):
                with self.assertRaises(exception):
                    AudioFFT(mmio=mmio, register_map=REGISTERS, timeout=0.003).read_snapshot()
                self.assertEqual(mmio.control, 0x80)

    def test_existing_freeze_remains_owned_by_caller(self):
        mmio = FakeMMIO()
        mmio.control |= 2
        AudioFFT(mmio=mmio, register_map=REGISTERS).read_snapshot()
        self.assertEqual(mmio.control, 0x82)

    def test_indexed_port(self):
        registers = FFTRegisterMap(control_offset=0, sequence_offset=4, status_offset=8,
                                   freeze_mask=2, freeze_ack_mask=4,
                                   bin_index_offset=0x10, bin_power_offset=0x14)
        result = AudioFFT(mmio=FakeMMIO(), register_map=registers).read_snapshot()
        self.assertEqual(int(result.power[-1]), 2047)

    def test_invalid_core_and_missing_contract(self):
        with self.assertRaises(RuntimeError):
            AudioFFT(mmio=FakeMMIO())
        with self.assertRaises(ValueError):
            AudioFFT(mmio=FakeMMIO(), register_map=None)

    def test_aud4_busy_clear_metadata_and_preflight(self):
        class AUD4MMIO:
            def __init__(self):
                self.control = 0x20
                self.busy_reads = 0
                self.read_bins = 0

            def write(self, offset, value):
                if offset != 0x0C:
                    raise AssertionError("Unexpected AUD4 write")
                self.control = value

            def read(self, offset):
                fixed = {0: 0x41554434, 8: 19, 0x10: 4096, 0x14: 48000,
                         0x18: 7, 0x1C: 1, 0x20: 1048576, 0x24: 4095,
                         0x100: 37, 0x104: 900}
                if offset == 0x0C:
                    return self.control
                if offset == 4:
                    self.busy_reads += 1
                    # Frozen request is visible while the in-flight transform still runs.
                    return 6 | int(self.busy_reads < 3)
                if offset >= 0x1000:
                    if self.busy_reads < 3:
                        raise AssertionError("Read spectrum before busy cleared")
                    self.read_bins += 1
                    return (offset - 0x1000) // 4
                return fixed[offset]
        mmio = AUD4MMIO()
        result = AudioFFT(mmio=mmio).read_snapshot()
        self.assertEqual(mmio.read_bins, 2048)
        self.assertEqual(mmio.control, 0x20)
        self.assertEqual(result.metadata["drops"], 7)
        self.assertEqual(result.metadata["peaks"][0]["bin"], 37)
        self.assertEqual(result.metadata["power_shift"], 20)
        self.assertEqual(result.metadata["power_scale_factor"], 1048576)
        self.assertEqual(result.metadata["window"], "hann")
        self.assertEqual(result.metadata["analysis_test_mode"], 0)
        self.assertEqual(AUD4_REGISTERS.power_base_offset, 0x1000)
        with patch.object(mmio, "read", side_effect=lambda offset: 0x41554434 if offset == 0 else 42):
            with self.assertRaises(RuntimeError):
                AudioFFT(mmio=mmio)

    def test_error_status_releases_freeze(self):
        mmio = FakeMMIO()
        registers = FFTRegisterMap(control_offset=0, sequence_offset=4, status_offset=8,
                                   freeze_mask=2, freeze_ack_mask=4, error_mask=1,
                                   power_base_offset=0x100)
        with self.assertRaises(RuntimeError):
            AudioFFT(mmio=mmio, register_map=registers).read_snapshot()
        self.assertEqual(mmio.control, 0x80)


class ServerTests(unittest.TestCase):
    def test_startup_discards_inflight_source_frames_and_handles_wrap(self):
        class Source:
            def __init__(self):
                self.sequences = iter((0xfffffffe, 0xfffffffe, 0xffffffff, 0, 1))
                self.reads = 0

            def read_snapshot(self):
                self.reads += 1
                return FFTSnapshot(next(self.sequences), np.zeros(2048))
        source = Source()
        with patch("audio_music_server.time.sleep"):
            settle_analysis_source(source)
        self.assertEqual(source.reads, 5)

    def test_real_http_json_and_health(self):
        class Source:
            def read_snapshot(self):
                return FFTSnapshot(7, np.zeros(2048, dtype=np.uint32),
                                   metadata={"fft_size": 4096, "sample_rate_hz": 48000,
                                             "bin_count": 2048, "window": "hann",
                                             "analysis_test_mode": 0})
        state = MusicState(Source())
        state.update()
        server = make_server(state, port=0)
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
        thread.start()
        base = "http://127.0.0.1:{}".format(server.server_port)
        try:
            with urlopen(base + "/api/audio", timeout=2) as response:
                payload = json.load(response)
                self.assertEqual(payload["status"], "silence")
                self.assertEqual(payload["sequence"], 7)
                self.assertEqual(payload["schema_version"], 1)
                self.assertEqual(payload["fft"]["bin_count"], 2048)
                self.assertEqual(payload["fft"]["analysis_test_mode"], 0)
                self.assertEqual(payload["spectrum"]["bins"], [0] * 2048)
                self.assertEqual(response.headers["Cache-Control"], "no-store")
            with urlopen(base + "/healthz", timeout=2) as response:
                self.assertTrue(json.load(response)["healthy"])
            with self.assertRaises(HTTPError) as error:
                urlopen(base + "/missing", timeout=2)
            self.assertEqual(error.exception.code, 404)
        finally:
            server.shutdown()
            thread.join(timeout=2)
            server.server_close()

    def test_full_spectrum_preserves_bin_order_scale_and_unsigned_values(self):
        class Source:
            def __init__(self, power):
                self.power = power

            def read_snapshot(self):
                return FFTSnapshot(8, self.power)
        integers = np.arange(2048, dtype=np.uint32)
        integers[-1] = 0xFFFFFFFF
        for power in (integers, np.linspace(0.0, 1e-9, 2048)):
            with self.subTest(dtype=power.dtype):
                state = MusicState(Source(power))
                state.update()
                payload = json.loads(json.dumps(state.snapshot(), allow_nan=False))
                self.assertEqual(payload["spectrum"]["bins"], power.tolist())
                self.assertEqual(len(payload["spectrum"]["bins"]), 2048)
                if np.issubdtype(power.dtype, np.integer):
                    self.assertEqual(payload["spectrum"]["bins"][-1], 4294967295)
                    self.assertIsInstance(payload["spectrum"]["bins"][-1], int)

    def test_unavailable_health_static_and_stalled_sequence(self):
        state = MusicState(None, unavailable_error="No PYNQ MMIO permissions")
        state.update()
        self.assertEqual(state.snapshot()["status"], "hardware_unavailable")
        self.assertEqual(state.snapshot()["notes"], [])
        self.assertFalse(state.snapshot()["hardware_available"])
        # Serve an existing assigned file; no web files or fixtures need to be written.
        server = make_server(state, port=0, web_root=Path(__file__).resolve().parents[1],
                             cors_origin="http://localhost:8080")
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
        thread.start()
        base = "http://127.0.0.1:{}".format(server.server_port)
        try:
            with urlopen(base + "/README_MUSIC.md", timeout=2) as response:
                self.assertIn(b"PS-side music recognition", response.read())
            with urlopen(base + "/api/audio", timeout=2) as response:
                self.assertEqual(json.load(response)["status"], "hardware_unavailable")
                self.assertEqual(response.headers["Access-Control-Allow-Origin"], "http://localhost:8080")
            with self.assertRaises(HTTPError) as error:
                urlopen(base + "/healthz", timeout=2)
            self.assertEqual(error.exception.code, 503)
        finally:
            server.shutdown()
            thread.join(timeout=2)
            server.server_close()
        class Source:
            def read_snapshot(self):
                return FFTSnapshot(10, np.zeros(2048))
        state = MusicState(Source())
        with patch("audio_music_server.time.monotonic", return_value=100.0):
            state.update()
        with patch("audio_music_server.time.monotonic", return_value=102.0):
            state.update()
            self.assertTrue(state.snapshot()["stale"])
            self.assertEqual(state.snapshot()["age_seconds"], 2.0)


class BoardSmokeTests(unittest.TestCase):
    def test_mode_checks_wait_new_frames_and_restore_live(self):
        class Source:
            def __init__(self, incorrect=False):
                self.sequence = 0
                self.mode = 0
                self.modes = []
                self.incorrect = incorrect

            def set_test_mode(self, mode):
                self.mode = mode
                self.modes.append(mode)

            def read_snapshot(self):
                self.sequence += 1
                notes = {1: [69], 2: [60, 64, 67], 3: [57, 60, 64]}[self.mode]
                power = synthesize([] if self.incorrect else notes, harmonics=1)
                return FFTSnapshot(self.sequence, power, metadata={"analysis_test_mode": self.mode})
        for incorrect in (False, True):
            with self.subTest(incorrect=incorrect):
                source = Source(incorrect)
                with patch("test_board_music.time.sleep"):
                    report = run_smoke(source, recognizer=MusicRecognizer())
                self.assertEqual(report["passed"], not incorrect, report)
                self.assertTrue(report["restored_live_mode"])
                self.assertEqual(source.mode, 0)
                self.assertEqual(source.modes[-1], 0)
                self.assertTrue(all(check["observed_new_frames"] >= 3 for check in report["checks"]))
                if not incorrect:
                    self.assertEqual(source.modes, [1, 2, 3, 0])

    def test_error_and_recovery_clear_old_notes(self):
        class Source:
            fail = False

            def read_snapshot(self):
                if self.fail:
                    raise TimeoutError("test freeze timeout")
                return FFTSnapshot(9, synthesize([60, 64, 67], harmonics=1))
        source = Source()
        state = MusicState(source)
        state.update()
        self.assertEqual(state.snapshot()["status"], "recognized")
        self.assertTrue(state.snapshot()["notes"])
        source.fail = True
        state.update()
        self.assertEqual(state.snapshot()["status"], "error")
        self.assertEqual(state.snapshot()["notes"], [])
        self.assertTrue(state.snapshot()["stale"])
        self.assertEqual(state.snapshot()["chord"]["status"], "unknown")
        self.assertIsNone(state.snapshot()["spectrum"])
        source.fail = False
        state.update()
        self.assertEqual(state.snapshot()["status"], "recognized")
        self.assertEqual(state.snapshot()["chord"]["label"], "C")
        self.assertFalse(state.snapshot()["stale"])


if __name__ == "__main__":
    unittest.main()
