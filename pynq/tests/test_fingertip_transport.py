"""Host/ARM transport parity; never opens a board or invokes real PL.

FINGERTIP_TRANSPORT_LIBRARY selects a compiled .so/.dll. If unset, use the
library beside fingertip_transport.py. FINGERTIP_TEST_BOARD_DRIVER=1 loads
the captured 20261026 driver and scheduler instead of repository versions.
On ARM the optional libhand_q16.so comparison checks the deployed native
unpack -> Clip -> pack contract including NEON rounding, not just NumPy.
"""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'pynq'), str(ROOT/'pynq/tests')]
if os.environ.get('FINGERTIP_TEST_BOARD_DRIVER') == '1':
    sys.path.insert(0, str(ROOT/'diagnostics/board_runtime_20261026'))

from fingertip_transport import (DeferredOutput, FingertipGraph,
                                FingertipQ16NeuralPL, NativeTransport)
from hand_graph import HandGraph
from hand_neural_pl import Q16NeuralPL, quantize_activation
import test_hand_q16


class TransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        library = os.environ.get('FINGERTIP_TRANSPORT_LIBRARY')
        cls.transport = NativeTransport(library)

    def engine(self):
        baseline = test_hand_q16.Q16Tests().engine()
        engine = FingertipQ16NeuralPL.__new__(FingertipQ16NeuralPL)
        engine.__dict__.update(baseline.__dict__)
        engine.output_generation = 0
        engine.transport = self.transport
        engine.transport_calls = 0
        engine.transport_ms = 0.
        engine.events = []
        class CachedBuffer(type(engine.buffers[0])):
            cacheable = True
            def flush(buffer):
                engine.events.append(('flush', buffer.ctypes.data))
            def invalidate(buffer):
                engine.events.append(('invalidate', buffer.ctypes.data))
        engine.buffers = [buffer.view(CachedBuffer) for buffer in engine.buffers]
        baseline.buffers = engine.buffers
        addresses = [buffer.ctypes.data for buffer in engine.buffers]
        def run(*args):
            engine.events.append(('run', addresses[0]))
            engine.output_generation += 1
            return baseline.run(*args)
        engine.run = run
        return engine

    def test_native_matches_float32_contract_tails_and_guards(self):
        rng = np.random.default_rng(118)
        for channels in (1, 3, 8, 9, 17):
            for pixels in (1, 3, 4, 15, 16, 17, 35):
                with self.subTest(channels=channels, pixels=pixels):
                    raw = rng.integers(-2**46, 2**46, (channels, pixels), dtype=np.int64)
                    factors = np.geomspace(1e-16, 1e-12, channels).astype(np.float32)
                    bias = rng.uniform(-2, 2, channels).astype(np.float32)
                    converted = raw.astype(np.float32)*factors[:, None]
                    converted += bias[:, None]
                    np.clip(converted, 0, 6, out=converted)
                    expected, scale = quantize_activation(converted.reshape(1, channels, 1, pixels))
                    guard = np.full(expected.size+16, -12345, np.int16)
                    actual = guard[8:-8]
                    source = raw.copy()
                    actual_scale = self.transport.clip_pack(raw, actual, factors, bias, channels, pixels)
                    self.assertEqual(actual_scale, float(np.float32(scale)))
                    # Scalar builds are exact; ARM is separately compared to
                    # deployed native code below (int64/reciprocal semantics).
                    if not hasattr(os, 'uname') or not os.uname().machine.startswith(('arm', 'aarch')):
                        np.testing.assert_array_equal(actual, expected.reshape(-1))
                    np.testing.assert_array_equal(actual.reshape(pixels, -1)[:, channels:], 0)
                    np.testing.assert_array_equal(guard[:8], -12345)
                    np.testing.assert_array_equal(guard[-8:], -12345)
                    np.testing.assert_array_equal(raw, source)

    def test_zero_ties_and_dynamic_maximum(self):
        for raw in (np.zeros((3, 7), np.int64),
                    np.array([[1, 3, 5, 7, 9, 11, 65534]], np.int64)):
            c, p = raw.shape
            factors, bias = np.full(c, 1/16384, np.float32), np.zeros(c, np.float32)
            values = np.clip(raw.astype(np.float32)*factors[:, None], 0, 6)
            expected, scale = quantize_activation(values.reshape(1, c, 1, p))
            result = np.empty(expected.size, np.int16)
            actual_scale = self.transport.clip_pack(raw, result, factors, bias, c, p)
            self.assertEqual(actual_scale, float(np.float32(scale)))
            np.testing.assert_array_equal(result, expected.reshape(-1))
            self.assertLess(actual_scale, 6/32767)

    def test_invalid_inputs_fail_before_writing_destination(self):
        raw = np.ones(3, np.int64)
        output = np.full(8, 91, np.int16)
        for factors, bias in (([np.nan]*3, [0]*3), ([1]*3, [np.inf]*3),
                              ([-1]*3, [0]*3), ([np.finfo(np.float32).max]*3, [0]*3)):
            source = raw if factors[0] != np.finfo(np.float32).max else raw*2
            with self.assertRaises(ValueError):
                self.transport.clip_pack(source, output, np.array(factors, np.float32),
                                         np.array(bias, np.float32), 3, 1)
            np.testing.assert_array_equal(output, 91)
        with self.assertRaises(ValueError):
            self.transport.clip_pack(raw, output[:2], np.ones(3, np.float32), np.zeros(3, np.float32), 3, 1)
        with self.assertRaises(ValueError):
            self.transport.clip_pack(raw, output, np.ones(3, np.float32), np.zeros(3, np.float32), 3, 131073)

    def test_deployed_native_unpack_clip_pack_bit_exact_when_available(self):
        path = Path(os.environ.get('FINGERTIP_BASELINE_LIBRARY', str(ROOT/'pynq/libhand_q16.so')))
        if not path.is_file():
            self.skipTest('deployed libhand_q16.so unavailable on host')
        native = ctypes.CDLL(str(path))
        native.hand_q16_unpack.argtypes = [ctypes.c_void_p]*4+[ctypes.c_float, ctypes.c_int, ctypes.c_int]
        native.hand_q16_unpack.restype = None
        native.hand_q16_pack.argtypes = [ctypes.c_void_p]*2+[ctypes.c_int]*2+[ctypes.POINTER(ctypes.c_float)]
        native.hand_q16_pack.restype = ctypes.c_int
        rng = np.random.default_rng(42)
        for c, p in ((3, 1), (3, 17), (8, 35), (9, 16), (32, 10816), (64, 2704)):
            for magnitude in (2**24, 2**31, 2**47, 2**62):
                raw = rng.integers(-magnitude, magnitude, (c, p), dtype=np.int64)
                scales = np.geomspace(1/magnitude, 100/magnitude, c).astype(np.float32)
                bias = rng.uniform(-2, 2, c).astype(np.float32)
                input_scale = np.float32(.13)
                values = np.empty((c, p), np.float32)
                native.hand_q16_unpack(raw.ctypes.data, values.ctypes.data, scales.ctypes.data,
                                       bias.ctypes.data, float(input_scale), c, p)
                np.clip(values, 0, 6, out=values)
                expected = np.empty(p*((c+7)//8*8), np.int16)
                expected_scale = ctypes.c_float()
                self.assertEqual(native.hand_q16_pack(values.ctypes.data, expected.ctypes.data,
                    c, p, ctypes.byref(expected_scale)), 0)
                actual = np.empty_like(expected)
                scale = self.transport.clip_pack(raw, actual, scales*input_scale, bias, c, p)
                self.assertEqual(scale, expected_scale.value)
                np.testing.assert_array_equal(actual, expected)

    def test_large_threaded_workspace_and_compatibility_abi_match(self):
        rng = np.random.default_rng(93)
        for c, p in ((3, 43264), (16, 8192), (32, 10816), (64, 2705)):
            raw = rng.integers(-2**30, 2**30, (c, p), dtype=np.int64)
            factors = np.full(c, 1e-8, np.float32)
            bias = rng.uniform(-2, 2, c).astype(np.float32)
            output = np.empty(p*((c+7)//8*8), np.int16)
            scratch_guard = np.full(c*p+16, -12345, np.float32)
            scratch = scratch_guard[8:-8]
            scale = self.transport.clip_pack(raw, output, factors, bias, c, p, scratch)
            expected = np.empty_like(output)
            other_scale = self.transport.clip_pack(raw, expected, factors, bias, c, p)
            self.assertEqual(scale, other_scale)
            np.testing.assert_array_equal(output, expected)
            np.testing.assert_array_equal(scratch_guard[:8], -12345)
            np.testing.assert_array_equal(scratch_guard[-8:], -12345)
            if not hasattr(os, 'uname') or not os.uname().machine.startswith(('arm', 'aarch')):
                values = raw.astype(np.float32)*factors[:, None]+bias[:, None]
                np.clip(values, 0, 6, out=values)
                packed, reference_scale = quantize_activation(values.reshape(1, c, 1, p))
                np.testing.assert_array_equal(output, packed.reshape(-1))
                self.assertEqual(scale, float(np.float32(reference_scale)))

    def test_two_pl_calls_equal_baseline_and_have_explicit_cache_handoffs(self):
        rng = np.random.default_rng(13)
        for c, o, k, g in ((3, 9, 3, 1), (9, 9, 3, 9), (13, 11, 1, 1)):
            engine, baseline = self.engine(), test_hand_q16.Q16Tests().engine()
            x = rng.normal(size=(1, c, 5, 7)).astype(np.float32)
            weights = rng.normal(size=(o, c//g, k, k)).astype(np.float32)
            bias = rng.normal(size=o).astype(np.float32)
            next_weights = rng.normal(size=(3, o, 1, 1)).astype(np.float32)
            pads = (k//2,)*4
            ref = baseline.convolve(x, weights, bias, 1, pads, g)
            ref = baseline.convolve(np.clip(ref, 0, 6), next_weights, np.zeros(3, np.float32), 1, (0,)*4, 1)
            engine.events.clear()
            token = engine.convolve(x, weights, bias, 1, pads, g, defer_output=True)
            self.assertFalse(any(event[0] == 'invalidate' for event in engine.events))
            token.clipped(0, 6)
            # Fusion must not materialize a full float intermediate.
            with patch.object(engine, '_materialize', side_effect=AssertionError('unexpected materialize')):
                actual = engine.convolve(token, next_weights, np.zeros(3, np.float32), 1, (0,)*4, 1)
            np.testing.assert_array_equal(actual, ref)
            self.assertEqual(engine.calls, 2)
            self.assertEqual(engine.transport_calls, 1)
            events = [name for name, _ in engine.events]
            self.assertEqual(events, ['run', 'invalidate', 'flush', 'run', 'invalidate'])
            with self.assertRaisesRegex(RuntimeError, 'overwritten'):
                token.check()

    def test_materialize_foreign_stale_and_missing_clip(self):
        engine = self.engine()
        args = (np.ones((1, 1, 1, 1), np.float32), np.zeros(1, np.float32), 1, (0,)*4, 1)
        token = engine.convolve(np.ones((1, 1, 1, 3), np.float32), *args, defer_output=True)
        np.testing.assert_allclose(engine.materialize(token), 1, atol=1e-6)
        with self.assertRaisesRegex(ValueError, 'requires Clip'):
            engine.convolve(token, *args)
        with self.assertRaisesRegex(ValueError, 'another engine'):
            self.engine().convolve(token.clipped(0, 6), *args)
        engine.convolve(np.ones((1, 1, 1, 3), np.float32), *args)
        with self.assertRaisesRegex(RuntimeError, 'overwritten'):
            engine.materialize(token)

    def test_inherited_timeout_quarantines_buffers(self):
        from vision_hls import _QUARANTINE
        engine = self.engine()
        del engine.run
        engine._register_values = {}
        engine.mmio = type('MMIO', (), {'read': lambda s, o: 4, 'write': lambda *a: None})()
        engine.timeout = -1
        buffers = engine.buffers[:]
        before = len(_QUARANTINE)
        try:
            with self.assertRaises(TimeoutError):
                engine.run([], [])
            self.assertTrue(engine.fault)
            self.assertEqual(engine.buffers, [])
            self.assertTrue(all(a is b for a, b in zip(_QUARANTINE[before:], buffers)))
            self.assertEqual(len(_QUARANTINE)-before, 3)
        finally:
            del _QUARANTINE[before:]

    def test_native_wait_done_timeout_and_invalid_timeout(self):
        wait = self.transport.wait_function
        if wait is None:
            self.skipTest('Native completion wait unavailable')
        control = np.array([2], np.uint32)
        self.assertEqual(wait(control.ctypes.data, .05), 0)
        control[0] = 0
        started = time.monotonic()
        self.assertEqual(wait(control.ctypes.data, .015), 1)
        self.assertGreaterEqual(time.monotonic()-started, .015)
        for timeout in (0., -1., float('nan')):
            self.assertEqual(wait(control.ctypes.data, timeout), -1)
        def complete():
            time.sleep(.01)
            control[0] = 2
        worker = threading.Thread(target=complete)
        worker.start()
        try:
            self.assertEqual(wait(control.ctypes.data, .5), 0)
        finally:
            worker.join()

    def test_native_driver_wait_keeps_register_cache_and_quarantines(self):
        from vision_hls import _QUARANTINE
        engine = self.engine()
        del engine.run
        engine._register_values = {}
        engine.timeout = .5
        class MMIO:
            array = np.array([4], np.uint32)
            def __init__(self):
                self.writes = []
            def read(self, offset):
                return 4
            def write(self, offset, value):
                self.writes.append((offset, value))
                if offset == 0:
                    self.array[0] = 2
        engine.mmio = MMIO()
        with patch.dict(os.environ, {'FRETLENS_FINGERTIP_NATIVE_WAIT': '1'}):
            engine.run([], [(16, 3)])
            engine.run([], [(16, 3)])
            self.assertEqual(engine.mmio.writes, [(16, 3), (0, 1), (0, 1)])
            self.assertEqual(engine.output_generation, 2)
            buffers = engine.buffers[:]
            before = len(_QUARANTINE)
            old_wait = self.transport.wait_function
            self.transport.wait_function = lambda *_: 1
            try:
                with self.assertRaises(TimeoutError):
                    engine.run([], [])
                self.assertTrue(engine.fault)
                self.assertEqual(engine.buffers, [])
                self.assertEqual(engine._register_values, {})
                self.assertTrue(all(a is b for a,b in zip(_QUARANTINE[before:], buffers)))
            finally:
                self.transport.wait_function = old_wait
                del _QUARANTINE[before:]

    def test_graph_identity_rejection_precedes_mmio_and_clears_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            engine = self.engine()
            engine._register_values = {16: 3}
            engine.processor = type('Processor', (), {'_ensure_active_overlay':
                lambda _: (_ for _ in ()).throw(RuntimeError('overlay changed'))})()
            graph = self.graph(folder, engine)
            graph.setInput(np.ones((1, 3, 2, 3), np.float32))
            with self.assertRaisesRegex(RuntimeError, 'overlay changed'):
                graph.forward()
            self.assertEqual(engine.events, [])
            self.assertEqual(engine._register_values, {})

    def graph(self, directory, engine, extra=None, outputs=None, clip_max=6):
        root = Path(directory)
        np.savez(root/'weights.npz', w=np.ones((3, 3, 1, 1), np.float32), b=np.array([-.3, .2, .1], np.float32))
        nodes = [dict(op='PLConvolution', inputs=['x', 'w', 'b'], output='a', attrs={}),
                 dict(op='Clip', inputs=['a'], output='bclip', attrs=dict(min=0, max=clip_max)),
                 dict(op='PLConvolution', inputs=['bclip', 'w', 'b'], output='y', attrs={})]
        nodes.extend(extra or [])
        plan = dict(version=1, inputs=['x'], outputs=outputs or ['y'], nodes=nodes,
                    weights_sha256=hashlib.sha256((root/'weights.npz').read_bytes()).hexdigest())
        (root/'graph.json').write_text(json.dumps(plan))
        return FingertipGraph(root, engine)

    def test_graph_fuses_and_capture_and_diagnostic_outputs_do_not(self):
        with tempfile.TemporaryDirectory() as folder:
            graph = self.graph(folder, self.engine())
            x = np.linspace(-1, 1, 45, dtype=np.float32).reshape(1, 3, 3, 5)
            graph.setInput(x)
            fused = graph.forward()[0]
            self.assertEqual(graph.transport_fused_calls, 1)
            self.assertEqual(graph.engine.calls, 2)
            graph.capture = True
            np.testing.assert_array_equal(graph.forward()[0], fused)
            self.assertEqual(graph.transport_fused_calls, 0)
            self.assertEqual(len(graph.captured), 3)
            graph.capture = False
            diagnostic = graph.forward(['a', 'bclip', 'y'])
            np.testing.assert_array_equal(diagnostic[-1], fused)
            self.assertEqual(graph.transport_fused_calls, 0)
            self.assertEqual(len(graph.timings), 3)

    def test_shared_edges_and_graph_outputs_and_non_relu6_are_not_fused(self):
        variants = [dict(extra=[dict(op='Add', inputs=['a', 'y'], output='sum', attrs={})], outputs=['sum']),
                    dict(extra=[dict(op='Add', inputs=['bclip', 'y'], output='sum', attrs={})], outputs=['sum']),
                    dict(outputs=['a', 'y']), dict(outputs=['bclip', 'y']), dict(clip_max=5)]
        for options in variants:
            with tempfile.TemporaryDirectory() as folder:
                graph = self.graph(folder, self.engine(), **options)
                graph.setInput(np.ones((1, 3, 2, 3), np.float32))
                graph.forward()
                self.assertEqual(graph.transport_fused_calls, 0)

    def test_208_graph_edge_selection_is_available_without_hardware(self):
        directory = ROOT/'models/hand/handpose_execution'
        if not directory.is_dir():
            self.skipTest('compiled hand graph unavailable')
        graph = FingertipGraph(directory, self.engine())
        self.assertGreater(len(graph.transport_pairs), 0)
        for index in graph.transport_pairs:
            self.assertEqual(graph.nodes[index+1]['attrs']['max'], 6)


if __name__ == '__main__':
    unittest.main()
