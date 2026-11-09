"""Quantization/packing boundaries; hardware arithmetic is checked by CSim."""
import sys
from pathlib import Path
import threading
import ctypes
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hand_neural_pl import Q16NeuralPL, quantize_activation, quantize_weights
from test_hand_neural_pl import ReferenceEngine


class Buffer(np.ndarray):
    cacheable = False
    flushed = []
    def flush(self):
        self.flushed.append(self.ctypes.data)
    def invalidate(self):
        pass


class Q16Tests(unittest.TestCase):
    def engine(self):
        engine = Q16NeuralPL.__new__(Q16NeuralPL)
        engine.buffers = [np.zeros(16000,dtype=t).view(Buffer) for t in (np.int16,np.int16,np.int64)]
        engine.lock = threading.Lock()
        engine.stopping = engine.closed = engine.fault = False
        engine.weight_cache = {}
        engine.weight_used = engine.calls = 0
        engine.output_generation = 0
        engine.requant = None
        engine.kernel_ms = engine.total_ms = 0.
        engine.profile = True
        engine.native = None
        engine.layer_profile = []
        engine.offset = lambda n: n
        engine.copy = lambda dst,src: np.copyto(dst,src.reshape(-1))
        def run(pointers, scalar_items):
            params = dict(scalar_items)
            c,h,w,o,k,g = [params[n] for n in ('channels','height','width','outputs','kernel','groups')]
            pointer_map = dict(pointers)
            x = pointer_map['input'][:h*w*((c+7)//8*8)].reshape(h,w,-1)[:,:,:c].transpose(2,0,1)[None].astype(np.float64)
            if g == c == o:
                weights = pointer_map['weights'].reshape(k,k,-1)[:,:,:o].transpose(2,0,1)[:,None]
            else:
                weights = pointer_map['weights'].reshape(o,k,k,-1)[:,:,:,:c].transpose(0,3,1,2)
            # Float64 represents every integer product/sum here exactly.
            oh,ow = params['out_h'],params['out_w']
            top,left = params['pad_top'],params['pad_left']
            bottom = max(0,(oh-1)*params['stride']+k-h-top)
            right = max(0,(ow-1)*params['stride']+k-w-left)
            result = ReferenceEngine().convolve(x,weights,np.zeros(o),params['stride'],(top,left,bottom,right),g)
            engine.buffers[2][:result.size] = result.reshape(-1).astype(np.int64)
            return .01
        engine.run = run
        return engine

    def test_padded_activation_pack_and_zero(self):
        x = np.arange(3*5*7,dtype=np.float32).reshape(1,3,5,7)/13
        packed,scale = quantize_activation(x)
        self.assertEqual(packed.shape,(5,7,8))
        self.assertFalse(packed[:,:,3:].any())
        np.testing.assert_allclose(packed[:,:,:3].transpose(2,0,1)*scale,x[0],atol=scale*.501,rtol=0)
        packed,scale = quantize_activation(np.zeros_like(x))
        self.assertFalse(packed.any())
        self.assertGreater(scale,0)

    def test_nonfinite_never_reaches_hardware(self):
        for invalid in (np.nan,np.inf,-np.inf):
            with self.assertRaises(ValueError):
                quantize_activation(np.full((1,3,1,1),invalid,np.float32))
            with self.assertRaises(ValueError):
                quantize_weights(np.full((3,1,1,1),invalid,np.float32))

    def test_native_pack_writes_cached_operand_and_flushes_before_run(self):
        engine = self.engine()
        engine.buffers[0].cacheable = True
        packed_addresses = []
        class Native:
            def hand_q16_pack(self,source,target,channels,pixels,scale):
                packed_addresses.append(target)
                values = np.ctypeslib.as_array(ctypes.cast(target,ctypes.POINTER(ctypes.c_int16)),
                                              shape=(pixels*((channels+7)//8*8),))
                values[:] = 0
                values.reshape(pixels,-1)[:,:channels] = 3
                scale._obj.value = 1.
                return 0
            def hand_q16_unpack(self,source,target,scales,bias,scale,channels,pixels):
                raw = np.ctypeslib.as_array(ctypes.cast(source,ctypes.POINTER(ctypes.c_int64)),shape=(channels*pixels,))
                factors = np.ctypeslib.as_array(ctypes.cast(scales,ctypes.POINTER(ctypes.c_float)),shape=(channels,))
                result = np.ctypeslib.as_array(ctypes.cast(target,ctypes.POINTER(ctypes.c_float)),shape=(channels*pixels,))
                result[:] = (raw.reshape(channels,pixels)*factors[:,None]*scale).reshape(-1)
        engine.native = Native()
        original_run = engine.run
        def checked_run(*args):
            self.assertIn(engine.buffers[0].ctypes.data,Buffer.flushed)
            return original_run(*args)
        engine.run = checked_run
        Buffer.flushed.clear()
        result = engine.convolve(np.ones((1,3,5,7),np.float32)*3,
            np.ones((2,3,1,1),np.float32),np.zeros(2,np.float32),1,(0,0,0,0),1)
        self.assertEqual(packed_addresses,[engine.buffers[0].ctypes.data])
        np.testing.assert_allclose(result,9,atol=1e-6)

    def test_per_output_scaling_preserves_small_weight_channels(self):
        weights = np.stack([np.full((3,3,3),v,np.float32) for v in (1e-5,.1,10.)])
        packed,scales = quantize_weights(weights)
        restored = packed[:,:,:,:3].transpose(0,3,1,2)*scales[:,None,None,None]
        np.testing.assert_allclose(restored,weights,rtol=1e-6)

    def test_driver_regular_depthwise_and_dense_match_float_reference(self):
        rng = np.random.default_rng(15)
        for c,o,k,g,s,pads in ((3,9,3,1,2,(0,1,1,0)),(9,9,5,9,1,(2,2,2,2)),
                               (13,11,1,1,1,(0,0,0,0)),(9,9,1,9,1,(0,0,0,0))):
            with self.subTest(c=c,o=o,k=k,g=g):
                engine = self.engine()
                x = rng.normal(size=(1,c,7,9)).astype(np.float32)
                weights = rng.normal(size=(o,c//g,k,k)).astype(np.float32)
                bias = rng.normal(size=o).astype(np.float32)
                expected = ReferenceEngine().convolve(x,weights,bias,s,pads,g)
                actual = engine.convolve(x,weights,bias,s,pads,g)
                np.testing.assert_allclose(actual,expected,atol=.003,rtol=.003)
                used = engine.weight_used
                np.testing.assert_array_equal(engine.convolve(x,weights,bias,s,pads,g),actual)
                self.assertEqual(engine.weight_used,used)
                self.assertEqual(len(engine.weight_cache),1)
                self.assertEqual(engine.calls,2)


if __name__ == '__main__':
    unittest.main()
