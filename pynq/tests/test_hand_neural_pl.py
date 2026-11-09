"""Check graph-to-accelerator boundaries; these are not hardware speed claims."""
import sys
from pathlib import Path
import unittest
import threading
import weakref

import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hand_neural_pl import PLConvolution, PLDense, NeuralPL, register


class ReferenceEngine:
    def __init__(self):
        self.calls = 0

    def convolve(self,x,weights,bias,stride,pads,groups):
        self.calls += 1
        top,left,bottom,right = pads
        x = np.pad(x,((0,0),(0,0),(top,bottom),(left,right)))
        kh,kw = weights.shape[2:]
        windows = np.lib.stride_tricks.sliding_window_view(x,(kh,kw),axis=(2,3))[:,:,::stride,::stride]
        if groups == 1:
            y = np.einsum('nchwij,ocij->nohw',windows,weights,optimize=True)
        else:
            y = np.einsum('nchwij,cij->nchw',windows,weights[:,0],optimize=True)
        return y+bias.reshape(1,-1,1,1)


class NeuralLayerTests(unittest.TestCase):
    def tearDown(self):
        register(None)

    def test_no_cpu_fallback(self):
        register(None)
        layer = PLConvolution({},[np.ones((1,1,1,1),np.float32)])
        with self.assertRaises(RuntimeError):
            layer.forward([np.ones((1,1,2,2),np.float32)])

    def test_hls_reserved_port_names_use_hwh_offsets(self):
        engine = NeuralPL.__new__(NeuralPL)
        engine.registers = {'input_r':{'address_offset':'0x10'},
                            'output_r':{'address_offset':0x34}}
        self.assertEqual(engine.offset('input'),0x10)
        self.assertEqual(engine.offset('output'),0x34)
        engine.registers = {'input_r_1':{'address_offset':0x10},
                            'weights_1':{'address_offset':0x1c}}
        self.assertEqual(engine.offset('input'),0x10)
        self.assertEqual(engine.offset('weights'),0x1c)

    def test_asymmetric_output_shape_and_bias(self):
        engine = ReferenceEngine()
        register(engine)
        layer = PLConvolution({'stride':(2,2),'pad':(0,1,1,0)},
            [np.ones((2,3,3,3),np.float32),np.array([1,2],np.float32)])
        x = np.ones((1,3,6,6),np.float32)
        result = layer.forward([x])[0]
        self.assertEqual(list(result.shape),layer.getMemoryShapes([x.shape])[0])
        np.testing.assert_allclose(result[:,1]-result[:,0],1)
        self.assertEqual(engine.calls,1)

    def test_dense_routes_to_same_pl_engine(self):
        engine = ReferenceEngine()
        register(engine)
        weights = np.arange(20,dtype=np.float32).reshape(4,5)
        x = np.arange(5,dtype=np.float32).reshape(1,5)
        layer = PLDense({},[weights,np.arange(4,dtype=np.float32)])
        np.testing.assert_allclose(layer.forward([x])[0],x@weights.T+np.arange(4))
        self.assertEqual(engine.calls,1)

    def test_cv42_retained_returns_do_not_retain_old_tensors(self):
        register(ReferenceEngine())
        layers = (
            (PLConvolution({},[np.ones((1,1,1,1),np.float32)]), (1,1,8,8)),
            (PLDense({},[np.ones((2,8),np.float32),np.zeros(2,np.float32)]), (1,8)),
        )
        for layer,shape in layers:
            retained_returns, tensors = [], []
            for value in range(30):
                # OpenCV 4.2's pycvLayer retains the returned Python list.
                result = layer.forward([np.full(shape,value,np.float32)])
                retained_returns.append(result)
                tensors.append(weakref.ref(result[0]))
            self.assertEqual(len({id(item) for item in retained_returns}),1)
            self.assertEqual(sum(item() is not None for item in tensors),1)
            self.assertGreater(float(result[0].flat[0]),0)

    def test_optional_profile_records_successful_hardware_call(self):
        class Buffer(np.ndarray):
            def invalidate(self):
                pass

        engine = NeuralPL.__new__(NeuralPL)
        engine.buffers = [np.zeros(16,np.float32).view(Buffer) for _ in range(4)]
        engine.lock = threading.Lock()
        engine.stopping = False
        engine.profile = True
        engine.layer_profile = []
        engine.calls = 0
        engine.kernel_ms = engine.total_ms = 0.
        engine.offset = lambda name: 0
        engine.copy = lambda dst,src: np.copyto(dst,src.reshape(-1))
        engine.run = lambda *args: 0.
        engine.buffers[3][:] = 7.
        result = engine.convolve(np.ones((1,2,2,2),np.float32),
            np.ones((3,2,1,1),np.float32),np.zeros(3,np.float32),1,(0,0,0,0),1)
        np.testing.assert_array_equal(result,7.)
        row = engine.layer_profile[0]
        self.assertEqual(row['kind'],'pointwise')
        self.assertEqual(row['weight_shape'],[3,2,1,1])
        self.assertEqual(row['call'],1)
        self.assertGreaterEqual(row['total_ms'],row['kernel_ms'])
        engine.profile = False
        engine.convolve(np.ones((1,2,2,2),np.float32),
            np.ones((3,2,1,1),np.float32),np.zeros(3,np.float32),1,(0,0,0,0),1)
        self.assertEqual(engine.calls,2)
        self.assertEqual(len(engine.layer_profile),1)


if __name__ == '__main__':
    unittest.main()
