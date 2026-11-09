"""Palm direct-graph regression tests; all learned MACs must route to the engine."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'pynq'))
sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_hand_execution import export, output_order_sha256
from palm_graph import PalmGraph
from test_hand_neural_pl import ReferenceEngine


class Engine:
    def __init__(self):
        self.calls = []

    def convolve(self, x, weights, bias, stride, pads, groups):
        self.calls.append((x.copy(), weights.copy(), bias.copy(), stride, pads, groups))
        return x*weights.reshape(1, 1, 1, 1)+bias.reshape(1, 1, 1, 1)


class PalmGraphTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        np.savez(self.root/'weights.npz', w=np.array([[[[2.]]]], np.float32), b=np.array([1.], np.float32),
                 slope=np.array([[[[.25]]]], np.float32), shape=np.array([1, 1, 4, 4], np.int64))
        outputs = ['reshaped', 'joined']
        self.plan = dict(version=1, inputs=['x'], outputs=outputs,
            output_order_sha256=output_order_sha256(outputs),
            weights_sha256=hashlib.sha256((self.root/'weights.npz').read_bytes()).hexdigest(), nodes=[
                dict(op='PLConvolution', inputs=['x', 'w', 'b'], output='conv', attrs={}),
                dict(op='PRelu', inputs=['conv', 'slope'], output='prelu', attrs={}),
                dict(op='Upsample', inputs=['prelu'], output='up', attrs=dict(mode='bilinear', scales=[1.,1.,2.,2.])),
                dict(op='Reshape', inputs=['up', 'shape'], output='reshaped', attrs={}),
                dict(op='Concat', inputs=['reshaped', 'reshaped'], output='joined', attrs=dict(axis=1)),
            ])
        (self.root/'graph.json').write_text(json.dumps(self.plan))

    def test_routes_learned_convolution_to_engine_and_matches_reference_tensor_ops(self):
        engine = Engine()
        graph = PalmGraph(self.root, engine)
        source = np.array([[[[-2., 1.], [3., -4.]]]], np.float32)
        graph.setInput(source)
        reshaped, joined = graph.forward()
        convolved = source*2+1
        prelu = np.where(convolved >= 0, convolved, convolved*.25)
        expected = np.array([[-.75, 1.125, 3., 3.],
                             [3.125, 1.875, .625, .625],
                             [7., 2.625, -1.75, -1.75],
                             [7., 2.625, -1.75, -1.75]], np.float32)[None, None]
        np.testing.assert_allclose(reshaped, expected)
        np.testing.assert_allclose(joined, np.concatenate((expected, expected), axis=1))
        self.assertEqual(len(engine.calls), 1)
        self.assertEqual(engine.calls[0][3:], (1, (0, 0, 0, 0), 1))
        self.assertEqual(graph.getUnconnectedOutLayersNames(), ('reshaped', 'joined'))

    def test_rejects_changed_output_order_before_execution(self):
        self.plan['outputs'] = list(reversed(self.plan['outputs']))
        (self.root/'graph.json').write_text(json.dumps(self.plan))
        with self.assertRaisesRegex(ValueError, 'output order checksum'):
            PalmGraph(self.root, Engine())

    def test_exports_actual_palm_graph_with_complete_operator_and_output_contract(self):
        destination = self.root/'palm_execution'
        source = ROOT/'models/hand/palm_detection_mediapipe_2023feb_pl.onnx'
        export(source, destination)
        plan = json.loads((destination/'graph.json').read_text())
        self.assertEqual(plan['outputs'], ['Identity', 'Identity_1'])
        self.assertEqual(plan['output_order_sha256'], output_order_sha256(plan['outputs']))
        self.assertEqual(plan['source_sha256'], hashlib.sha256(source.read_bytes()).hexdigest())
        self.assertEqual(hashlib.sha256((destination/'weights.npz').read_bytes()).hexdigest(), plan['weights_sha256'])
        ops = [node['op'] for node in plan['nodes']]
        self.assertEqual(ops.count('PLConvolution'), 53)
        self.assertEqual({op for op in ops} - {'PLConvolution', 'Transpose', 'Pad', 'Add', 'MaxPool', 'PRelu', 'Upsample', 'Reshape', 'Concat'}, set())

    def test_actual_palm_graph_matches_opencv_reference_in_declared_output_order(self):
        engine = ReferenceEngine()
        graph = PalmGraph(ROOT/'models/hand/palm_execution', engine,
                          ROOT/'models/hand/palm_detection_mediapipe_2023feb_pl.onnx')
        source = np.random.default_rng(20261022).random((1, 192, 192, 3), dtype=np.float32)
        graph.setInput(source)
        actual = graph.forward()
        reference = cv2.dnn.readNetFromONNX(str(ROOT/'models/hand/palm_detection_mediapipe_2023feb_cv42.onnx'))
        reference.setInput(source)
        expected = reference.forward(reference.getUnconnectedOutLayersNames())
        self.assertEqual(engine.calls, 53)
        self.assertEqual([item.shape for item in actual], [(1, 2016, 18), (1, 2016, 1)])
        for result, golden in zip(actual, expected):
            np.testing.assert_allclose(result, golden, rtol=3e-3, atol=5e-3)


if __name__ == '__main__':
    unittest.main()
