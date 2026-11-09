"""Fusion scheduling must preserve graph ownership and diagnostic outputs."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hand_graph import HandGraph


class Engine:
    requant = None

    def __init__(self):
        self.calls = 0
        self.stream = self
        self.fused = 0
        self.accept = True

    def supports(self, *args):
        return self.accept

    def convolve(self, x, weights, bias, *args):
        # Fused signature has a second weight and bias before stride/padding.
        if len(args) == 4:
            pw, pb, stride, pads = args
            self.fused += 1
            intermediate = np.clip(self._conv(x, weights, bias), 0, 6)
            return self._conv(intermediate, pw, pb)
        self.calls += 1
        return self._conv(x, weights, bias)

    @staticmethod
    def _conv(x, weights, bias):
        if weights.shape[1] == 1:
            return x * weights[:, 0, 1, 1][None, :, None, None] + bias[None, :, None, None]
        return np.einsum('nchw,oc->nohw', x, weights[:, :, 0, 0]) + bias[None, :, None, None]


class GraphStreamTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        dw = np.zeros((2, 1, 3, 3), np.float32)
        dw[:, 0, 1, 1] = [2, 3]
        np.savez(self.root / 'weights.npz', dw=dw, db=np.zeros(2, np.float32),
                 pw=np.ones((3, 2, 1, 1), np.float32), pb=np.ones(3, np.float32))
        self.nodes = [
            dict(op='PLConvolution', inputs=['x', 'dw', 'db'], output='depth', attrs=dict(group=2, pads=[1]*4)),
            dict(op='Clip', inputs=['depth'], output='relu', attrs=dict(min=0, max=6)),
            dict(op='PLConvolution', inputs=['relu', 'pw', 'pb'], output='point', attrs={}),
            dict(op='Add', inputs=['point', 'point'], output='result', attrs={}),
        ]

    def graph(self, outputs=None):
        plan = dict(version=1, inputs=['x'], outputs=outputs or ['result'], nodes=self.nodes,
                    weights_sha256=hashlib.sha256((self.root/'weights.npz').read_bytes()).hexdigest())
        (self.root/'graph.json').write_text(json.dumps(plan))
        self.engine = Engine()
        graph = HandGraph(self.root, self.engine)
        graph.setInput(np.arange(8, dtype=np.float32).reshape(1, 2, 2, 2))
        return graph

    def test_fused_matches_unfused_with_final_fanout(self):
        graph = self.graph()
        with patch.dict('os.environ', HAND_STREAM_FUSION='0'):
            expected = graph.forward()[0]
        with patch.dict('os.environ', HAND_STREAM_FUSION='1'):
            actual = graph.forward()[0]
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(self.engine.fused, 1)
        self.assertEqual(graph.stream_fused_calls, 1)
        self.assertEqual(len(graph.timings), 4)

    def test_requested_intermediate_disables_fusion_and_is_retained(self):
        graph = self.graph()
        with patch.dict('os.environ', HAND_STREAM_FUSION='1'):
            depth, result = graph.forward(['depth', 'result'])
        self.assertEqual(depth.shape, (1, 2, 2, 2))
        self.assertEqual(result.shape, (1, 3, 2, 2))
        self.assertEqual(self.engine.fused, 0)

    def test_model_intermediate_output_disables_fusion(self):
        graph = self.graph(['depth', 'result'])
        with patch.dict('os.environ', HAND_STREAM_FUSION='1'):
            self.assertEqual(len(graph.forward(['result'])), 1)
        self.assertEqual(self.engine.fused, 0)

    def test_requested_intermediate_cannot_borrow_requant_accumulator(self):
        graph = self.graph()
        self.engine.requant = object()
        recorded = []
        original = self.engine.convolve
        def convolve(*args, **kwargs):
            recorded.append(kwargs.get('defer_output'))
            return original(*args)
        self.engine.convolve = convolve
        with patch.dict('os.environ', HAND_STREAM_FUSION='1'):
            graph.forward(['depth', 'result'])
        self.assertEqual(recorded, [False, False])

    def test_capture_and_unsupported_shapes_use_existing_engine(self):
        graph = self.graph()
        with patch.dict('os.environ', HAND_STREAM_FUSION='1'):
            graph.capture = True
            graph.forward()
            self.assertEqual(set(graph.captured), {'depth', 'relu', 'point', 'result'})
            graph.capture = False
            self.engine.accept = False
            graph.forward()
        self.assertEqual(self.engine.fused, 0)
        self.assertEqual(self.engine.calls, 4)

    def test_shared_depthwise_output_is_not_fused(self):
        self.nodes.append(dict(op='Add', inputs=['result', 'depth'], output='extra', attrs={}))
        graph = self.graph()
        self.assertEqual(graph.stream_fusions, {})

    def test_unrequested_neural_output_branch_is_not_executed(self):
        self.nodes.append(dict(op='PLConvolution', inputs=['x', 'side', 'sb'],
                               output='unused_world', attrs={}))
        np.savez(self.root / 'weights.npz', dw=np.zeros((2, 1, 3, 3), np.float32),
                 db=np.zeros(2, np.float32), pw=np.ones((3, 2, 1, 1), np.float32),
                 pb=np.ones(3, np.float32), side=np.ones((1, 2, 1, 1), np.float32),
                 sb=np.zeros(1, np.float32))
        self.engine = None
        graph = self.graph(['result', 'unused_world'])

        with patch.dict('os.environ', HAND_STREAM_FUSION='0'):
            result = graph.forward(['result'])

        self.assertEqual(len(result), 1)
        self.assertEqual(self.engine.calls, 2)
        self.assertNotIn('unused_world', graph.captured)
        self.assertEqual(len(graph.timings), len(self.nodes))


if __name__ == '__main__':
    unittest.main()
