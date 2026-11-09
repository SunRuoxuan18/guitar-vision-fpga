"""Fixed palm graph scheduler; learned convolutions are always executed on PL."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np


def output_order_sha256(outputs):
    return hashlib.sha256('\0'.join(outputs).encode('utf-8')).hexdigest()


class PalmGraph:
    """Execute the fixed palm detector with PS tensor ops and PL convolution MACs."""
    def __init__(self, directory, engine, model_path=None):
        root = Path(directory)
        plan = json.loads((root/'graph.json').read_text())
        if plan.get('version') != 1 or len(plan.get('inputs', ())) != 1:
            raise ValueError('Unsupported compiled palm graph')
        if plan.get('output_order_sha256') != output_order_sha256(plan['outputs']):
            raise ValueError('Palm graph output order checksum mismatch')
        if model_path is not None and hashlib.sha256(Path(model_path).read_bytes()).hexdigest() != plan['source_sha256']:
            raise ValueError('Palm graph does not match the selected model')
        if hashlib.sha256((root/'weights.npz').read_bytes()).hexdigest() != plan['weights_sha256']:
            raise ValueError('Palm graph weight checksum mismatch')
        with np.load(root/'weights.npz', allow_pickle=False) as arrays:
            self.weights = {name: np.ascontiguousarray(arrays[name]) for name in arrays.files}
        self.nodes = plan['nodes']
        self.outputs = plan['outputs']
        self.input_name = plan['inputs'][0]
        self.engine = engine
        self.uses = Counter(name for node in self.nodes for name in node['inputs'])
        self.uses.update(self.outputs)
        self.input = None
        self.timings = []

    def setInput(self, blob):
        self.input = np.asarray(blob, dtype=np.float32)

    def getLayerNames(self):
        return tuple(node['output'] for node in self.nodes)

    def getUnconnectedOutLayersNames(self):
        return tuple(self.outputs)

    def getPerfProfile(self):
        ticks = np.asarray(self.timings, np.float64)*cv2.getTickFrequency()/1000
        return float(np.sum(ticks)), ticks

    @staticmethod
    def _upsample(x, attrs):
        if attrs.get('mode') != 'bilinear' or attrs.get('scales') != [1., 1., 2., 2.]:
            raise ValueError('Unsupported palm graph upsample')
        if x.ndim != 4 or x.shape[0] != 1:
            raise ValueError('Palm graph upsample requires batch-one NCHW')
        # Legacy ONNX Upsample's bilinear coordinates are output/scale, unlike
        # cv2.resize's half-pixel convention.  The final row/column clamp.
        right = np.concatenate((x[..., 1:], x[..., -1:]), axis=3)
        horizontal = np.empty((x.shape[0], x.shape[1], x.shape[2], x.shape[3]*2), np.float32)
        horizontal[..., ::2] = x
        horizontal[..., 1::2] = (x+right)*.5
        below = np.concatenate((horizontal[:, :, 1:], horizontal[:, :, -1:]), axis=2)
        out = np.empty((x.shape[0], x.shape[1], x.shape[2]*2, x.shape[3]*2), np.float32)
        out[:, :, ::2] = horizontal
        out[:, :, 1::2] = (horizontal+below)*.5
        return out

    @staticmethod
    def _reshape(x, shape):
        target = [x.shape[index] if value == 0 else int(value) for index, value in enumerate(shape)]
        if target.count(-1) > 1:
            raise ValueError('Invalid palm graph reshape')
        return x.reshape(target)

    def forward(self, names=None):
        if self.input is None:
            raise ValueError('Palm graph input is missing')
        values = dict(self.weights)
        values[self.input_name] = self.input
        uses = self.uses.copy()
        requested = self.outputs if names is None else list(names)
        uses.update(name for name in requested if name not in self.outputs)
        self.timings = []
        for node in self.nodes:
            started = time.perf_counter()
            args = [values[name] for name in node['inputs']]
            op, attrs, x = node['op'], node['attrs'], args[0]
            if op == 'PLConvolution':
                strides = attrs.get('strides', [1, 1])
                if strides[0] != strides[1] or attrs.get('dilations', [1, 1]) != [1, 1]:
                    raise ValueError('Unsupported palm graph convolution')
                bias = args[2] if len(args) > 2 else np.zeros(args[1].shape[0], np.float32)
                out = self.engine.convolve(x, args[1], bias, strides[0], tuple(attrs.get('pads', [0]*4)),
                                           attrs.get('group', 1))
            elif op == 'Transpose':
                out = np.transpose(x, attrs['perm'])
            elif op == 'Pad':
                if attrs.get('mode', 'constant') != 'constant':
                    raise ValueError('Unsupported palm graph padding')
                pads = attrs['pads']
                out = np.pad(x, tuple(zip(pads[:x.ndim], pads[x.ndim:])), constant_values=attrs.get('value', 0))
            elif op == 'Add':
                out = np.add(x, args[1])
            elif op == 'MaxPool':
                if attrs.get('kernel_shape') != [2, 2] or attrs.get('strides') != [2, 2] or any(attrs.get('pads', [0]*4)):
                    raise ValueError('Unsupported palm graph pooling')
                h, w = x.shape[-2]//2, x.shape[-1]//2
                out = np.maximum(np.maximum(x[..., :2*h:2, :2*w:2], x[..., 1:2*h:2, :2*w:2]),
                                 np.maximum(x[..., :2*h:2, 1:2*w:2], x[..., 1:2*h:2, 1:2*w:2]))
            elif op == 'PRelu':
                out = np.where(x >= 0, x, x*args[1])
            elif op == 'Upsample':
                out = self._upsample(x, attrs)
            elif op == 'Reshape':
                out = self._reshape(x, args[1].reshape(-1))
            elif op == 'Concat':
                out = np.concatenate(args, axis=attrs.get('axis', 1))
            else:
                raise ValueError('Unsupported palm graph operation: '+op)
            values[node['output']] = out
            self.timings.append((time.perf_counter()-started)*1000)
            for name in node['inputs']:
                uses[name] -= 1
                if uses[name] == 0 and name not in self.weights:
                    values.pop(name, None)
        return [values[name] for name in requested]
