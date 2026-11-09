"""Fixed hand graph scheduling on PS; all learned MACs remain on PL."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np


class HandGraph:
    def __init__(self, directory, engine, model_path=None):
        root = Path(directory)
        plan = json.loads((root/'graph.json').read_text())
        if plan['version'] != 1 or len(plan['inputs']) != 1:
            raise ValueError('Unsupported compiled hand graph')
        if model_path is not None and hashlib.sha256(Path(model_path).read_bytes()).hexdigest() != plan['source_sha256']:
            raise ValueError('Hand graph does not match the selected model')
        if hashlib.sha256((root/'weights.npz').read_bytes()).hexdigest() != plan['weights_sha256']:
            raise ValueError('Hand graph weight checksum mismatch')
        with np.load(root/'weights.npz', allow_pickle=False) as arrays:
            self.weights = {name: np.ascontiguousarray(arrays[name]) for name in arrays.files}
        self.nodes = plan['nodes']
        self.outputs = plan['outputs']
        self.input_name = plan['inputs'][0]
        self.producers = {node['output']: index for index, node in enumerate(self.nodes)}
        self.engine = engine
        self.uses = Counter(name for node in self.nodes for name in node['inputs'])
        self.uses.update(self.outputs)
        self.input = None
        self.timings = []
        self.capture = False
        self.hardware_requant = os.environ.get('HAND_PL_REQUANT', '1') == '1'
        self.captured = {}
        # Only adjacent, single-consumer edges may borrow the shared sums.
        self.deferred_nodes = set()
        self.stream_fusions = {}
        for index, node in enumerate(self.nodes[:-2]):
            clip, following = self.nodes[index+1:index+3]
            if (node['op'] == 'PLConvolution' and clip['op'] == 'Clip'
                    and following['op'] == 'PLConvolution'
                    and clip['inputs'][0] == node['output']
                    and following['inputs'][0] == clip['output']
                    and self.uses[node['output']] == self.uses[clip['output']] == 1):
                self.deferred_nodes.add(node['output'])
                first_weight = self.weights[node['inputs'][1]]
                second_weight = self.weights[following['inputs'][1]]
                attr_group = node['attrs'].get('group', 1)
                if attr_group == first_weight.shape[0] and attr_group > 1:
                    if (first_weight.shape[1] == 1 and first_weight.shape[2:] in ((3,3),(5,5))
                            and clip['attrs'].get('min') == 0 and clip['attrs'].get('max') == 6
                            and second_weight.shape[1] == first_weight.shape[0]
                            and second_weight.shape[2:] == (1,1)
                            and following['attrs'].get('group',1) == 1
                            and following['attrs'].get('strides',[1,1]) == [1,1]
                            and following['attrs'].get('dilations',[1,1]) == [1,1]
                            and not any(following['attrs'].get('pads',[0]*4))):
                        self.stream_fusions[index] = (clip, following)

    def setInput(self, blob):
        self.input = np.asarray(blob, dtype=np.float32)

    def getLayerNames(self):
        return tuple(node['output'] for node in self.nodes)

    def getLayer(self, index):
        from types import SimpleNamespace
        return SimpleNamespace(type=self.nodes[index-1]['op'])

    def getPerfProfile(self):
        import cv2
        ticks = np.asarray(self.timings, np.float64)*cv2.getTickFrequency()/1000
        return float(np.sum(ticks)), ticks

    def _required_nodes(self, requested):
        required = set()
        pending = list(requested)
        while pending:
            producer = self.producers.get(pending.pop())
            if producer is None or producer in required:
                continue
            required.add(producer)
            pending.extend(self.nodes[producer]['inputs'])
        return required

    def forward(self, names=None):
        if self.input is None:
            raise ValueError('Hand graph input is missing')
        values = dict(self.weights)
        values[self.input_name] = self.input
        requested = list(self.outputs if names is None else names)
        required = self._required_nodes(requested)
        uses = Counter(name for index in required for name in self.nodes[index]['inputs'])
        uses.update(requested)
        # Diagnostic outputs also own their tensors until this call returns.
        self.timings = []
        self.captured = {}
        skipped = set()
        self.stream_fused_calls = 0
        for index, node in enumerate(self.nodes):
            if index not in required or index in skipped:
                self.timings.append(0.)
                continue
            started = time.perf_counter()
            args = [values[name] for name in node['inputs']]
            op, attr = node['op'], node['attrs']
            x = args[0]
            executed = [node]
            output_name = node['output']
            if op == 'PLConvolution':
                strides = attr.get('strides', [1, 1])
                if strides[0] != strides[1] or attr.get('dilations', [1, 1]) != [1, 1]:
                    raise ValueError('Unsupported graph convolution')
                bias = args[2] if len(args) > 2 else np.zeros(args[1].shape[0], np.float32)
                stream = getattr(self.engine, 'stream', None)
                fusion = self.stream_fusions.get(index)
                pads = tuple(attr.get('pads', [0]*4))
                can_fuse = (os.environ.get('HAND_STREAM_FUSION','0') == '1' and
                    not self.capture and stream is not None and fusion is not None and
                    all(name not in requested and name not in self.outputs
                        for name in (node['output'],fusion[0]['output'])))
                if can_fuse:
                    clip, following = fusion
                    next_args = [values[name] for name in following['inputs'][1:]]
                    can_fuse = stream.supports(x, args[1], next_args[0], strides[0], pads)
                if can_fuse:
                    next_bias = next_args[1] if len(next_args)>1 else np.zeros(next_args[0].shape[0],np.float32)
                    out = stream.convolve(x,args[1],bias,next_args[0],next_bias,strides[0],pads)
                    executed = [node,clip,following]
                    output_name = following['output']
                    skipped.update((index+1,index+2))
                    self.stream_fused_calls += 1
                else:
                    kwargs = {}
                    if self.hardware_requant and getattr(self.engine, 'requant', None) is not None and not self.capture:
                        kwargs['defer_output'] = (node['output'] in self.deferred_nodes
                                                  and node['output'] not in requested)
                    out = self.engine.convolve(x, args[1], bias, strides[0], pads, attr.get('group', 1), **kwargs)
            elif op == 'PLDense':
                out = self.engine.convolve(x.reshape(1, -1, 1, 1), args[1].reshape(args[1].shape[0], -1, 1, 1),
                                           args[2], 1, (0, 0, 0, 0), 1).reshape(1, -1)
            elif op == 'Transpose':
                out = np.transpose(x, attr['perm'])
            elif op == 'Clip':
                from hand_requant import AccumulatorTensor
                # Reuse only a tensor with no other consumer, never parameters.
                reusable = uses[node['inputs'][0]] == 1 and node['inputs'][0] not in self.weights
                if isinstance(x, AccumulatorTensor):
                    out = x.clipped(attr.get('min', -np.finfo(np.float32).max), attr.get('max', np.finfo(np.float32).max))
                else:
                    out = x if reusable and x.flags.writeable and not self.capture else np.empty_like(x)
                    np.clip(x, attr.get('min', -np.inf), attr.get('max', np.inf), out=out)
            elif op == 'Add':
                out = np.add(x, args[1])
            elif op == 'Pad':
                if attr.get('mode', 'constant') != 'constant':
                    raise ValueError('Unsupported graph padding')
                pads = attr['pads']
                out = np.pad(x, tuple(zip(pads[:x.ndim], pads[x.ndim:])), constant_values=attr.get('value', 0))
            elif op == 'MaxPool':
                if attr.get('kernel_shape') != [2, 2] or attr.get('strides') != [2, 2] or any(attr.get('pads', [0]*4)):
                    raise ValueError('Unsupported graph pooling')
                h, w = x.shape[-2]//2, x.shape[-1]//2
                out = np.maximum(np.maximum(x[..., :2*h:2, :2*w:2], x[..., 1:2*h:2, :2*w:2]),
                                 np.maximum(x[..., :2*h:2, 1:2*w:2], x[..., 1:2*h:2, 1:2*w:2]))
            elif op == 'GlobalAveragePool':
                out = np.mean(x, axis=(2, 3), keepdims=True, dtype=np.float32)
            elif op == 'Flatten':
                axis = attr.get('axis', 1)
                out = x.reshape(int(np.prod(x.shape[:axis])), -1)
            elif op == 'Sigmoid':
                out = np.reciprocal(np.float32(1)+np.exp(-x))
            else:
                raise ValueError('Unsupported graph operation: '+op)
            values[output_name] = out
            if self.capture:
                self.captured[node['output']] = out.copy()
            self.timings.append((time.perf_counter()-started)*1000)
            for done in executed:
                for name in done['inputs']:
                    uses[name] -= 1
                    if uses[name] == 0 and name not in self.weights:
                        values.pop(name, None)
        return [values[name] for name in requested]
