"""Opt-in PS transport fusion for the existing hand_conv_q16 overlay.

Build native/fingertip_transport.c as libfingertip_transport.so beside this file.
Use FingertipQ16NeuralPL(processor, profile=True) and
FingertipGraph(graph_directory, engine) in place of Q16NeuralPL and HandGraph.
Works with the board's older driver, which has no defer_output argument.
Capture mode uses the parent float graph. Learned MACs always execute in PL.
"""
from collections import Counter
import ctypes
import os
from pathlib import Path
import time

import numpy as np

from hand_graph import HandGraph
from hand_neural_pl import (Q16NeuralPL, convolution_shape,
                            quantize_activation, quantize_weights)


class NativeTransport:
    def __init__(self, library=None):
        path = Path(library) if library else Path(__file__).with_name('libfingertip_transport.so')
        self.library = ctypes.CDLL(str(path))
        self.function = self.library.fingertip_q16_clip_pack
        self.function.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*2 + [ctypes.POINTER(ctypes.c_float)]
        self.function.restype = ctypes.c_int
        self.workspace_function = self.library.fingertip_q16_clip_pack_workspace
        self.workspace_function.argtypes = self.function.argtypes + [ctypes.c_void_p]
        self.workspace_function.restype = ctypes.c_int
        self.wait_function = getattr(self.library, 'fingertip_wait_done', None)
        if self.wait_function is not None:
            self.wait_function.argtypes = [ctypes.c_void_p, ctypes.c_double]
            self.wait_function.restype = ctypes.c_int

    def clip_pack(self, source, destination, factors, bias, channels, pixels, scratch=None):
        padded_size = pixels*((channels+7)//8*8)
        arrays = ((source, np.int64, channels*pixels),
                  (destination, np.int16, padded_size),
                  (factors, np.float32, channels), (bias, np.float32, channels))
        if not (0 < channels <= 1024 and 0 < pixels and padded_size <= 1048576):
            raise ValueError('Fused transport exceeds Q16 bounds')
        for array, dtype, size in arrays:
            if (array.dtype != dtype or not array.dtype.isnative or
                    not array.flags.c_contiguous or not array.flags.aligned or array.size < size):
                raise ValueError('Invalid fused transport array')
        if not destination.flags.writeable or any(np.shares_memory(destination, array)
                                                  for array in (source, factors, bias)):
            raise ValueError('Fused transport destination must be writable and disjoint')
        scale = ctypes.c_float()
        arguments = (source.ctypes.data, destination.ctypes.data, factors.ctypes.data,
                     bias.ctypes.data, channels, pixels, ctypes.byref(scale))
        if scratch is None:
            status = self.function(*arguments)
        else:
            if (scratch.dtype != np.float32 or not scratch.flags.c_contiguous or
                    not scratch.flags.aligned or not scratch.flags.writeable or
                    scratch.size < channels*pixels or any(np.shares_memory(scratch, a)
                    for a in (source, destination, factors, bias))):
                raise ValueError('Invalid fused transport scratch')
            status = self.workspace_function(*arguments, scratch.ctypes.data)
        if status != 0 or not np.isfinite(scale.value) or scale.value <= 0:
            raise ValueError('Invalid fused Q16 conversion or floating point mode')
        return scale.value


class DeferredOutput:
    """Borrowed NCHW int64 sums; the next PL start invalidates this handle."""
    ndim = 4

    def __init__(self, engine, shape, scales, bias, input_scale):
        self.engine, self.shape = engine, tuple(shape)
        self.scales = np.ascontiguousarray(scales, dtype=np.float32).copy()
        self.bias = np.ascontiguousarray(bias, dtype=np.float32).copy()
        self.input_scale = np.float32(input_scale)
        self.generation = engine.output_generation
        self.clip = None

    def check(self):
        if (self.engine.closed or self.engine.fault or
                self.generation != self.engine.output_generation):
            raise RuntimeError('PL accumulator tensor was overwritten or engine is closed/faulted')

    def clipped(self, low, high):
        self.check()
        if low != 0 or high != 6:
            raise ValueError('Fused transport requires Clip(0,6)')
        self.clip = (0, 6)
        return self


class FingertipQ16NeuralPL(Q16NeuralPL):
    backend_name = 'pl_q16_conv_ps_fused_transport'

    def __init__(self, processor, profile=False, library=None):
        # Fail before allocating DMA buffers when the opt-in library is absent.
        self.transport = NativeTransport(library)
        super().__init__(processor, profile)
        self.output_generation = 0
        self.transport_calls = 0
        self.transport_ms = 0.

    def run(self, pointers, scalars=()):
        # Advance before start, including failed starts, so stale sums cannot
        # survive timeout/quarantine or a later ordinary parent convolve call.
        self.output_generation += 1
        control = getattr(getattr(self, 'mmio', None), 'array', None)
        wait = getattr(self.transport, 'wait_function', None)
        if (wait is None or os.environ.get('FRETLENS_FINGERTIP_NATIVE_WAIT', '1') != '1'
                or not isinstance(control, np.ndarray) or control.dtype != np.uint32
                or not control.flags.c_contiguous or not control.size):
            return super().run(pointers, scalars)
        try:
            if self.fault or self.closed:
                raise RuntimeError('HLS kernel faulted')
            if not self.mmio.read(0) & 4:
                raise RuntimeError('HLS kernel is not idle')
        except Exception:
            self._register_values.clear()
            raise
        updates = []
        for offset, buffer in pointers:
            updates.extend(((offset, int(buffer.physical_address)), (offset+4, 0)))
        updates.extend((offset, int(value)) for offset, value in scalars)
        try:
            for offset, value in updates:
                if self._register_values.get(offset) != value:
                    self.mmio.write(offset, value)
            started = time.perf_counter()
            self.mmio.write(0, 1)
            status = wait(control.ctypes.data, self.timeout)
            if status == 1:
                raise TimeoutError('HLS timeout; live DMA buffers quarantined')
            if status != 0:
                raise RuntimeError('Native HLS wait failed; live DMA buffers quarantined')
        except Exception:
            from vision_hls import _QUARANTINE
            self._register_values.clear()
            self.fault = True
            _QUARANTINE.extend(self.buffers)
            self.buffers = []
            raise
        self._register_values.update(updates)
        return (time.perf_counter()-started)*1000

    def _read_sums(self, size):
        written = self.buffers[2][:size]
        written.invalidate()
        values = np.asarray(written)
        return values if getattr(written, 'cacheable', False) else values.copy()

    def _materialize(self, tensor):
        tensor.check()
        _, channels, height, width = tensor.shape
        sums = self._read_sums(channels*height*width)
        result = self._convert(sums, tensor.scales, tensor.bias,
                               tensor.input_scale, tensor.shape)
        if tensor.clip is not None:
            np.clip(result, *tensor.clip, out=result)
        return result

    def materialize(self, tensor):
        if not isinstance(tensor, DeferredOutput) or tensor.engine is not self:
            raise ValueError('PL accumulator belongs to another engine')
        with self.lock:
            return self._materialize(tensor)

    def _convert(self, sums, scales, bias, scale, shape):
        if self.native is None:
            result = sums.astype(np.float32).reshape(shape)
            result *= (scales*np.float32(scale))[None, :, None, None]
            result += bias[None, :, None, None]
        else:
            result = np.empty(shape, np.float32)
            self.native.hand_q16_unpack(sums.ctypes.data, result.ctypes.data,
                scales.ctypes.data, bias.ctypes.data, float(scale), shape[1], shape[2]*shape[3])
        return result

    def convolve(self, x, weights, bias, stride, pads, groups, defer_output=False):
        deferred_input = isinstance(x, DeferredOutput)
        if not deferred_input and not defer_output:
            # The active board parent needs no API changes.
            return super().convolve(x, weights, bias, stride, pads, groups)
        started = time.perf_counter()
        if deferred_input:
            if x.engine is not self:
                raise ValueError('PL accumulator belongs to another engine')
            x.check()
            if x.clip != (0, 6):
                raise ValueError('Deferred input requires Clip(0,6)')
        else:
            x = np.asarray(x, np.float32)
        weights = np.asarray(weights, np.float32)
        bias = np.ascontiguousarray(bias, dtype=np.float32).reshape(-1)
        channels, height, width, outputs, kernel, oh, ow = convolution_shape(
            x, weights, bias, stride, pads, groups)
        if not np.isfinite(bias).all():
            raise ValueError('Nonfinite neural bias')
        with self.lock:
            if self.stopping or self.closed or self.fault:
                raise RuntimeError('PL neural inference is stopping or faulted')
            if deferred_input:
                x.check()
            padded = (channels+7)//8*8
            size = height*width*padded
            if size > self.buffers[0].size or outputs*oh*ow > self.buffers[2].size:
                raise ValueError('PL neural tensor exceeds allocated DDR')
            key = (weights.ctypes.data, weights.shape, weights.strides, groups)
            cached = self.weight_cache.get(key)
            if cached is None:
                packed_weights, scales = quantize_weights(weights, groups == channels == outputs)
                end = self.weight_used+packed_weights.size
                if end > self.buffers[1].size:
                    raise ValueError('PL resident weights exceed allocated DDR')
                resident = self.buffers[1][self.weight_used:end]
                self.copy(resident, packed_weights)
                cached = (resident, scales, weights)
                self.weight_cache[key] = cached
                self.weight_used = end
            resident, scales, _ = cached
            pack_started = time.perf_counter()
            source_invalidated = pack_done = pack_started
            destination = self.buffers[0][:size]
            direct = getattr(destination, 'cacheable', False)
            if deferred_input:
                sums = self._read_sums(channels*height*width)
                source_invalidated = time.perf_counter()
                packed = np.asarray(destination) if direct else np.empty(size, np.int16)
                factors = x.scales*x.input_scale
                count = channels*height*width
                if not hasattr(self, '_transport_scratch') or self._transport_scratch.size < count:
                    self._transport_scratch = np.empty(count, np.float32)
                scale = self.transport.clip_pack(sums, packed, factors, x.bias, channels,
                                                height*width, self._transport_scratch)
                pack_done = time.perf_counter()
                self.transport_calls += 1
                self.transport_ms += (pack_done-source_invalidated)*1000
            elif self.native is not None:
                packed = np.asarray(destination) if direct else np.empty(size, np.int16)
                activation = np.ascontiguousarray(x)
                native_scale = ctypes.c_float()
                if self.native.hand_q16_pack(activation.ctypes.data, packed.ctypes.data,
                        channels, height*width, ctypes.byref(native_scale)) != 0:
                    raise ValueError('Nonfinite neural activation')
                scale = native_scale.value
                pack_done = time.perf_counter()
            else:
                packed, scale = quantize_activation(x)
                direct = False
                pack_done = time.perf_counter()
            if direct:
                destination.flush()
            else:
                self.copy(destination, packed)
            copied = time.perf_counter()
            scalars = dict(height=height, width=width, channels=channels, outputs=outputs,
                kernel=kernel, stride=stride, pad_top=pads[0], pad_left=pads[1],
                out_h=oh, out_w=ow, groups=groups)
            kernel_ms = self.run([(self.offset('input'), self.buffers[0]),
                (self.offset('weights'), resident), (self.offset('output'), self.buffers[2])],
                [(self.offset(name), value) for name, value in scalars.items()])
            kernel_done = time.perf_counter()
            shape = (1, outputs, oh, ow)
            if defer_output:
                result = DeferredOutput(self, shape, scales, bias, scale)
                read_done = kernel_done
            else:
                sums = self._read_sums(outputs*oh*ow)
                read_done = time.perf_counter()
                result = self._convert(sums, scales, bias, scale, shape)
            finished = time.perf_counter()
            self.calls += 1
            self.kernel_ms += kernel_ms
            elapsed = (finished-started)*1000
            self.total_ms += elapsed
            if self.profile:
                self.layer_profile.append(dict(call=self.calls,
                    kind='depthwise' if groups == channels == outputs else
                         ('pointwise' if kernel == 1 else 'regular'),
                    input_shape=list(x.shape), weight_shape=list(weights.shape), output_shape=list(shape),
                    stride=stride, groups=groups, kernel_ms=kernel_ms, total_ms=elapsed,
                    deferred_output=bool(defer_output), fused_transport=deferred_input,
                    activation_pack_ms=0. if deferred_input else (pack_done-pack_started)*1000,
                    fused_transport_ms=(pack_done-source_invalidated)*1000 if deferred_input else 0.,
                    input_invalidate_copy_ms=(source_invalidated-pack_started)*1000,
                    input_flush_copy_ms=(copied-pack_done)*1000,
                    invalidate_ms=(read_done-kernel_done)*1000,
                    output_convert_ms=0. if defer_output else (finished-read_done)*1000,
                    host_ms=max(0., elapsed-kernel_ms)))
            return result


class FingertipGraph(HandGraph):
    """Old-board-compatible graph scheduler with conservative deferred edges."""

    def __init__(self, directory, engine, model_path=None):
        super().__init__(directory, engine, model_path)
        self.producers = {node['output']: index for index, node in enumerate(self.nodes)}
        self.transport_pairs = {}
        for index in range(len(self.nodes)-2):
            first, clip, following = self.nodes[index:index+3]
            if (first['op'] == following['op'] == 'PLConvolution' and clip['op'] == 'Clip'
                    and clip['inputs'] == [first['output']]
                    and following['inputs'][0] == clip['output']
                    and clip['attrs'].get('min') == 0 and clip['attrs'].get('max') == 6
                    and self.uses[first['output']] == self.uses[clip['output']] == 1):
                self.transport_pairs[index] = (first['output'], clip['output'])
        self.transport_fused_calls = 0

    def forward(self, names=None):
        self.transport_fused_calls = 0
        # The worker is asynchronous with respect to VisionPL.process, so its
        # inference must validate live ownership before the first PL access.
        processor = getattr(self.engine, 'processor', None)
        if processor is not None:
            with self.engine.lock:
                try:
                    processor._ensure_active_overlay()
                except Exception:
                    self.engine._register_values.clear()
                    raise
        if self.capture or not isinstance(self.engine, FingertipQ16NeuralPL):
            return super().forward(names)
        if self.input is None:
            raise ValueError('Hand graph input is missing')
        requested = list(self.outputs if names is None else names)
        pending, required = list(requested), set()
        while pending:
            name = pending.pop()
            index = self.producers.get(name)
            if index is not None and index not in required:
                required.add(index)
                pending.extend(self.nodes[index]['inputs'])
        uses = Counter(name for index in required for name in self.nodes[index]['inputs'])
        uses.update(requested)
        deferred = {index for index, edges in self.transport_pairs.items()
                    if all(i in required for i in (index, index+1, index+2))
                    and all(name not in requested and name not in self.outputs for name in edges)}
        values = dict(self.weights)
        values[self.input_name] = self.input
        self.timings, self.captured = [], {}
        for index, node in enumerate(self.nodes):
            if index not in required:
                self.timings.append(0.)
                continue
            started = time.perf_counter()
            args = [values[name] for name in node['inputs']]
            x, op, attr = args[0], node['op'], node['attrs']
            if op == 'PLConvolution':
                strides = attr.get('strides', [1, 1])
                if strides[0] != strides[1] or attr.get('dilations', [1, 1]) != [1, 1]:
                    raise ValueError('Unsupported graph convolution')
                bias = args[2] if len(args) > 2 else np.zeros(args[1].shape[0], np.float32)
                fused_input = isinstance(x, DeferredOutput)
                out = self.engine.convolve(x, args[1], bias, strides[0],
                    tuple(attr.get('pads', [0]*4)), attr.get('group', 1), defer_output=index in deferred)
                self.transport_fused_calls += int(fused_input)
            elif op == 'Clip':
                if isinstance(x, DeferredOutput):
                    out = x.clipped(attr['min'], attr['max'])
                else:
                    reusable = uses[node['inputs'][0]] == 1 and node['inputs'][0] not in self.weights
                    out = x if reusable and x.flags.writeable else np.empty_like(x)
                    np.clip(x, attr.get('min', -np.inf), attr.get('max', np.inf), out=out)
            else:
                out = self._ordinary(node, args)
            values[node['output']] = out
            self.timings.append((time.perf_counter()-started)*1000)
            for name in node['inputs']:
                uses[name] -= 1
                if uses[name] == 0 and name not in self.weights:
                    values.pop(name, None)
        return [values[name] for name in requested]

    def _ordinary(self, node, args):
        x, op, attr = args[0], node['op'], node['attrs']
        if op == 'PLDense':
            return self.engine.convolve(x.reshape(1, -1, 1, 1),
                args[1].reshape(args[1].shape[0], -1, 1, 1), args[2], 1, (0, 0, 0, 0), 1).reshape(1, -1)
        if op == 'Transpose':
            return np.transpose(x, attr['perm'])
        if op == 'Add':
            return np.add(x, args[1])
        if op == 'Pad':
            if attr.get('mode', 'constant') != 'constant':
                raise ValueError('Unsupported graph padding')
            pads = attr['pads']
            return np.pad(x, tuple(zip(pads[:x.ndim], pads[x.ndim:])), constant_values=attr.get('value', 0))
        if op == 'MaxPool':
            if attr.get('kernel_shape') != [2, 2] or attr.get('strides') != [2, 2] or any(attr.get('pads', [0]*4)):
                raise ValueError('Unsupported graph pooling')
            h, w = x.shape[-2]//2, x.shape[-1]//2
            return np.maximum(np.maximum(x[..., :2*h:2, :2*w:2], x[..., 1:2*h:2, :2*w:2]),
                              np.maximum(x[..., :2*h:2, 1:2*w:2], x[..., 1:2*h:2, 1:2*w:2]))
        if op == 'GlobalAveragePool':
            return np.mean(x, axis=(2, 3), keepdims=True, dtype=np.float32)
        if op == 'Flatten':
            axis = attr.get('axis', 1)
            return x.reshape(int(np.prod(x.shape[:axis])), -1)
        if op == 'Sigmoid':
            return np.reciprocal(np.float32(1)+np.exp(-x))
        raise ValueError('Unsupported graph operation: '+op)
