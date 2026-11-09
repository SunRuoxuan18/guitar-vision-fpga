"""OpenCV custom neural layers backed by the active hand_conv HLS core.

PS schedules layers and executes activations/crops; convolution and dense-layer
multiply-accumulate operations run on PL. There is no CPU convolution fallback.
"""
import time
import ctypes
from pathlib import Path

import cv2
import numpy as np

from vision_hls import HLSKernel

_ENGINE = None
_REGISTERED = False


def convolution_shape(x, weights, bias, stride, pads, groups):
    if x.ndim != 4 or weights.ndim != 4 or x.shape[0] != 1:
        raise ValueError('PL neural layer requires fixed batch-one NCHW')
    _, channels, height, width = x.shape
    outputs, per_group, kh, kw = weights.shape
    top, left, bottom, right = pads
    if (kh != kw or kh not in (1,3,5) or stride not in (1,2)
            or groups not in (1, channels) or channels != per_group*groups
            or (groups != 1 and outputs != channels)
            or max(channels,outputs) > 1024 or min(channels,outputs,per_group,height,width) < 1
            or max(height,width) > 256 or min(pads) < 0 or max(pads) > 4
            or bias.size != outputs):
        raise ValueError('Unsupported hand_conv dimensions')
    oh, ow = (height+top+bottom-kh)//stride+1, (width+left+right-kw)//stride+1
    if min(oh,ow) < 1:
        raise ValueError('Empty convolution output')
    return channels,height,width,outputs,kh,oh,ow


def quantize_activation(x):
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite neural activation')
    scale = max(float(np.max(np.abs(x)))/32767., np.finfo(np.float32).tiny)
    channels = x.shape[1]
    packed = np.zeros((x.shape[2],x.shape[3],(channels+7)//8*8),np.int16)
    packed[:,:,:channels] = np.rint(np.clip(x[0].transpose(1,2,0)/scale,-32767,32767)).astype(np.int16)
    return packed,scale


def quantize_weights(weights, depthwise=False):
    if not np.isfinite(weights).all():
        raise ValueError('Nonfinite neural weights')
    scales = np.maximum(np.max(np.abs(weights),axis=(1,2,3))/32767.,np.finfo(np.float32).tiny)
    q = np.rint(np.clip(weights/scales[:,None,None,None],-32767,32767)).astype(np.int16)
    if depthwise:
        packed = np.zeros((weights.shape[2],weights.shape[3],(weights.shape[0]+7)//8*8),np.int16)
        packed[:,:,:weights.shape[0]] = q[:,0].transpose(1,2,0)
    else:
        packed = np.zeros((weights.shape[0],weights.shape[2],weights.shape[3],(weights.shape[1]+7)//8*8),np.int16)
        packed[:,:,:,:weights.shape[1]] = q.transpose(0,2,3,1)
    return packed,scales


class NeuralPL(HLSKernel):
    backend_name = 'pl_fp32_conv_ps_graph'

    def __init__(self, processor, profile=False):
        # Largest tensor in these fixed models is below 2**20 floats. Bounds
        # are checked before copying, so malformed shapes cannot overwrite DDR.
        super().__init__(processor, 'hand_conv',
            [((1048576,), np.float32), ((1048576,), np.float32),
             ((1024,), np.float32), ((1048576,), np.float32)], timeout=30.)
        description = processor._overlay.ip_dict['hand_conv']
        self.registers = description['registers']
        self.calls = 0
        self.kernel_ms = 0.
        self.total_ms = 0.
        self.stopping = False
        self.profile = profile
        self.layer_profile = []

    def request_stop(self):
        # Finish the current AXI transaction, then reject the next layer.
        self.stopping = True
        if getattr(self, 'stream', None) is not None:
            self.stream.request_stop()

    def offset(self, name):
        key = next((key for key in (name,name+'_r',name+'_1',name+'_r_1')
                    if key in self.registers),None)
        if key is None:
            raise ValueError('Missing neural HWH register '+name)
        value = self.registers[key]['address_offset']
        return int(value,0) if isinstance(value,str) else int(value)

    def convolve(self, x, weights, bias, stride, pads, groups):
        started = time.perf_counter()
        x = np.ascontiguousarray(x, dtype=np.float32)
        weights = np.ascontiguousarray(weights, dtype=np.float32)
        bias = np.ascontiguousarray(bias, dtype=np.float32).reshape(-1)
        channels,height,width,outputs,kh,oh,ow = convolution_shape(x,weights,bias,stride,pads,groups)
        top,left = pads[:2]
        sizes = (x.size, weights.size, bias.size, outputs*oh*ow)
        if any(n > b.size for n,b in zip(sizes,self.buffers)):
            raise ValueError('PL neural tensor exceeds allocated DDR')
        with self.lock:
            if self.stopping:
                raise RuntimeError('PL neural inference is stopping')
            for buffer,array in zip(self.buffers,(x,weights,bias)):
                self.copy(buffer[:array.size],array)
            names = ('input','weights','bias','output')
            scalars = dict(height=height,width=width,channels=channels,outputs=outputs,
                           kernel=kh,stride=stride,pad_top=top,pad_left=left,
                           out_h=oh,out_w=ow,groups=groups)
            kernel_ms = self.run([(self.offset(n),b) for n,b in zip(names,self.buffers)],
                                 [(self.offset(n),v) for n,v in scalars.items()])
            written = self.buffers[3][:sizes[3]]
            written.invalidate()
            result = np.asarray(written).copy().reshape(1,outputs,oh,ow)
            self.calls += 1
            self.kernel_ms += kernel_ms
            elapsed_ms = (time.perf_counter()-started)*1000
            self.total_ms += elapsed_ms
            if self.profile:
                self.layer_profile.append(dict(call=self.calls,
                    kind='depthwise' if groups == channels == outputs else
                         ('pointwise' if kh == 1 else 'regular'),
                    input_shape=list(x.shape),weight_shape=list(weights.shape),
                    output_shape=list(result.shape),stride=stride,groups=groups,
                    kernel_ms=kernel_ms,total_ms=elapsed_ms,
                    host_ms=max(0.,elapsed_ms-kernel_ms)))
            return result


class Q16NeuralPL(NeuralPL):
    backend_name = 'pl_q16_conv_ps_graph'

    def __init__(self, processor, profile=False):
        HLSKernel.__init__(self,processor,'hand_conv_q16',
            [((1048576,),np.int16),((8*1048576,),np.int16),((1048576,),np.int64)],timeout=10.)
        self.registers = processor._overlay.ip_dict['hand_conv_q16']['registers']
        self.calls = 0
        self.kernel_ms = self.total_ms = 0.
        self.stopping = False
        self.profile = profile
        self.layer_profile = []
        self.weight_cache = {}
        self.weight_used = 0
        self._offset_cache = {}
        self._register_values = {}
        self.output_generation = 0
        self.requant = None
        self.requant_calls = 0
        self.requant_ms = 0.
        if 'hand_requant' in processor._overlay.ip_dict:
            from hand_requant import HandRequant
            self.requant = HandRequant(processor)
        from hand_stream import create_hand_stream
        self.stream = None
        try:
            self.stream = create_hand_stream(processor, profile=profile)
        except Exception:
            self.close()
            raise
        library = Path(__file__).with_name('libhand_q16.so')
        self.native = ctypes.CDLL(str(library)) if library.is_file() else None
        if self.native is not None:
            self.native.hand_q16_pack.argtypes = [ctypes.c_void_p,ctypes.c_void_p,
                ctypes.c_int,ctypes.c_int,ctypes.POINTER(ctypes.c_float)]
            self.native.hand_q16_pack.restype = ctypes.c_int
            self.native.hand_q16_unpack.argtypes = [ctypes.c_void_p]*4+[
                ctypes.c_float,ctypes.c_int,ctypes.c_int]
            self.native.hand_q16_unpack.restype = None

    def offset(self, name):
        if name not in self._offset_cache:
            self._offset_cache[name] = super().offset(name)
        return self._offset_cache[name]

    def run(self, pointers, scalars=()):
        # This driver owns the core under convolve's lock. HLS argument
        # registers retain their values between ap_start transactions.
        updates = []
        for offset, buffer in pointers:
            updates.extend(((offset, int(buffer.physical_address)), (offset+4, 0)))
        updates.extend((offset, int(value)) for offset, value in scalars)
        changed = [(offset, value) for offset, value in updates
                   if self._register_values.get(offset) != value]
        try:
            elapsed = super().run([], changed)
        except Exception:
            self._register_values.clear()
            raise
        self._register_values.update(updates)
        return elapsed

    def convolve(self, x, weights, bias, stride, pads, groups, defer_output=False):
        started = time.perf_counter()
        from hand_requant import AccumulatorTensor
        deferred_input = isinstance(x, AccumulatorTensor)
        if deferred_input:
            if x.engine is not self or self.requant is None:
                raise ValueError('PL accumulator belongs to another engine')
            x.check()
        else:
            x = np.asarray(x,np.float32)
        weights = np.asarray(weights,np.float32)
        bias = np.asarray(bias,np.float32).reshape(-1)
        channels,height,width,outputs,kernel,oh,ow = convolution_shape(x,weights,bias,stride,pads,groups)
        if defer_output and self.requant is None:
            raise RuntimeError('Hardware requantization core is unavailable')
        if not np.isfinite(bias).all():
            raise ValueError('Nonfinite neural bias')
        with self.lock:
            if self.stopping or self.closed or self.fault:
                raise RuntimeError('PL neural inference is stopping or faulted')
            if deferred_input:
                x.check()
            packed_shape = (height,width,(channels+7)//8*8)
            packed_size = height*width*packed_shape[2]
            if packed_size > self.buffers[0].size or outputs*oh*ow > self.buffers[2].size:
                raise ValueError('PL neural tensor exceeds allocated DDR')
            direct_packed = self.native is not None and getattr(self.buffers[0],'cacheable',False)
            quant_started = time.perf_counter()
            requant_ms = 0.
            if deferred_input:
                try:
                    scale, requant_ms = self.requant.convert(x, self.buffers[2], self.buffers[0])
                except Exception:
                    if self.requant.fault:
                        # The bridge borrows our DDR too. A timed-out AXI master
                        # may still access it even after its own driver closes.
                        from vision_hls import _QUARANTINE
                        self.fault = True
                        _QUARANTINE.extend(self.buffers)
                        self.buffers = []
                    raise
                self.requant_calls = getattr(self, 'requant_calls', 0)+1
                self.requant_ms = getattr(self, 'requant_ms', 0.)+requant_ms
            elif self.native is None:
                packed,scale = quantize_activation(x)
            else:
                x = np.ascontiguousarray(x)
                # Cached CMA is ordinary PS memory until the explicit flush.
                # Pack once into the actual AXI operand instead of copying it.
                packed = (np.asarray(self.buffers[0][:packed_size]).reshape(packed_shape)
                          if direct_packed else np.empty(packed_shape,np.int16))
                native_scale = ctypes.c_float()
                if self.native.hand_q16_pack(x.ctypes.data,packed.ctypes.data,
                        channels,height*width,ctypes.byref(native_scale)) != 0:
                    raise ValueError('Nonfinite neural activation')
                scale = native_scale.value
            quant_done = time.perf_counter()
            key = (weights.ctypes.data,weights.shape,weights.strides,groups)
            cached = self.weight_cache.get(key)
            if cached is None:
                qw,scales = quantize_weights(weights,groups == channels == outputs)
                end = self.weight_used+qw.size
                if end > self.buffers[1].size:
                    raise ValueError('PL resident weights exceed allocated DDR')
                resident = self.buffers[1][self.weight_used:end]
                self.copy(resident,qw)
                # Retain the immutable model blob so its address cannot be reused.
                cached = (resident,scales,weights)
                self.weight_cache[key] = cached
                self.weight_used = end
            resident,scales,_ = cached
            if deferred_input:
                pass
            elif direct_packed:
                self.buffers[0][:packed_size].flush()
            else:
                self.copy(self.buffers[0][:packed.size],packed)
            copied = time.perf_counter()
            scalars = dict(height=height,width=width,channels=channels,outputs=outputs,kernel=kernel,
                stride=stride,pad_top=pads[0],pad_left=pads[1],out_h=oh,out_w=ow,groups=groups)
            kernel_ms = self.run([(self.offset('input'),self.buffers[0]),
                (self.offset('weights'),resident),(self.offset('output'),self.buffers[2])],
                [(self.offset(n),v) for n,v in scalars.items()])
            kernel_done = time.perf_counter()
            self.output_generation += 1
            if defer_output:
                if self.requant is None:
                    raise RuntimeError('Hardware requantization core is unavailable')
                result = AccumulatorTensor(self, (1,outputs,oh,ow),
                    scales*np.float32(scale), bias.copy(), self.output_generation)
                invalidated = output_copied = time.perf_counter()
            else:
                result, invalidated, output_copied = self._read_output(outputs, oh, ow, scales, bias, scale)
            self.calls += 1
            self.kernel_ms += kernel_ms
            elapsed = (time.perf_counter()-started)*1000
            self.total_ms += elapsed
            if self.profile:
                self.layer_profile.append(dict(call=self.calls,
                    kind='depthwise' if groups == channels == outputs else ('pointwise' if kernel == 1 else 'regular'),
                    input_shape=list(x.shape),weight_shape=list(weights.shape),output_shape=list(result.shape),
                    stride=stride,groups=groups,kernel_ms=kernel_ms,total_ms=elapsed,
                    quantize_copy_ms=(copied-quant_started)*1000,
                    activation_pack_ms=0. if deferred_input else (quant_done-quant_started)*1000,
                    requant_pl_ms=requant_ms, deferred_output=bool(defer_output),
                    invalidate_ms=(invalidated-kernel_done)*1000,
                    output_copy_ms=(output_copied-invalidated)*1000,
                    output_convert_ms=(time.perf_counter()-output_copied)*1000,
                    host_ms=max(0.,elapsed-kernel_ms-requant_ms)))
            return result

    def _read_output(self, outputs, oh, ow, scales, bias, scale):
        written = self.buffers[2][:outputs*oh*ow]
        written.invalidate()
        invalidated = time.perf_counter()
        # Uncached CMA must be burst-copied before PS arithmetic.
        cached_output = np.asarray(written)
        if not getattr(written, 'cacheable', False):
            cached_output = cached_output.copy()
        output_copied = time.perf_counter()
        if self.native is None:
            result = cached_output.astype(np.float32).reshape(1,outputs,oh,ow)
            result *= (scales*np.float32(scale))[None,:,None,None]
            result += bias[None,:,None,None]
        else:
            bias = np.ascontiguousarray(bias)
            result = np.empty((1,outputs,oh,ow),np.float32)
            self.native.hand_q16_unpack(cached_output.ctypes.data,result.ctypes.data,
                scales.ctypes.data,bias.ctypes.data,scale,outputs,oh*ow)
        return result, invalidated, output_copied

    def close(self):
        super().close()
        if getattr(self, 'requant', None) is not None:
            self.requant.close()
        if getattr(self, 'stream', None) is not None:
            self.stream.close()


def create_neural_engine(processor, profile=False):
    names = processor._overlay.ip_dict
    if 'hand_conv_q16' in names:
        return Q16NeuralPL(processor,profile)
    if 'hand_conv' in names:
        return NeuralPL(processor,profile)
    return None


class PLConvolution:
    def __init__(self, params, blobs):
        # OpenCV 4.2 pycvLayer fails to DECREF forward's return value.
        # Reuse its container so leaked references cannot retain old tensors.
        self._outputs = [None]
        self.weights = np.ascontiguousarray(blobs[0],dtype=np.float32)
        self.bias = (np.ascontiguousarray(blobs[1],dtype=np.float32).reshape(-1)
                     if len(blobs) > 1 else np.zeros(self.weights.shape[0],np.float32))
        self.strides = tuple(params.get('stride',(1,1)))
        self.pads = tuple(params.get('pad',(0,0,0,0)))
        self.groups = int(params.get('group',1))
        dilation = tuple(params.get('dilation',(1,1)))
        if len(self.pads) == 2:
            self.pads = self.pads*2
        if (self.strides[0] != self.strides[1] or dilation != (1,1)
                or len(self.pads) != 4):
            raise ValueError('Unsupported custom convolution attributes')

    def getMemoryShapes(self, inputs):
        n,c,h,w = inputs[0]
        top,left,bottom,right = self.pads
        _,_,kh,kw = self.weights.shape
        return [[n,self.weights.shape[0],(h+top+bottom-kh)//self.strides[0]+1,
                 (w+left+right-kw)//self.strides[1]+1]]

    def forward(self, inputs):
        if _ENGINE is None:
            raise RuntimeError('PL neural engine is not bound; CPU fallback forbidden')
        try:
            self._outputs[0] = _ENGINE.convolve(inputs[0],self.weights,self.bias,self.strides[0],self.pads,self.groups)
            return self._outputs
        except Exception:
            import traceback
            traceback.print_exc()
            raise


class PLDense:
    def __init__(self, params, blobs):
        self._outputs = [None]
        self.weights = np.ascontiguousarray(blobs[0],dtype=np.float32)
        self.bias = np.ascontiguousarray(blobs[1],dtype=np.float32).reshape(-1)

    def getMemoryShapes(self, inputs):
        return [[inputs[0][0],self.weights.shape[0]]]

    def forward(self, inputs):
        if _ENGINE is None:
            raise RuntimeError('PL neural engine is not bound; CPU fallback forbidden')
        x = inputs[0].reshape(1,-1,1,1)
        weights = self.weights.reshape(self.weights.shape[0],-1,1,1)
        self._outputs[0] = _ENGINE.convolve(x,weights,self.bias,1,(0,0,0,0),1).reshape(1,-1)
        return self._outputs


def register(engine):
    global _ENGINE, _REGISTERED
    if not _REGISTERED:
        cv2.dnn_registerLayer('PLConvolution', PLConvolution)
        cv2.dnn_registerLayer('PLDense', PLDense)
        _REGISTERED = True
    _ENGINE = engine
