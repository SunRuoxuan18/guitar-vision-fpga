"""Optional driver for the fused ``hand_dw_pw_stream`` HLS core.

The stream core owns the depthwise ReLU6/requantize boundary.  This module
only packs operands, starts the core, and converts its int64 pointwise sums;
there is deliberately no CPU convolution fallback.
"""
import os
import time

import numpy as np

from hand_neural_pl import quantize_activation, quantize_weights
from vision_hls import HLSKernel


CORE_NAME = "hand_dw_pw_stream"
CORE_ADDRESS = 0x43C50000
MAX_CHANNELS = 144
MAX_OUTPUTS = 24
MAX_HEIGHT = MAX_WIDTH = 32
MAX_LINE_VALUES = 4096
RELU6_SCALE = np.float32(6.0 / 32767.0)


def _address(description):
    value = description.get("phys_addr", -1)
    return int(value, 0) if isinstance(value, str) else int(value)


def stream_core_available(processor):
    """Return whether the optional stream core is present and correctly mapped."""
    description = processor._overlay.ip_dict.get(CORE_NAME)
    if description is None:
        return False
    if (_address(description) != CORE_ADDRESS
            or ":{}:".format(CORE_NAME) not in description.get("type", "")):
        raise ValueError("Combined overlay has an invalid {} identity".format(CORE_NAME))
    return True


def _shape(x, dw_weights, dw_bias, pw_weights, pw_bias, stride, pads):
    x = np.ascontiguousarray(x, dtype=np.float32)
    dw_weights = np.ascontiguousarray(dw_weights, dtype=np.float32)
    pw_weights = np.ascontiguousarray(pw_weights, dtype=np.float32)
    dw_bias = np.ascontiguousarray(dw_bias, dtype=np.float32).reshape(-1)
    pw_bias = np.ascontiguousarray(pw_bias, dtype=np.float32).reshape(-1)
    if (x.ndim != 4 or x.shape[0] != 1 or dw_weights.ndim != 4
            or pw_weights.ndim != 4 or len(pads) != 4):
        raise ValueError("hand_dw_pw_stream requires batch-one NCHW tensors")
    _, channels, height, width = x.shape
    outputs, pw_channels, pkh, pkw = pw_weights.shape
    dw_channels, dw_per_channel, kernel, kernel_width = dw_weights.shape
    top, left, bottom, right = map(int, pads)
    if (channels != dw_channels or dw_per_channel != 1 or kernel != kernel_width
            or kernel not in (3, 5) or pw_channels != channels or (pkh, pkw) != (1, 1)
            or dw_bias.size != channels or pw_bias.size != outputs or stride not in (1, 2)
            or min(channels, outputs, height, width) < 1 or channels > MAX_CHANNELS
            or outputs > MAX_OUTPUTS or height > MAX_HEIGHT or width > MAX_WIDTH
            or width * channels > MAX_LINE_VALUES or min(top, left, bottom, right) < 0
            or width * ((channels + 7) // 8) > 512
            or max(top, left, bottom, right) > 2):
        raise ValueError("Unsupported hand_dw_pw_stream dimensions")
    out_h = (height + top + bottom - kernel) // stride + 1
    out_w = (width + left + right - kernel) // stride + 1
    if min(out_h, out_w) < 1 or out_h > MAX_HEIGHT or out_w > MAX_WIDTH:
        raise ValueError("Invalid hand_dw_pw_stream output geometry")
    if not all(np.isfinite(a).all() for a in (x, dw_weights, dw_bias, pw_weights, pw_bias)):
        raise ValueError("Nonfinite hand_dw_pw_stream operand")
    return (x, dw_weights, dw_bias, pw_weights, pw_bias, channels, height, width,
            outputs, kernel, top, left, out_h, out_w)


class HandStream(HLSKernel):
    """Fused depthwise/pointwise HLS driver with resident packed weights."""
    backend_name = "pl_hand_dw_pw_stream"

    def __init__(self, processor, profile=False, timeout=10.0):
        if not stream_core_available(processor):
            raise RuntimeError("Combined overlay does not contain " + CORE_NAME)
        super().__init__(processor, CORE_NAME, [
            ((MAX_HEIGHT * MAX_WIDTH * MAX_CHANNELS,), np.int16),
            ((25 * MAX_CHANNELS,), np.int16),
            ((MAX_OUTPUTS * MAX_CHANNELS,), np.int16),
            ((2 * MAX_CHANNELS,), np.float32),
            ((MAX_OUTPUTS * MAX_HEIGHT * MAX_WIDTH,), np.int64),
        ], timeout=timeout)
        self.registers = processor._overlay.ip_dict[CORE_NAME]["registers"]
        self.profile = profile
        self.layer_profile = []
        self.calls = 0
        self.kernel_ms = self.total_ms = 0.0
        self.weight_cache = None
        self._offset_cache = {}
        self.stopping = False

    def request_stop(self):
        """Let an active transaction finish, then reject subsequent calls."""
        self.stopping = True

    def offset(self, name):
        if name not in self._offset_cache:
            key = next((key for key in (name, name + "_r", name + "_1", name + "_r_1")
                        if key in self.registers), None)
            if key is None:
                raise ValueError("Missing hand_dw_pw_stream HWH register " + name)
            value = self.registers[key]["address_offset"]
            self._offset_cache[name] = int(value, 0) if isinstance(value, str) else int(value)
        return self._offset_cache[name]

    @staticmethod
    def _weight_key(weights):
        return weights.ctypes.data, weights.shape, weights.strides

    @staticmethod
    def supports(x, dw_weights, pw_weights, stride, pads):
        """Return whether a depthwise/pointwise pair fits this core exactly."""
        try:
            dw = np.asarray(dw_weights)
            pw = np.asarray(pw_weights)
            _shape(x, dw, np.zeros(dw.shape[0], np.float32), pw,
                   np.zeros(pw.shape[0], np.float32), stride, pads)
        except (TypeError, ValueError, IndexError):
            return False
        return True

    def _load_weights(self, dw_weights, pw_weights, channels, outputs):
        key = self._weight_key(dw_weights), self._weight_key(pw_weights)
        if self.weight_cache is not None and self.weight_cache[0] == key:
            return self.weight_cache[1:3]
        packed_dw, dw_scales = quantize_weights(dw_weights, depthwise=True)
        packed_pw, pw_scales = quantize_weights(pw_weights)
        if packed_dw.size > self.buffers[1].size or packed_pw.size > self.buffers[2].size:
            raise ValueError("hand_dw_pw_stream weights exceed allocated DDR")
        self.copy(self.buffers[1][:packed_dw.size], packed_dw)
        self.copy(self.buffers[2][:packed_pw.size], packed_pw)
        # Keep model arrays alive: an address-only key can otherwise match a
        # different array allocated at the same address after a model reload.
        self.weight_cache = (key, dw_scales, pw_scales, dw_weights, pw_weights)
        return dw_scales, pw_scales

    def convolve(self, x, dw_weights, dw_bias, pw_weights, pw_bias, stride=1, pads=(0, 0, 0, 0)):
        """Execute a depthwise 3x3/5x5 followed by a pointwise 1x1 layer."""
        started = time.perf_counter()
        (x, dw_weights, dw_bias, pw_weights, pw_bias, channels, height, width,
         outputs, kernel, top, left, out_h, out_w) = _shape(
             x, dw_weights, dw_bias, pw_weights, pw_bias, stride, pads)
        with self.lock:
            if self.stopping or self.closed or self.fault:
                raise RuntimeError("hand_dw_pw_stream is stopping, closed or faulted")
            packed_x, input_scale = quantize_activation(x)
            if packed_x.size > self.buffers[0].size or outputs * out_h * out_w > self.buffers[4].size:
                raise ValueError("hand_dw_pw_stream tensor exceeds allocated DDR")
            dw_scales, pw_scales = self._load_weights(dw_weights, pw_weights, channels, outputs)
            self.copy(self.buffers[0][:packed_x.size], packed_x)
            # The core applies these before fixed-scale ReLU6 quantization.
            params = np.empty(2 * channels, np.float32)
            with np.errstate(over="ignore", invalid="ignore"):
                params[:channels] = np.asarray(dw_scales, np.float32) * np.float32(input_scale)
            params[channels:] = dw_bias
            if not np.isfinite(params).all():
                raise ValueError("Nonfinite hand_dw_pw_stream scale or bias")
            self.copy(self.buffers[3][:params.size], params)
            copied = time.perf_counter()
            names = ("input", "dw_weights", "pw_weights", "params", "output")
            scalars = dict(height=height, width=width, channels=channels, outputs=outputs,
                           kernel=kernel, stride=stride, pad_top=top, pad_left=left,
                           out_h=out_h, out_w=out_w)
            kernel_ms = self.run([(self.offset(name), buffer) for name, buffer in zip(names, self.buffers)],
                                 [(self.offset(name), value) for name, value in scalars.items()])
            kernel_done = time.perf_counter()
            output = self.buffers[4][:outputs * out_h * out_w]
            output.invalidate()
            invalidated = time.perf_counter()
            result = np.asarray(output).copy().reshape(1, outputs, out_h, out_w).astype(np.float32)
            with np.errstate(over="ignore", invalid="ignore"):
                result *= (RELU6_SCALE * pw_scales)[None, :, None, None]
                result += pw_bias[None, :, None, None]
            if not np.isfinite(result).all():
                raise ValueError("Nonfinite hand_dw_pw_stream output")
            finished = time.perf_counter()
            self.calls += 1
            self.kernel_ms += kernel_ms
            self.total_ms += (finished - started) * 1000
            self.timings = dict(total_ms=(finished - started) * 1000,
                                quantize_copy_ms=(copied - started) * 1000,
                                kernel_ms=kernel_ms,
                                invalidate_ms=(invalidated - kernel_done) * 1000,
                                output_convert_ms=(finished - invalidated) * 1000)
            if self.profile:
                self.layer_profile.append(dict(call=self.calls, kind="dw_pw_stream",
                    input_shape=list(x.shape), dw_weight_shape=list(dw_weights.shape),
                    pw_weight_shape=list(pw_weights.shape), output_shape=list(result.shape),
                    stride=stride, kernel=kernel, kernel_ms=kernel_ms,
                    total_ms=self.timings["total_ms"],
                    host_ms=max(0.0, self.timings["total_ms"] - kernel_ms)))
            return result

    def close(self):
        self.stopping = True
        self.weight_cache = None
        super().close()


def create_hand_stream(processor, profile=False, timeout=10.0):
    """Construct the optional stream driver, or return ``None`` when absent."""
    if os.environ.get("HAND_STREAM_FUSION") != "1":
        return None
    return HandStream(processor, profile=profile, timeout=timeout) if stream_core_available(processor) else None
