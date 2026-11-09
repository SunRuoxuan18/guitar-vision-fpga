"""PL conversion of raw convolution sums into the next packed operand."""
import numpy as np

from vision_hls import HLSKernel


class AccumulatorTensor:
    """Borrowed PL sums, valid only until the owning core runs again."""
    ndim = 4

    def __init__(self, engine, shape, scales, bias, generation):
        self.engine, self.shape = engine, tuple(shape)
        self.scales, self.bias = scales, bias
        self.generation = generation
        self.clip_min, self.clip_max = -np.finfo(np.float32).max, np.finfo(np.float32).max

    def check(self):
        if self.generation != self.engine.output_generation:
            raise RuntimeError('PL accumulator tensor was overwritten')

    def clipped(self, low, high):
        self.check()
        self.clip_min = np.clip(self.clip_min, low, high)
        self.clip_max = np.clip(self.clip_max, low, high)
        return self


class HandRequant(HLSKernel):
    def __init__(self, processor):
        super().__init__(processor, 'hand_requant', [((2048,), np.float32), ((2,), np.float32)], timeout=10.)
        self.registers = processor._overlay.ip_dict['hand_requant']['registers']
        self.offsets = {}
        for name in ('input', 'params', 'output', 'metadata', 'channels', 'pixels', 'clip_min', 'clip_max'):
            key = next((key for key in (name, name+'_r', name+'_1', name+'_r_1') if key in self.registers), None)
            if key is None:
                self.close()
                raise ValueError('Missing requant register '+name)
            offset = self.registers[key]['address_offset']
            self.offsets[name] = int(offset, 0) if isinstance(offset, str) else int(offset)

    def convert(self, tensor, source, destination):
        tensor.check()
        _, channels, height, width = tensor.shape
        pixels = height*width
        if not (0 < channels <= 1024 and 0 < channels*pixels <= 1048576):
            raise ValueError('Requant tensor exceeds hardware bounds')
        size = pixels*((channels+7)//8*8)
        if size > 1048576:
            raise ValueError('Requant padded tensor exceeds hardware bounds')
        if size > destination.size:
            raise ValueError('Requant destination exceeds allocated DDR')
        with self.lock:
            params, metadata = self.buffers
            params[:channels] = tensor.scales
            params[1024:1024+channels] = tensor.bias
            params.flush()
            # The source is written by PL and never dirtied or invalidated by
            # PS on this path. The destination is likewise consumed by PL.
            pointers = (('input', source), ('params', params), ('output', destination), ('metadata', metadata))
            bits = lambda x: int(np.float32(x).view(np.uint32))
            elapsed = self.run([(self.offsets[n], b) for n, b in pointers],
                [(self.offsets['channels'], channels), (self.offsets['pixels'], pixels),
                 (self.offsets['clip_min'], bits(tensor.clip_min)), (self.offsets['clip_max'], bits(tensor.clip_max))])
            metadata.invalidate()
            scale, status = float(metadata[0]), float(metadata[1])
            if status != 0 or not np.isfinite(scale) or scale <= 0:
                raise RuntimeError('PL requant failed: status={}'.format(status))
            return scale, elapsed
