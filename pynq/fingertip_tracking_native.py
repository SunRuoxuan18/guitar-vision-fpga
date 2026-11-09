"""Batched patch quality only; no neural imports or board ownership.

Build native/fingertip_patch_quality.c with gcc -O3 -shared -fPIC -std=c99
and -lm into libfingertip_patch_quality.so beside this module.
"""

import ctypes
from pathlib import Path

import numpy as np


class NativePatchQuality:
    """Evaluate 1-4 extracted 11x11 patches in one GIL-releasing CDLL call."""

    def __init__(self, library=None):
        path = Path(library) if library is not None else Path(__file__).with_name(
            "libfingertip_patch_quality.so")
        self.library = ctypes.CDLL(str(path))
        self.function = self.library.fingertip_patch_quality
        self.function.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int, ctypes.c_void_p]
        self.function.restype = ctypes.c_int

    def evaluate(self, previous, current, anchors, original, returned):
        """Return float64 Nx5: FB, old/new corr, anchor/new corr, new std, old std.

        Patches must be native float32 Nx11x11, coordinates native float32
        Nx2; all arrays must be finite, aligned and C-contiguous. Inputs are
        read-only and are never copied. Standard deviations use ddof=0; any
        zero-norm correlation is -1. Invalid input raises ValueError. Double
        accumulation can differ slightly from NumPy's float32 reductions.
        """
        arrays = (previous, current, anchors, original, returned)
        if any(not isinstance(a, np.ndarray) or a.dtype != np.dtype(np.float32)
               or not a.flags.c_contiguous or not a.flags.aligned for a in arrays):
            raise ValueError("Expected aligned, C-contiguous native float32 arrays")
        if previous.ndim != 3 or previous.shape[1:] != (11, 11):
            raise ValueError("Expected patches shaped (N,11,11)")
        count = previous.shape[0]
        if not 1 <= count <= 4:
            raise ValueError("Expected 1-4 points")
        if current.shape != previous.shape or anchors.shape != previous.shape:
            raise ValueError("Patch batch shapes must match")
        if original.shape != (count, 2) or returned.shape != (count, 2):
            raise ValueError("Expected coordinates shaped (N,2)")
        output = np.empty((count, 5), dtype=np.float64)
        status = self.function(*(a.ctypes.data for a in arrays), count, output.ctypes.data)
        if status != 0:
            raise ValueError("Invalid or nonfinite native patch quality input")
        return output
