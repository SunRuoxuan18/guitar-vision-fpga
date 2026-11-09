"""Optional native arithmetic; Python retains all recognition decisions."""
import ctypes
from pathlib import Path

import numpy as np


class AudioNative:
    def __init__(self):
        self.library = ctypes.CDLL(str(Path(__file__).with_name("libaudio_fit.so")))
        array = np.ctypeslib.ndpointer
        self.library.audio_sparse_dot.argtypes = [array(np.float64, flags="C_CONTIGUOUS"),
            array(np.intp, flags="C_CONTIGUOUS"), array(np.float32, flags="C_CONTIGUOUS"),
            ctypes.c_size_t, ctypes.c_size_t, array(np.float64, flags="C_CONTIGUOUS")]
        self.library.audio_sparse_dot.restype = None
        self.library.audio_sparse_dot32.argtypes = [array(np.float32, flags="C_CONTIGUOUS"),
            array(np.intp, flags="C_CONTIGUOUS"), array(np.float32, flags="C_CONTIGUOUS"),
            ctypes.c_size_t, ctypes.c_size_t, array(np.float32, flags="C_CONTIGUOUS")]
        self.library.audio_sparse_dot32.restype = None
        self.library.audio_nnls_weights.argtypes = [array(np.float32, flags="C_CONTIGUOUS"),
            array(np.float64, flags="C_CONTIGUOUS"), ctypes.c_size_t,
            array(np.float32, flags="C_CONTIGUOUS")]
        self.library.audio_nnls_weights.restype = None

    def prepare(self, bins, values, spectrum_length):
        bins, values = np.ascontiguousarray(bins), np.ascontiguousarray(values)
        if (bins.dtype != np.intp or values.dtype != np.float32 or bins.shape != values.shape
                or bins.ndim != 2 or bins.shape[1] > 2048
                or bins.size and (bins.min() < 0 or bins.max() >= spectrum_length)):
            raise ValueError("Invalid native sparse dot buffers")
        # Validate this immutable bank once. NumPy checks subsequent row selections.
        self.bins, self.values, self.spectrum_length = bins, values, spectrum_length
        self.bins.flags.writeable = self.values.flags.writeable = False

    def correlate(self, spectrum, indices=None):
        if spectrum.dtype not in (np.float32, np.float64):
            return None
        if spectrum.shape != (self.spectrum_length,):
            raise ValueError("Invalid native spectrum length")
        spectrum = np.ascontiguousarray(spectrum)
        bins, values = self.bins, self.values
        if indices is not None:
            bins, values = bins[indices], values[indices]
        output = np.empty(len(bins), dtype=spectrum.dtype)
        function = self.library.audio_sparse_dot32 if spectrum.dtype == np.float32 else self.library.audio_sparse_dot
        function(spectrum, bins, values, len(bins), bins.shape[1], output)
        return output

    def fit(self, design, spectrum):
        if not len(design):
            return np.empty(0, dtype=np.float32), spectrum.copy()
        gram = np.ascontiguousarray(design @ design.T, dtype=np.float32)
        target = np.ascontiguousarray(design @ spectrum, dtype=np.float64)
        weights = np.empty(len(design), dtype=np.float32)
        self.library.audio_nnls_weights(gram, target, len(design), weights)
        return weights, spectrum - weights @ design
