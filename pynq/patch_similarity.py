"""Batch PS appearance checks without changing PL flow or neural execution."""
import ctypes
from functools import lru_cache
from pathlib import Path

import numpy as np


@lru_cache(maxsize=1)
def _native():
    path = Path(__file__).with_name('libhand_q16.so')
    if not path.is_file():
        return None
    library = ctypes.CDLL(str(path))
    function = getattr(library,'tracking_patch_correlations',None)
    if function is not None:
        function.argtypes = [ctypes.c_void_p]*3+[ctypes.c_int]*2+[ctypes.c_float,ctypes.c_int]
        function.restype = ctypes.c_int
    return function


def patch_correlations(first, second, minimum_energy=0., flat_template=False):
    first,second = np.asarray(first),np.asarray(second)
    if (first.dtype != np.uint8 or second.dtype != np.uint8 or first.shape != second.shape
            or first.ndim < 2 or not 1 <= first.shape[0] <= 240
            or not 1 <= first[0].size <= 256 or not np.isfinite(minimum_energy)
            or minimum_energy < 0):
        raise ValueError('Invalid tracking appearance patches')
    count,pixels = first.shape[0],first[0].size
    first,second = np.ascontiguousarray(first),np.ascontiguousarray(second)
    native = _native()
    if native is not None:
        output = np.empty(count,np.float32)
        if native(first.ctypes.data,second.ctypes.data,output.ctypes.data,
                  count,pixels,minimum_energy,int(flat_template)) != 0:
            raise RuntimeError('Native appearance check rejected valid patches')
        return output
    a,b = [patch.reshape(count,pixels).astype(np.float32) for patch in (first,second)]
    a -= a.mean(axis=1,keepdims=True)
    b -= b.mean(axis=1,keepdims=True)
    ea,eb = (a*a).sum(axis=1),(b*b).sum(axis=1)
    denominator = np.sqrt(ea*eb)
    output = np.zeros(count,np.float32)
    np.divide((a*b).sum(axis=1),denominator,out=output,where=denominator > minimum_energy)
    np.clip(output,-1,1,out=output)
    if flat_template:
        output[eb == 0] = 1
    return output
