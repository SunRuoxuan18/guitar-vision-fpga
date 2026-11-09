"""Optional batched PS OpenCV calls with the Python reference as fallback."""
import ctypes
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def library():
    path = Path(__file__).with_name('libtracking_cv.so')
    if not path.is_file():
        return None
    native = ctypes.CDLL(str(path))
    native.tracking_cv_patches.argtypes = ([ctypes.c_void_p]*2+[ctypes.c_int]*2+
        [ctypes.c_void_p,ctypes.c_int]+[ctypes.c_void_p]*3)
    native.tracking_cv_patches.restype = ctypes.c_int
    native.tracking_cv_skin.argtypes = [ctypes.c_void_p]+[ctypes.c_int]*2+[
        ctypes.c_void_p,ctypes.c_float,ctypes.c_float,ctypes.c_int,ctypes.c_void_p]
    native.tracking_cv_skin.restype = ctypes.c_int
    return native
