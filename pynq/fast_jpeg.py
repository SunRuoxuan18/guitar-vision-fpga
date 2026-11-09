"""PS libjpeg transport codecs; build native/fast_jpeg.c with -ljpeg."""
import ctypes
from pathlib import Path

import numpy as np

_MAX_WIDTH = 1280
_MAX_HEIGHT = 720


class FastJpegEncoder:
    """Callable native encoder returning (jpeg, resolution, source_resolution)."""
    backend_name = "native-fast"

    def __init__(self, library_path=None):
        self.library_path = Path(library_path) if library_path else Path(__file__).with_name("libfast_jpeg.so")
        self._library = None

    def _load_library(self):
        if self._library is None:
            try:
                library = ctypes.CDLL(str(self.library_path))
            except OSError as exc:
                raise RuntimeError(
                    "Native JPEG library unavailable at {}. Build it on the board with: "
                    "gcc -O3 -shared -fPIC -o libfast_jpeg.so native/fast_jpeg.c -ljpeg".format(self.library_path)
                ) from exc
            library.fast_jpeg_encode.argtypes = (
                ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)),
                ctypes.POINTER(ctypes.c_ulong),
            )
            library.fast_jpeg_encode.restype = ctypes.c_int
            library.fast_jpeg_free.argtypes = (ctypes.POINTER(ctypes.c_ubyte),)
            library.fast_jpeg_free.restype = None
            library.fast_jpeg_error_string.argtypes = (ctypes.c_int,)
            library.fast_jpeg_error_string.restype = ctypes.c_char_p
            self._library = library
        return self._library

    def __call__(self, frame, max_width=1280, max_height=720, quality=70):
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Capture must supply a uint8 HxWx3 BGR frame")
        if not frame.shape[0] or not frame.shape[1] or not frame.flags.c_contiguous:
            raise ValueError("Capture must supply a nonempty contiguous uint8 HxWx3 BGR frame")
        if not all(type(value) is int and 0 < value <= limit
                   for value, limit in ((max_width, _MAX_WIDTH), (max_height, _MAX_HEIGHT))):
            raise ValueError("max_width/max_height must be positive integers within 1280x720")
        if type(quality) is not int or not 1 <= quality <= 100:
            raise ValueError("quality must be an integer from 1 through 100")

        height, width = frame.shape[:2]
        source_resolution = {"width": width, "height": height}
        scale = min(1.0, max_width / width, max_height / height)
        if scale < 1:
            try:
                import cv2
            except ImportError as exc:
                raise RuntimeError("OpenCV is required to resize frames for native JPEG encoding") from exc
            width, height = max(1, int(width * scale)), max(1, int(height * scale))
            frame = np.ascontiguousarray(cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA))

        library = self._load_library()
        output = ctypes.POINTER(ctypes.c_ubyte)()
        output_size = ctypes.c_ulong()
        pointer = frame.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        status = library.fast_jpeg_encode(pointer, frame.nbytes, width, height, frame.strides[0], quality,
                                          ctypes.byref(output), ctypes.byref(output_size))
        if status:
            reason = library.fast_jpeg_error_string(status).decode("ascii", "replace")
            raise RuntimeError("Native JPEG encoding failed: " + reason)
        try:
            return ctypes.string_at(output, output_size.value), {"width": width, "height": height}, source_resolution
        finally:
            library.fast_jpeg_free(output)


class FastJpegDecoder(FastJpegEncoder):
    """Decode directly into a bounded RGBA array without intermediate BGR."""
    def __call__(self, body, width=640, height=480):
        if (not isinstance(body, bytes) or not 4 <= len(body) <= 1280*720*4
                or not body.startswith(b'\xff\xd8')
                or type(width) is not int or not 0 < width <= _MAX_WIDTH
                or type(height) is not int or not 0 < height <= _MAX_HEIGHT):
            raise ValueError('Invalid JPEG frame')
        library = self._load_library()
        decode = library.fast_jpeg_decode_rgba
        decode.argtypes = (ctypes.c_char_p, ctypes.c_size_t, ctypes.c_void_p,
                           ctypes.c_size_t, ctypes.c_int, ctypes.c_int)
        decode.restype = ctypes.c_int
        rgba = np.empty((height, width, 4), np.uint8)
        if decode(body, len(body), rgba.ctypes.data, rgba.nbytes, width, height):
            raise ValueError('Invalid JPEG frame or dimensions')
        return rgba
