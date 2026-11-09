"""Run optional native coverage with: FAST_JPEG_LIBRARY=/path/libfast_jpeg.so python test_fast_jpeg.py"""
import ctypes
import os
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fast_jpeg import FastJpegEncoder, FastJpegDecoder

_LIBRARY = None
if "--library" in sys.argv:
    index = sys.argv.index("--library")
    try:
        _LIBRARY = sys.argv[index + 1]
    except IndexError:
        raise SystemExit("--library requires a path")
    del sys.argv[index:index + 2]
_LIBRARY = _LIBRARY or os.environ.get("FAST_JPEG_LIBRARY")


class FastJpegEncoderTests(unittest.TestCase):
    def test_decoder_rejects_invalid_input_before_loading_library(self):
        decoder = FastJpegDecoder('missing.so')
        for body in (b'', b'not jpeg', None, bytearray(b'\xff\xd8xx')):
            with self.assertRaises(ValueError):
                decoder(body)

    @unittest.skipUnless(_LIBRARY, 'set FAST_JPEG_LIBRARY for native coverage')
    def test_native_rgba_decode_dimensions_corruption_and_color(self):
        import cv2
        decoder = FastJpegDecoder(_LIBRARY)
        image = np.full((480,640,3), (20,100,220), np.uint8)
        body = cv2.imencode('.jpg', image)[1].tobytes()
        for _ in range(20):
            rgba = decoder(body)
            self.assertEqual(rgba.shape, (480,640,4))
            self.assertLess(np.abs(rgba[:,:,:3].astype(int)-(220,100,20)).mean(), 3)
            self.assertTrue((rgba[:,:,3] == 255).all())
        for invalid in (b'\xff\xd8bad jpeg', body[:len(body)//2],
                        cv2.imencode('.jpg', image[:200])[1].tobytes()):
            with self.assertRaises(ValueError):
                decoder(invalid)

    def test_invalid_input_does_not_load_library(self):
        encoder = FastJpegEncoder("missing.so")
        for frame in (None, np.zeros((2, 2), np.uint8), np.zeros((2, 2, 4), np.uint8),
                      np.zeros((0, 2, 3), np.uint8), np.zeros((2, 2, 3), np.float32),
                      np.zeros((2, 2, 3), np.uint8)[:, ::2]):
            with self.subTest(frame_type=type(frame).__name__):
                with self.assertRaisesRegex(ValueError, "BGR frame"):
                    encoder(frame)

    def test_missing_library_is_actionable(self):
        with self.assertRaisesRegex(RuntimeError, "Build it on the board"):
            FastJpegEncoder("definitely-missing-fast-jpeg.so")(np.zeros((2, 2, 3), np.uint8))

    @unittest.skipUnless(_LIBRARY, "set FAST_JPEG_LIBRARY or pass --library for native coverage")
    def test_native_abi_rejects_invalid_input(self):
        library = ctypes.CDLL(_LIBRARY)
        library.fast_jpeg_encode.argtypes = (
            ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)),
            ctypes.POINTER(ctypes.c_ulong),
        )
        library.fast_jpeg_encode.restype = ctypes.c_int
        output = ctypes.POINTER(ctypes.c_ubyte)()
        output_size = ctypes.c_ulong()
        self.assertEqual(library.fast_jpeg_encode(None, 0, 1, 1, 3, 70,
                                                  ctypes.byref(output), ctypes.byref(output_size)), 1)

    @unittest.skipUnless(_LIBRARY, "pass --library /path/to/libfast_jpeg.so for native coverage")
    def test_native_jpeg_and_resize(self):
        import cv2
        frame = np.full((960, 1600, 3), (20, 100, 220), np.uint8)
        jpeg, resolution, original = FastJpegEncoder(_LIBRARY)(frame, quality=70)
        decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(jpeg[:2], b"\xff\xd8")
        self.assertEqual(resolution, {"width": 1200, "height": 720})
        self.assertEqual(original, {"width": 1600, "height": 960})
        self.assertEqual(decoded.shape, (720, 1200, 3))
        self.assertLess(np.abs(decoded.astype(np.int16) - (20, 100, 220)).mean(), 5)

    @unittest.skipUnless(_LIBRARY, "set FAST_JPEG_LIBRARY for native coverage")
    def test_native_destination_grows_for_detailed_frames(self):
        import cv2
        frame = np.random.RandomState(42).randint(0, 256, (480, 640, 3)).astype(np.uint8)
        encoder = FastJpegEncoder(_LIBRARY)
        for _ in range(3):
            jpeg, _, _ = encoder(frame, quality=95)
            self.assertGreater(len(jpeg), 65536)
            decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            self.assertEqual(decoded.shape, frame.shape)

    @unittest.skipUnless(_LIBRARY, "set FAST_JPEG_LIBRARY for native coverage")
    def test_native_odd_dimensions_and_repeated_encoding(self):
        import cv2
        encoder = FastJpegEncoder(_LIBRARY)
        frame = np.zeros((49, 65, 3), np.uint8)
        frame[:, :32] = (240, 30, 15)
        frame[:, 32:] = (10, 40, 230)
        for _ in range(12):
            jpeg, resolution, _ = encoder(frame)
            decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            self.assertEqual(decoded.shape, frame.shape)
            self.assertEqual(resolution, {"width": 65, "height": 49})
            self.assertGreater(float(decoded[20, 10, 0]), 200)
            self.assertGreater(float(decoded[20, 50, 2]), 200)


if __name__ == "__main__":
    unittest.main()
