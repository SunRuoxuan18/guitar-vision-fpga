"""Compile both transports in temporary storage; no board or neural imports.

Run with CC set, or gcc/clang/cc on PATH:
python -B -m unittest discover -s tests -p test_neural_serial.py
ARMv7 hosts also compile the existing NEON paths with their original FP flags.
"""

import ctypes
import hashlib
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class NeuralSerialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configured = os.environ.get("CC")
        compiler = shlex.split(configured) if configured else next(
            ([path] for name in ("gcc", "clang", "cc") if (path := shutil.which(name))), None)
        if not compiler:
            raise unittest.SkipTest("No C compiler; set CC or run on an ARMv7 host")
        if os.name == "nt":
            executable = shutil.which(compiler[0]) or compiler[0]
            cls.addClassCleanup(os.add_dll_directory(str(Path(executable).resolve().parent)).close)
        directory = tempfile.TemporaryDirectory(prefix="neural-serial-")
        cls.addClassCleanup(directory.cleanup)
        cls.libraries = []
        for name in ("fingertip_transport", "hand_q16"):
            source = (ROOT / "pynq/native" / (name + ".c")).as_posix()
            harness = Path(directory.name) / (name + "_test.c")
            harness.write_text(r'''
#define _POSIX_C_SOURCE 200809L
#include <pthread.h>
#include <stdlib.h>
#include <errno.h>
static int creates, fail_create;
static int test_create(pthread_t *thread, const pthread_attr_t *attr,
                       void *(*start)(void *), void *argument) {
    ++creates;
    return fail_create ? EAGAIN : pthread_create(thread, attr, start, argument);
}
#define pthread_create test_create
#include "SOURCE_PATH"
#undef pthread_create
int neural_test_creates(void) { return creates; }
void neural_test_reset(int fail) { creates = 0; fail_create = fail; }
int neural_test_env(const char *name, const char *value) {
#ifdef _WIN32
    return _putenv_s(name, value ? value : "");
#else
    return value ? setenv(name, value, 1) : unsetenv(name);
#endif
}
'''.replace("SOURCE_PATH", source), encoding="ascii")
            output = Path(directory.name) / (name + (".dll" if os.name == "nt" else ".so"))
            flags = ["-O3", "-shared", "-fPIC", "-std=c99", "-fno-fast-math", "-ffp-contract=off"]
            if platform.machine().lower().startswith("arm"):
                flags += ["-mfpu=neon", "-mfloat-abi=hard"]
            result = subprocess.run(compiler + flags + [str(harness), "-pthread", "-lm",
                                    "-o", str(output)], capture_output=True, text=True, timeout=60)
            if result.returncode:
                raise RuntimeError("Native test compilation failed:\n" + result.stdout + result.stderr)
            library = ctypes.CDLL(str(output))
            cls.libraries.append(library)
            if os.name == "nt":
                cls.addClassCleanup(cls.unload_windows, library)
            library.neural_test_creates.argtypes = []
            library.neural_test_creates.restype = ctypes.c_int
            library.neural_test_reset.argtypes = [ctypes.c_int]
            library.neural_test_reset.restype = None
            library.neural_test_env.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
            library.neural_test_env.restype = ctypes.c_int
        cls.fused, cls.hand = cls.libraries
        cls.fused.fingertip_q16_clip_pack_workspace.argtypes = [ctypes.c_void_p] * 4 + [
            ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_void_p]
        cls.fused.fingertip_q16_clip_pack_workspace.restype = ctypes.c_int
        cls.hand.hand_q16_pack.argtypes = [ctypes.c_void_p] * 2 + [
            ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
        cls.hand.hand_q16_pack.restype = ctypes.c_int
        cls.hand.hand_q16_unpack.argtypes = [ctypes.c_void_p] * 4 + [
            ctypes.c_float, ctypes.c_int, ctypes.c_int]
        cls.hand.hand_q16_unpack.restype = None

    @staticmethod
    def unload_windows(library):
        import _ctypes
        _ctypes.FreeLibrary(library._handle)

    def setUp(self):
        self.saved = {name: os.environ.get(name) for name in
                      ("FRETLENS_NEURAL_SERIAL", "FRETLENS_FINGERTIP_DIRECT")}
        self.addCleanup(self.restore_environment)
        self.environment("FRETLENS_NEURAL_SERIAL", None)
        self.environment("FRETLENS_FINGERTIP_DIRECT", None)

    def environment(self, name, value):
        for library in self.libraries:
            self.assertEqual(library.neural_test_env(
                name.encode("ascii"), None if value is None else value.encode("ascii")), 0)

    def restore_environment(self):
        for name, value in self.saved.items():
            self.environment(name, value)

    @staticmethod
    def arrays(channels=9, pixels=29131):
        rng = np.random.default_rng(1138)
        sums = rng.integers(-(2**40), 2**40, size=(channels, pixels), dtype=np.int64)
        sums.flat[:8] = [-(2**63), 2**63-1, -(2**31)-1, 2**31, -1, 0, 1, 2**24+1]
        factors = np.linspace(1e-12, 4e-12, channels, dtype=np.float32)
        bias = np.linspace(-.25, 1.25, channels, dtype=np.float32)
        return sums, factors, bias

    @staticmethod
    def digest(output, scale=None):
        data = output.tobytes()
        if scale is not None:
            data += ctypes.string_at(ctypes.byref(scale), ctypes.sizeof(scale))
        return hashlib.sha256(data).hexdigest()

    def fused_call(self, arrays, serial, direct=None, fail=False, invalid=False):
        self.environment("FRETLENS_NEURAL_SERIAL", serial)
        self.environment("FRETLENS_FINGERTIP_DIRECT", direct)
        self.fused.neural_test_reset(int(fail))
        sums, factors, bias = arrays
        channels, pixels = sums.shape
        output = np.full((pixels, (channels+7)//8*8), -12345, np.int16)
        scratch = np.empty(sums.shape, np.float32)
        scale = ctypes.c_float(-1.)
        status = self.fused.fingertip_q16_clip_pack_workspace(
            sums.ctypes.data, output.ctypes.data, factors.ctypes.data, bias.ctypes.data,
            channels, pixels, ctypes.byref(scale), scratch.ctypes.data)
        self.assertEqual(status, -1 if invalid else 0)
        if invalid:
            np.testing.assert_array_equal(output, -12345)
            self.assertEqual(scale.value, -1.)
        else:
            np.testing.assert_array_equal(output[:, channels:], 0)
        return self.digest(output, scale), self.fused.neural_test_creates()

    def hand_call(self, arrays, serial, unpack=False, fail=False, invalid=False):
        self.environment("FRETLENS_NEURAL_SERIAL", serial)
        self.hand.neural_test_reset(int(fail))
        sums, factors, bias = arrays
        channels, pixels = sums.shape
        if unpack:
            output = np.empty(sums.shape, np.float32)
            self.hand.hand_q16_unpack(sums.ctypes.data, output.ctypes.data,
                factors.ctypes.data, bias.ctypes.data, .125, channels, pixels)
            digest = self.digest(output)
        else:
            source = sums.astype(np.float32) * np.float32(1e-6)
            if invalid:
                source.flat[-1] = np.nan
            output = np.full((pixels, (channels+7)//8*8), -12345, np.int16)
            scale = ctypes.c_float(-1.)
            status = self.hand.hand_q16_pack(source.ctypes.data, output.ctypes.data,
                                           channels, pixels, ctypes.byref(scale))
            self.assertEqual(status, -1 if invalid else 0)
            if invalid:
                np.testing.assert_array_equal(output, -12345)
            else:
                np.testing.assert_array_equal(output[:, channels:], 0)
            digest = self.digest(output, scale)
        return digest, self.hand.neural_test_creates()

    def test_two_pass_fused_exact_with_channel_padding_and_pixel_tails(self):
        arrays = self.arrays()
        expected, threads = self.fused_call(arrays, None)
        self.assertEqual(threads, 2)
        self.assertEqual(self.fused_call(arrays, "1"), (expected, 0))
        self.assertEqual(self.fused_call(arrays, None, fail=True), (expected, 2))

    def test_direct_fused_exact_and_serial_suppresses_helper(self):
        arrays = self.arrays()
        arrays[0][0, 0] = 2**40
        arrays[1][0] = np.float32(1e-11)
        expected, threads = self.fused_call(arrays, None, direct="1")
        self.assertEqual(threads, 1)
        self.assertEqual(self.fused_call(arrays, "1", direct="1"), (expected, 0))
        self.assertEqual(self.fused_call(arrays, None, direct="1", fail=True), (expected, 1))

    def test_unsaturated_direct_fallback_is_still_exact(self):
        arrays = self.arrays()
        arrays[0][:] %= 1000
        expected, threads = self.fused_call(arrays, None, direct="1")
        self.assertEqual(threads, 2)
        self.assertEqual(self.fused_call(arrays, "1", direct="1"), (expected, 0))

    def test_pack_and_unpack_exact_including_existing_create_failure_fallback(self):
        for unpack in (False, True):
            with self.subTest(unpack=unpack):
                arrays = self.arrays()
                expected, threads = self.hand_call(arrays, None, unpack=unpack)
                self.assertEqual(threads, 1)
                self.assertEqual(self.hand_call(arrays, "1", unpack=unpack), (expected, 0))
                self.assertEqual(self.hand_call(arrays, None, unpack=unpack, fail=True), (expected, 1))

    def test_only_exact_one_opts_in_and_each_entry_rereads_environment(self):
        arrays = self.arrays()
        for call, threads in ((self.fused_call, 2), (self.hand_call, 1),
                              (lambda a, s: self.hand_call(a, s, unpack=True), 1)):
            expected = call(arrays, None)[0]
            for value in ("0", "", "true", "01", "10", "1", "0", "1", None):
                with self.subTest(value=value, threads=threads):
                    self.assertEqual(call(arrays, value), (expected, 0 if value == "1" else threads))

    def test_small_tensors_remain_serial_with_exact_tails(self):
        arrays = self.arrays(channels=3, pixels=19)
        for call in (self.fused_call, self.hand_call,
                     lambda a, s: self.hand_call(a, s, unpack=True)):
            expected, threads = call(arrays, None)
            self.assertEqual(threads, 0)
            self.assertEqual(call(arrays, "1"), (expected, 0))

    def test_nonfinite_rejection_and_conversion_overflow_are_preserved(self):
        for serial in (None, "1"):
            self.hand_call(self.arrays(), serial, invalid=True)
            for value in (np.nan, np.inf, -np.inf):
                arrays = self.arrays()
                arrays[1][-1] = value
                self.assertEqual(self.fused_call(arrays, serial, invalid=True)[1], 0)
            arrays = self.arrays()
            arrays[1][-1] = np.finfo(np.float32).max
            self.fused_call(arrays, serial, invalid=True)


if __name__ == "__main__":
    unittest.main()
