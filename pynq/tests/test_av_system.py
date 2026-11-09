import hashlib
import json
from pathlib import Path
import tempfile
import os
import unittest

from av_system import CORES, validate_artifacts, validate_layout
from audio_fft import AUD4_REGISTERS, AudioFFT, DevMemMMIO
from unittest.mock import Mock
import numpy as np
from test_audio_music import FakeMMIO, REGISTERS


class BulkMMIOTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "posix", "Linux /dev/mem mmap contract")
    def test_aligned_unsigned_words_bounds_and_close(self):
        with tempfile.TemporaryDirectory() as folder:
            device = Path(folder) / "mapped-registers"
            device.write_bytes(bytes(8192))
            mmio = DevMemMMIO(4, 32, device=str(device))
            try:
                values = [0, 0x80000000, 0xffffffff, 123]
                for index, value in enumerate(values):
                    mmio.write(4 + index * 4, value)
                np.testing.assert_array_equal(mmio.read_words(4, 4), values)
                for offset, count in ((1, 1), (28, 2), (0, 0), (-4, 1)):
                    with self.subTest(offset=offset, count=count):
                        with self.assertRaises(ValueError):
                            mmio.read_words(offset, count)
            finally:
                mmio.close()
            with self.assertRaises(RuntimeError):
                mmio.read_words(0, 1)


class UnifiedContractTests(unittest.TestCase):
    def layout(self):
        return {name: {"phys_addr": address, "type": kind + "1.0", "addr_range": 0x10000}
                for name, (address, kind) in CORES.items()}

    def test_correct_layout(self):
        validate_layout(self.layout())

    def test_stream_core_address_type_and_range(self):
        layout = self.layout()
        core = dict(phys_addr=0x43c50000, addr_range=0x10000,
                    type='xilinx.com:hls:hand_dw_pw_stream:1.0')
        layout['hand_dw_pw_stream'] = core.copy()
        validate_layout(layout)
        for key, value in (('phys_addr', 0x43c30000), ('addr_range', 0x80),
                           ('type', 'xilinx.com:hls:wrong:1.0')):
            layout['hand_dw_pw_stream'] = dict(core, **{key: value})
            with self.assertRaisesRegex(ValueError, 'hand_dw_pw_stream'):
                validate_layout(layout)

    def test_requant_core_address_type_and_range(self):
        layout = self.layout()
        core = dict(phys_addr=0x43c40000, addr_range=0x10000,
                    type='xilinx.com:hls:hand_requant:1.0')
        layout['hand_requant'] = core.copy()
        validate_layout(layout)
        for key, value in (('phys_addr', 0x43c30000), ('addr_range', 0x80),
                           ('type', 'xilinx.com:hls:wrong:1.0')):
            layout['hand_requant'] = dict(core, **{key: value})
            with self.assertRaisesRegex(ValueError, 'hand_requant'):
                validate_layout(layout)

    def test_neural_release_requires_expected_pl_core(self):
        layout = self.layout()
        with self.assertRaisesRegex(ValueError,'hand_conv'):
            validate_layout(layout,require_neural=True)
        layout['hand_conv'] = dict(phys_addr=0x43c30000,addr_range=0x10000,
                                  type='xilinx.com:hls:hand_conv:1.0')
        validate_layout(layout,require_neural=True)
        layout['hand_conv']['phys_addr'] = 0x43c20000
        with self.assertRaisesRegex(ValueError,'hand_conv'):
            validate_layout(layout,require_neural=True)

    def test_legacy_audio_conflicts_with_flow(self):
        layout = self.layout()
        layout["music"]["phys_addr"] = 0x43c00000
        with self.assertRaisesRegex(ValueError, "music"):
            validate_layout(layout)

    def test_q16_core_is_valid_but_duplicate_neural_address_is_not(self):
        layout = self.layout()
        layout['hand_conv_q16'] = dict(phys_addr=0x43c30000,addr_range=0x10000,
            type='xilinx.com:hls:hand_conv_q16:1.0')
        validate_layout(layout,require_neural=True)
        layout['hand_conv'] = dict(phys_addr=0x43c30000,addr_range=0x10000,
            type='xilinx.com:hls:hand_conv:1.0')
        with self.assertRaisesRegex(ValueError,'exactly one'):
            validate_layout(layout,require_neural=True)

    def test_wrong_core_at_correct_address(self):
        layout = self.layout()
        layout["lk_pyramid"]["type"] = "xilinx.com:hls:roi_projection:1.0"
        with self.assertRaisesRegex(ValueError, "lk_pyramid"):
            validate_layout(layout)

    def test_truncated_audio_range(self):
        layout = self.layout()
        layout["music"]["addr_range"] = 0x1000
        with self.assertRaisesRegex(ValueError, "range"):
            validate_layout(layout)

    def test_mixed_release_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            bitfile = Path(folder) / "av_system.bit"
            manifest = Path(folder) / "manifest.json"
            hashes = {}
            for suffix in (".bit", ".hwh"):
                body = suffix.encode()
                bitfile.with_suffix(suffix).write_bytes(body)
                hashes[suffix] = hashlib.sha256(body).hexdigest()
            manifest.write_text(json.dumps({"sha256": hashes}), encoding="utf-8")
            self.assertEqual(validate_artifacts(bitfile, manifest), hashes)
            bitfile.with_suffix(".hwh").write_bytes(b"old design")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                validate_artifacts(bitfile, manifest)

    def test_unchanged_sequence_does_not_freeze_or_read_bins(self):
        driver = AudioFFT.__new__(AudioFFT)
        driver.registers = AUD4_REGISTERS
        driver.mmio = Mock()
        driver.mmio.read.return_value = 7
        driver.read_snapshot = Mock(return_value="fresh")
        self.assertIsNone(driver.read_snapshot_if_new(7))
        driver.mmio.write.assert_not_called()
        driver.read_snapshot.assert_not_called()
        self.assertEqual(driver.read_snapshot_if_new(6), "fresh")
        self.assertEqual(driver.read_snapshot_if_new(None), "fresh")

    def test_bulk_snapshot_preserves_freeze_and_bin_order(self):
        class BulkMMIO(FakeMMIO):
            bulk_calls = 0

            def read_words(self, offset, count):
                self.bulk_calls += 1
                return np.array([self.read(offset+i*4) for i in range(count)], dtype=np.uint32)
        reference = AudioFFT(mmio=FakeMMIO(), register_map=REGISTERS).read_snapshot()
        mmio = BulkMMIO()
        snapshot = AudioFFT(mmio=mmio, register_map=REGISTERS).read_snapshot()
        np.testing.assert_array_equal(snapshot.power, reference.power)
        self.assertEqual(snapshot.sequence, reference.sequence)
        self.assertEqual(mmio.bulk_calls, 1)
        self.assertFalse(snapshot.power.flags.writeable)
        self.assertFalse(mmio.read(REGISTERS.control_offset) & REGISTERS.freeze_mask)


if __name__ == "__main__":
    unittest.main()
