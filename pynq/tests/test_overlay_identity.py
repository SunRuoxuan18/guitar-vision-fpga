import sys
from pathlib import Path
import socket
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from overlay_identity import IdentityServer,live_timestamp
from vision_pl import VisionPL,VisionPLError


@unittest.skipUnless(hasattr(socket,'AF_UNIX'),'Unix socket support')
class IdentityTests(unittest.TestCase):
    def test_committed_server_change_is_observed_without_cached_metadata(self):
        state = SimpleNamespace(_data=['design.bit','first'])
        with IdentityServer(SimpleNamespace(_server=state)):
            self.assertEqual(live_timestamp(.5),(True,'first'))
            state._data = ['new.bit','second']
            self.assertEqual(live_timestamp(.5),(True,'second'))
        self.assertEqual(live_timestamp(.5),(False,None))

    def test_external_server_keeps_original_validation(self):
        with patch.dict('os.environ',{},clear=True):
            with IdentityServer(SimpleNamespace(_server=None)):
                self.assertEqual(live_timestamp(.5),(False,None))

    def test_global_server_endpoint_observes_committed_state(self):
        state = SimpleNamespace(_data=['design.bit','first'])
        with tempfile.TemporaryDirectory() as directory:
            endpoint = str(Path(directory)/'identity.sock')
            with IdentityServer(SimpleNamespace(_server=state),endpoint):
                with patch.dict('os.environ',{'FRETLENS_PL_IDENTITY_SOCKET':endpoint}):
                    with IdentityServer(SimpleNamespace(_server=None)):
                        self.assertEqual(live_timestamp(.5),(True,'first'))
                        state._data = ['new.bit','second']
                        self.assertEqual(live_timestamp(.5),(True,'second'))
            self.assertFalse(Path(endpoint).exists())
        self.assertEqual(live_timestamp(.5),(False,None))

    def test_configured_global_channel_failure_is_not_a_fallback(self):
        with patch.dict('os.environ',{'FRETLENS_PL_IDENTITY_SOCKET':'missing-global.sock'}):
            with IdentityServer(SimpleNamespace(_server=None)):
                with self.assertRaises(OSError):
                    live_timestamp(.05)
        self.assertEqual(live_timestamp(.5),(False,None))

    def test_owner_channel_failure_never_falls_back_to_stale_state(self):
        with patch('overlay_identity._ENDPOINT','missing-overlay-owner.sock'):
            with self.assertRaises(OSError):
                live_timestamp(.05)

    def test_compact_identity_mismatch_is_rejected(self):
        driver = VisionPL.__new__(VisionPL)
        driver.timeout = .5
        driver._device_timestamp = 'first'
        driver._overlay = SimpleNamespace(timestamp='first',device=SimpleNamespace(timestamp='first'))
        with patch('overlay_identity.live_timestamp',return_value=(True,'first')):
            driver._ensure_active_overlay()
        with patch('overlay_identity.live_timestamp',return_value=(True,'replacement')):
            with self.assertRaises(VisionPLError):
                driver._ensure_active_overlay()
        driver._overlay.timestamp = ''
        with patch('overlay_identity.live_timestamp',return_value=(True,'first')):
            driver._ensure_active_overlay()
        with patch('overlay_identity.live_timestamp',return_value=(True,'replacement')):
            with self.assertRaises(VisionPLError):
                driver._ensure_active_overlay()
