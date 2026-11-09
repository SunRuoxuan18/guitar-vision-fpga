"""Compact live identity query for a supervisor-owned PYNQ local PL server."""
import json
import os
from pathlib import Path
import socket
import tempfile
import threading

_ENDPOINT = None


def live_timestamp(timeout):
    if _ENDPOINT is None:
        return False,None
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as connection:
        connection.settimeout(timeout)
        connection.connect(_ENDPOINT)
        data = bytearray()
        while not data.endswith(b'\n'):
            block = connection.recv(256)
            if not block or len(data)+len(block) > 512:
                raise RuntimeError('Overlay identity response incomplete or oversized')
            data.extend(block)
    payload = json.loads(data)
    if not isinstance(payload,dict) or set(payload) != {'timestamp'}:
        raise RuntimeError('Invalid overlay identity response')
    return True,payload['timestamp']


class IdentityServer:
    def __init__(self,device,endpoint=None):
        self.server = getattr(device,'_server',None)
        self.endpoint = endpoint
        self.directory = self.socket = self.thread = None
        self.previous_endpoint = None
        self.stopping = threading.Event()

    def __enter__(self):
        global _ENDPOINT
        self.previous_endpoint = _ENDPOINT
        if self.server is None or not isinstance(getattr(self.server,'_data',None),list):
            # A configured endpoint is served inside the actual global PL
            # server process. Channel failure must fail closed, never cache.
            _ENDPOINT = os.environ.get('FRETLENS_PL_IDENTITY_SOCKET') or _ENDPOINT
            return self
        endpoint = self.endpoint
        if endpoint is None:
            self.directory = tempfile.TemporaryDirectory(prefix='fretlens-id-')
            endpoint = str(Path(self.directory.name)/'state.sock')
        self.socket = socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
        self.socket.bind(endpoint)
        os.chmod(endpoint,0o600)
        self.socket.listen(4)
        self.socket.settimeout(.2)
        self.thread = threading.Thread(target=self._serve,name='overlay-live-identity',daemon=True)
        self.thread.start()
        _ENDPOINT = endpoint
        return self

    def _serve(self):
        while not self.stopping.is_set():
            try:
                connection,_ = self.socket.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with connection:
                connection.settimeout(.1)
                try:
                    # PYNQ atomically replaces _data on each committed update.
                    # Never read a forked copy from the worker itself.
                    stamp = self.server._data[1]
                    connection.sendall(json.dumps({'timestamp':stamp}).encode()+b'\n')
                except (OSError,TypeError,IndexError):
                    pass

    def __exit__(self,*_):
        global _ENDPOINT
        _ENDPOINT = self.previous_endpoint
        self.stopping.set()
        if self.socket is not None:
            self.socket.close()
        if self.thread is not None:
            self.thread.join(1)
        if self.directory is not None:
            self.directory.cleanup()
        elif self.socket is not None and self.endpoint is not None:
            Path(self.endpoint).unlink(missing_ok=True)
