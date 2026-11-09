"""PYNQ global metadata server with a read-only live timestamp socket.

Run in the board's PYNQ environment under systemd, after stopping AV users.
The original authenticated DeviceServer protocol and state remain unchanged.
"""
from contextlib import ExitStack
import os
from types import SimpleNamespace

from overlay_identity import IdentityServer


def main():
    from pynq.pl_server.device import Device
    from pynq.pl_server.server import DeviceServer
    Device.start_global = True
    devices = list(Device.devices)
    if not devices or type(devices[0]).__name__ != 'EmbeddedDevice':
        raise RuntimeError('Compact identity requires the default embedded PL device')
    endpoint = os.environ['FRETLENS_PL_IDENTITY_SOCKET']
    servers = [DeviceServer(device.tag) for device in devices]
    with ExitStack() as contexts:
        for index, server in enumerate(servers):
            server.start(False)
            if index == 0:
                contexts.enter_context(IdentityServer(SimpleNamespace(_server=server),endpoint))
        for server in servers:
            server.thread.join()


if __name__ == '__main__':
    main()
