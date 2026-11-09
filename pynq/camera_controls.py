"""Linux V4L2 controls used to keep exposure from lowering video cadence."""

import os
import struct
import sys


def prefer_fixed_frame_rate(camera):
    if not sys.platform.startswith("linux"):
        return False
    try:
        import fcntl
    except ImportError:
        return False
    # linux/videodev2.h: VIDIOC_S_CTRL; v4l2_control is two 32-bit fields.
    descriptor = None
    try:
        descriptor = os.open("/dev/video{}".format(camera), os.O_RDWR | os.O_NONBLOCK)
        fcntl.ioctl(descriptor, 0xC008561C, struct.pack("Ii", 0x009A0903, 0))
        return True
    except OSError:
        return False
    finally:
        if descriptor is not None:
            os.close(descriptor)
