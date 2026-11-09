"""Disable runtime power saving only for the connected Logitech C922."""

from pathlib import Path


def main():
    for device in Path("/sys/bus/usb/devices").iterdir():
        try:
            if ((device / "idVendor").read_text().strip(),
                    (device / "idProduct").read_text().strip()) != ("046d", "085c"):
                continue
            (device / "power/control").write_text("on")
            (device / "power/autosuspend").write_text("-1")
            parent = device.resolve().parent
            if (parent / "power/control").exists():
                (parent / "power/control").write_text("on")
            print("C922 runtime power saving disabled: {}".format(device.name))
        except FileNotFoundError:
            continue


if __name__ == "__main__":
    main()
