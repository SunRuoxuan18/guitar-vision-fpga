"""Install staged vision files with backup, without changing the audio overlay.

Run as root after extracting software_update.tar.gz into a NEW staging folder:
python3 STAGING/pynq/install_vision_update.py --stage STAGING
"""
import argparse
from datetime import datetime
import os
from pathlib import Path
import shutil
import subprocess


def install(stage, home=Path("/home/xilinx"), system_root=Path("/")):
    stage, home = Path(stage).resolve(), Path(home).resolve()
    system_root = Path(system_root).resolve()
    if not (stage / "pynq/vision_server.py").is_file():
        raise ValueError("The stage does not contain a vision update")
    names = {"vision_server.py", "fretboard_tracker.py", "patch_similarity.py", "auto_fretboard.py",
             "camera_controls.py", "keep_camera_awake.py"}
    names.update(path.name for path in (stage / "pynq").glob("yolo_*.py"))
    present = sorted(name for name in names if (stage / "pynq" / name).is_file())
    # Compile the staged code before stopping the running service.
    for name in present:
        path = stage / "pynq" / name
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    backup = home / "vision_backups" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup.mkdir(parents=True, exist_ok=False)
    changes = [(stage / "pynq" / name, home / name) for name in present]
    unit = stage / "pynq/fretlens-vision.service"
    rule = stage / "pynq/99-fretlens-camera.rules"
    for source, target in ((unit, system_root / "etc/systemd/system/fretlens-vision.service"),
                           (rule, system_root / "etc/udev/rules.d/99-fretlens-camera.rules")):
        if source.is_file():
            changes.append((source, target))
    originals = {}
    for source, target in changes:
        if source.resolve() == target.resolve():
            raise ValueError("Extract into a new staging folder, not the installation directory")
        if target.exists():
            saved = backup / target.name
            shutil.copy2(str(target), str(saved))
            originals[target] = saved
        else:
            originals[target] = None
    active = subprocess.run(["systemctl", "is-active", "--quiet", "fretlens-vision.service"],
                            check=False, timeout=15).returncode == 0
    subprocess.run(["systemctl", "stop", "fretlens-vision.service"], check=True, timeout=30)
    applied = []
    try:
        for source, target in changes:
            applied.append(target)
            shutil.copy2(str(source), str(target))
        if unit.is_file():
            subprocess.run(["systemctl", "daemon-reload"], check=True, timeout=15)
        if rule.is_file():
            subprocess.run(["udevadm", "control", "--reload-rules"], check=True, timeout=15)
        subprocess.run(["systemctl", "start", "fretlens-vision.service"], check=True, timeout=30)
        subprocess.run(["systemctl", "is-active", "--quiet", "fretlens-vision.service"],
                       check=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        for target in reversed(applied):
            saved = originals[target]
            if saved is None:
                if target.exists():
                    target.unlink()
            else:
                shutil.copy2(str(saved), str(target))
        subprocess.run(["systemctl", "daemon-reload"], check=False, timeout=15)
        if rule.is_file():
            subprocess.run(["udevadm", "control", "--reload-rules"], check=False, timeout=15)
        if active:
            subprocess.run(["systemctl", "start", "fretlens-vision.service"], check=False, timeout=30)
        raise
    print("VISION_UPDATE_INSTALLED backup={}".format(backup))
    print("Audio bitstream and audio service were not modified.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, type=Path)
    args = parser.parse_args()
    if not hasattr(os, "geteuid") or os.geteuid() != 0:
        parser.error("Run on PYNQ Linux as root")
    install(args.stage)
