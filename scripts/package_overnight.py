"""Create the offline software update and checksums; never access the board."""
import hashlib
import json
from pathlib import Path
import tarfile


root = Path(__file__).resolve().parents[1]
destination = root / "output/overnight_20261016"
destination.mkdir(parents=True, exist_ok=True)
files = []
for directory in ("pynq", "web"):
    for path in sorted((root / directory).rglob("*")):
        if (path.is_file() and "__pycache__" not in path.parts and
                "tests" not in path.parts and not path.name.startswith("tests_") and
                path.suffix not in (".pyc", ".log")):
            files.append(path)
files.extend(root / name for name in ("RETURN_CHECKLIST.md", "LICENSE", "THIRD_PARTY.md",
                                      "diagnostics/restore_audio_after_camera_reboot.py"))
for pattern in ("yolo_*.py", "yolo_*.ps1", "yolo_*.json", "yolo_*.md", "requirements-yolo.txt"):
    files.extend(sorted((root / "vision").glob(pattern)))
manifest = {
    "version": "overnight_20261016", "deployment_status": "prepared_offline_not_deployed",
    "bitstream_changed": False,
    "audio_bit_sha256": "aced9993f1beddde4a4bca3fe667405252e4c4345d4288d0078a74a15e90b610",
    "files": [{"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
               "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in files]
}
with tarfile.open(str(destination / "software_update.tar.gz"), "w:gz") as archive:
    for path in files:
        archive.add(str(path), arcname=path.relative_to(root).as_posix())
manifest["archive_sha256"] = hashlib.sha256((destination / "software_update.tar.gz").read_bytes()).hexdigest()
(destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("PREPARED_OFFLINE", len(files), "files", destination)
