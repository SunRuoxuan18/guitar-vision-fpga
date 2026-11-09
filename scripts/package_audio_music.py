"""Package PS software and browser files for an offline serial transfer."""
from pathlib import Path
import sys
import tarfile

root = Path(__file__).resolve().parents[1]
with tarfile.open(sys.argv[1], "w:gz") as archive:
    for folder in ("pynq", "web"):
        for path in sorted((root / folder).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                archive.add(str(path), arcname=str(path.relative_to(root)))
print("PACKAGED: PS software and webpage")
