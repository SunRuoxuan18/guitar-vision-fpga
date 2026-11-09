#!/bin/sh
set -eu

runtime=/home/xilinx/fingertip_candidate_20261026/first_fold/runtime
candidate=/tmp/roi_fretboard_frets_20261030.py
stamp=$(date +%Y%m%d_%H%M%S)
backup=/home/xilinx/vision_backups/fret_visibility_$stamp

test -f "$candidate"
mkdir -p "$backup"
cp -p "$runtime/roi_fretboard.py" "$backup/roi_fretboard.py"
cp "$candidate" "$runtime/roi_fretboard.py"
chown xilinx:xilinx "$runtime/roi_fretboard.py"
chmod 0644 "$runtime/roi_fretboard.py"
python3 -m py_compile "$runtime/roi_fretboard.py"
systemctl restart fretlens-av.service
systemctl is-active --quiet fretlens-av.service

printf 'FRET_VISIBILITY_INSTALLED backup=%s\n' "$backup"
sha256sum "$backup/roi_fretboard.py" "$runtime/roi_fretboard.py"
