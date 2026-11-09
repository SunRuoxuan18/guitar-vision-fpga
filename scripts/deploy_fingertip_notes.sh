#!/bin/sh
set -eu
root=/home/xilinx/fingertip_candidate_20261026/first_fold/runtime
stage=/tmp/fingertip_notes_20261027
python=/usr/local/share/pynq-venv/bin/python3
cd "$stage"
sha256sum -c SHA256SUMS
"$python" -m py_compile visual_positions.py fingertip_notes.py vision_pc_bridge.py
backup="$root/previous_notes_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$backup"
cp -p "$root/visual_positions.py" "$root/vision_pc_bridge.py" "$backup/"
if [ -f "$root/fingertip_notes.py" ]; then cp -p "$root/fingertip_notes.py" "$backup/"; fi
committed=0
finish() {
    if [ "$committed" -eq 0 ]; then
        cp -p "$backup/visual_positions.py" "$backup/vision_pc_bridge.py" "$root/"
        if [ -f "$backup/fingertip_notes.py" ]; then cp -p "$backup/fingertip_notes.py" "$root/"; fi
        systemctl restart fretlens-av.service
    fi
}
trap finish EXIT
systemctl stop fretlens-av.service
cp visual_positions.py fingertip_notes.py vision_pc_bridge.py "$root/"
(cd "$root"; "$python" -c 'import visual_positions, fingertip_notes, vision_pc_bridge; print("NOTES_IMPORT_PASS")')
systemctl start fretlens-av.service
for attempt in 1 2 3 4 5 6 7 8 9 10 11 12; do
    if curl -fsS --max-time 2 http://127.0.0.1:8084/api/vision/pc/status > status.json; then
        "$python" -c 'import json; s=json.load(open("status.json")); assert s["status"]=="ready"; print("NOTES_DEPLOY_PASS")'
        committed=1
        printf 'backup=%s\n' "$backup"
        exit 0
    fi
    sleep 2
done
exit 1
