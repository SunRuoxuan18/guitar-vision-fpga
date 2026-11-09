#!/bin/sh
# Software-only update: keep the deployed overlay, models and audio code.
set -eu
root=/home/xilinx/av_system
stage=/tmp/fretlens_tracking_priority
python=/usr/local/share/pynq-venv/bin/python3
cd "$stage"
sha256sum -c SHA256SUMS
"$python" -m py_compile vision_pc_bridge.py work_priority.py
backup="$root/previous_units/tracking_priority_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$backup"
cp -p "$root/vision_pc_bridge.py" "$backup/vision_pc_bridge.py"
had_priority=0
if [ -f "$root/work_priority.py" ]; then
    cp -p "$root/work_priority.py" "$backup/work_priority.py"
    had_priority=1
fi
committed=0
finish() {
    if [ "$committed" -eq 0 ]; then
        cp -p "$backup/vision_pc_bridge.py" "$root/vision_pc_bridge.py"
        if [ "$had_priority" -eq 1 ]; then
            cp -p "$backup/work_priority.py" "$root/work_priority.py"
        else
            rm -f "$root/work_priority.py"
        fi
        systemctl restart fretlens-av.service
    fi
}
trap finish EXIT
systemctl stop fretlens-av.service
cp vision_pc_bridge.py work_priority.py "$root/"
(cd "$root"; "$python" -c 'import vision_pc_bridge, work_priority; print("PRIORITY_IMPORT_PASS")')
systemctl start fretlens-av.service
for attempt in 1 2 3 4 5 6 7 8 9 10; do
    if curl -fsS --max-time 2 http://127.0.0.1:8084/api/vision/pc/status > "$stage/status.json"; then
        "$python" -c 'import json; s=json.load(open("status.json")); assert s["status"]=="ready" and s["tracking_priority_enabled"]; print("TRACKING_PRIORITY_DEPLOY_PASS")'
        committed=1
        printf 'backup=%s\n' "$backup"
        exit 0
    fi
    sleep 2
done
exit 1
