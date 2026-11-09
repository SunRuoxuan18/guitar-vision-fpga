#!/bin/sh
# Update only tracking software in the currently deployed first-fold runtime.
set -eu
root=/home/xilinx/fingertip_candidate_20261026/first_fold/runtime
stage=/tmp/neck_realtime/deploy
python=/usr/local/share/pynq-venv/bin/python3
cd "$stage"
sha256sum -c SHA256SUMS
"$python" -m py_compile neck_region.py vision_pc_bridge.py
test -f "$root/libneck_region.so"
backup="$root/previous_neck_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$backup"
cp -p "$root/neck_region.py" "$root/vision_pc_bridge.py" "$backup/"
committed=0
finish() {
    if [ "$committed" -eq 0 ]; then
        cp -p "$backup/neck_region.py" "$backup/vision_pc_bridge.py" "$root/"
        systemctl restart fretlens-av.service
    fi
}
trap finish EXIT
systemctl stop fretlens-av.service
cp neck_region.py vision_pc_bridge.py "$root/"
(cd "$root"; "$python" -c 'import neck_region, vision_pc_bridge; assert neck_region.native_backend() is not None; print("NECK_IMPORT_PASS")')
systemctl start fretlens-av.service
for attempt in 1 2 3 4 5 6 7 8 9 10; do
    if curl -fsS --max-time 2 http://127.0.0.1:8084/api/vision/pc/status > status.json; then
        "$python" -c 'import json; s=json.load(open("status.json")); assert s["status"]=="ready"; print("NECK_DEPLOY_PASS")'
        committed=1
        printf 'backup=%s\n' "$backup"
        exit 0
    fi
    sleep 2
done
exit 1
