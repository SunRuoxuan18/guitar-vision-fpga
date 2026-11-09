#!/bin/sh
set -eu
stage=/tmp/fretlens_realtime_update
root=/home/xilinx/av_system
python=/usr/local/share/pynq-venv/bin/python3
cd "$stage"
sha256sum -c SHA256SUMS
modules='hand_tracking.py palm_graph.py vision_pl.py vision_pc_bridge.py hand_region.py roi_fretboard.py auto_fretboard.py neck_region.py'
"$python" -m py_compile $modules
gcc -O3 -mfpu=neon -shared -fPIC -o libvision_transport.so native/vision_transport.c
backup="$root/previous_units/realtime_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$backup"
for name in $modules palm_execution libvision_transport.so; do
    if [ -e "$root/$name" ]; then cp -a "$root/$name" "$backup/$name"; fi
done
committed=0
finish() {
    if [ "$committed" -eq 0 ]; then
        systemctl stop fretlens-av.service || true
        for name in $modules libvision_transport.so; do
            if [ -f "$backup/$name" ]; then cp -p "$backup/$name" "$root/$name";
            else rm -f "$root/$name"; fi
        done
        if [ -d "$backup/palm_execution" ]; then cp -a "$backup/palm_execution/." "$root/palm_execution/";
        else rm -f "$root/palm_execution/graph.json" "$root/palm_execution/weights.npz"; fi
        systemctl start fretlens-av.service || true
    fi
}
trap finish EXIT
systemctl stop fretlens-av.service
cp $modules libvision_transport.so "$root/"
mkdir -p "$root/palm_execution"
cp palm_execution/graph.json palm_execution/weights.npz "$root/palm_execution/"
(cd "$root"; "$python" -c 'import hand_tracking, palm_graph, vision_pl, vision_pc_bridge; print("REALTIME_IMPORT_PASS")')
systemctl start fretlens-av.service
for attempt in 1 2 3 4 5 6 7 8 9 10; do
    if curl -fsS --max-time 2 http://127.0.0.1:8084/api/vision/pc/status > "$stage/status.json"; then
        "$python" -c 'import json; s=json.load(open("status.json")); assert s["status"]=="ready" and not s["neural_pl_fault"]; print("REALTIME_DEPLOY_PASS")'
        committed=1
        printf 'backup=%s\n' "$backup"
        exit 0
    fi
    sleep 2
done
exit 1
