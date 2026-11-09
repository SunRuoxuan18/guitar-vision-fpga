#!/bin/sh
# Run as root in a verified neck-only update stage. No bitstream is replaced.
set -eu
root=/home/xilinx/av_system
python=/usr/local/share/pynq-venv/bin/python3
sha256sum -c SHA256SUMS
modules='vision_pc_bridge.py neck_model.py neck_color.py yolo_detector.py yolo_dfl.py neck_region.py roi_fretboard.py fretboard_tracker.py hand_region.py vision_pl.py vision_hls.py tracking_native.py'
"$python" -m py_compile $modules
# Compile for the board CPU before interrupting the running service.
g++ -O3 -std=c++11 -Wno-psabi -ffp-contract=off -shared -fPIC -o libneck_region.so native/neck_region.cpp
g++ -O3 -std=c++11 -Wno-psabi -ffp-contract=off -shared -fPIC -o libtracking_cv.so native/tracking_cv.cpp $(pkg-config --cflags --libs opencv4)
g++ -O3 -std=c++11 -Wno-psabi -ffp-contract=off -shared -fPIC -o libneck_color.so native/neck_color.cpp native/neck_color_ridges.cpp native/neck_color_candidates.cpp $(pkg-config --cflags --libs opencv4)
"$python" -c 'import neck_region; assert neck_region.native_backend() is not None'
backup="$root/previous_units/neck_$(date +%Y%m%d_%H%M%S)"
config=/etc/systemd/system/fretlens-av.service.d/neck-model.conf
mkdir -p "$backup" "$(dirname "$config")"
for name in $modules; do
    if [ -e "$root/$name" ]; then cp -a "$root/$name" "$backup/$name"; fi
done
if [ -e "$root/libneck_region.so" ]; then cp -p "$root/libneck_region.so" "$backup/libneck_region.so"; fi
if [ -e "$root/libtracking_cv.so" ]; then cp -p "$root/libtracking_cv.so" "$backup/libtracking_cv.so"; fi
if [ -e "$root/libneck_color.so" ]; then cp -p "$root/libneck_color.so" "$backup/libneck_color.so"; fi
if [ -e "$root/neck_models" ]; then cp -a "$root/neck_models" "$backup/neck_models"; fi
if [ -e "$config" ]; then cp -p "$config" "$backup/neck-model.conf"; fi
committed=0
finish() {
    if [ "$committed" -eq 0 ]; then
        systemctl stop fretlens-av.service || true
        for name in $modules; do
            if [ -e "$backup/$name" ]; then cp -p "$backup/$name" "$root/$name";
            else rm -f "$root/$name"; fi
        done
        if [ -e "$backup/libneck_region.so" ]; then cp -p "$backup/libneck_region.so" "$root/libneck_region.so";
        else rm -f "$root/libneck_region.so"; fi
        if [ -e "$backup/libtracking_cv.so" ]; then cp -p "$backup/libtracking_cv.so" "$root/libtracking_cv.so";
        else rm -f "$root/libtracking_cv.so"; fi
        if [ -e "$backup/libneck_color.so" ]; then cp -p "$backup/libneck_color.so" "$root/libneck_color.so";
        else rm -f "$root/libneck_color.so"; fi
        if [ -e "$backup/neck-model.conf" ]; then cp -p "$backup/neck-model.conf" "$config";
        else rm -f "$config"; fi
        if [ -d "$backup/neck_models" ]; then cp -a "$backup/neck_models/." "$root/neck_models/";
        else rm -f "$root/neck_models/model.onnx" "$root/neck_models/model.json" "$root/neck_models/source.json"; fi
        systemctl daemon-reload
        systemctl start fretlens-av.service || true
    fi
}
trap finish EXIT
systemctl stop fretlens-av.service
cp $modules "$root/"
cp libneck_region.so "$root/"
cp libtracking_cv.so "$root/"
cp libneck_color.so "$root/"
mkdir -p "$root/neck_models"
cp model.onnx model.json source.json "$root/neck_models/"
cp neck-model.conf "$config"
systemctl daemon-reload
systemctl start fretlens-av.service
for attempt in 1 2 3 4 5 6 7 8 9 10; do
    if curl -fsS --max-time 2 http://127.0.0.1:8084/api/vision/pc/status > status.json; then
        if "$python" -c 'import json; s=json.load(open("status.json")); n=s["neck_model"]; assert s["status"]=="ready" and n["status"] in ("ready","tracking","no_detection") and not n.get("last_error"); print("NECK_DEPLOY_PASS",n)'; then
            committed=1
            printf 'backup=%s\n' "$backup"
            exit 0
        fi
    fi
    sleep 2
done
exit 1
