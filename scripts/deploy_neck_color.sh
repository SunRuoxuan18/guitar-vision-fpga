#!/bin/sh
# Software-only neck acquisition update; rollback on failed service validation.
set -eu
stage=${1:-/tmp/neck_color_probe_20261028}
case "$stage" in /tmp/neck_* ) ;; *) echo 'Expected a staged /tmp/neck_* directory' >&2; exit 1;; esac
root=/home/xilinx/fingertip_candidate_20261026/first_fold/runtime
python=/usr/local/share/pynq-venv/bin/python3
config=/etc/systemd/system/fretlens-av.service.d/neck-acquisition.conf
backup="$root/previous_color_neck_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$backup"
cp -p "$root/vision_pc_bridge.py" "$root/fretboard_tracker.py" "$backup/"
if [ -e "$root/neck_color.py" ]; then cp -p "$root/neck_color.py" "$backup/"; fi
if [ -e "$root/libneck_color.so" ]; then cp -p "$root/libneck_color.so" "$backup/"; fi
if [ -e "$config" ]; then cp -p "$config" "$backup/"; fi
systemctl show fretlens-av.service > "$backup/service_before.txt"
systemctl cat fretlens-av.service > "$backup/unit_before.txt"
sha256sum "$root/av_system.bit" "$root/av_system.hwh" > "$backup/hardware.sha256"
committed=0
finish() {
    if [ "$committed" -eq 0 ]; then
        cp -p "$backup/vision_pc_bridge.py" "$backup/fretboard_tracker.py" "$root/"
        if [ -e "$backup/neck_color.py" ]; then cp -p "$backup/neck_color.py" "$root/";
        else rm -f "$root/neck_color.py"; fi
        if [ -e "$backup/libneck_color.so" ]; then cp -p "$backup/libneck_color.so" "$root/";
        else rm -f "$root/libneck_color.so"; fi
        if [ -e "$backup/neck-acquisition.conf" ]; then cp -p "$backup/neck-acquisition.conf" "$config";
        else rm -f "$config"; fi
        systemctl daemon-reload
        systemctl restart fretlens-av.service
        printf 'ROLLED_BACK=%s\n' "$backup"
    fi
}
trap finish EXIT
"$python" -m py_compile "$stage/pynq/neck_color.py" "$stage/pynq/vision_pc_bridge.py" "$stage/pynq/fretboard_tracker.py"
"$python" -c 'import sys;sys.path.insert(0,sys.argv[1]);from neck_color import native_backend;assert native_backend() is not None' "$stage/pynq"
cp "$stage/pynq/neck_color.py" "$stage/pynq/vision_pc_bridge.py" "$stage/pynq/fretboard_tracker.py" "$root/"
cp "$stage/pynq/libneck_color.so" "$root/"
cp "$stage/neck-acquisition.conf" "$config"
systemctl daemon-reload
systemctl restart fretlens-av.service
for attempt in $(seq 1 25); do
    if curl -fsS --max-time 2 http://127.0.0.1:8084/api/vision/pc/status > "$backup/status_after.json"; then
        if "$python" -c 'import json,sys; s=json.load(open(sys.argv[1])); assert s["status"]=="ready" and not s.get("error"); assert s["neck_model"]["acquisition_mode"]=="color_border_then_flow"; assert not s["neck_model"]["model_primary_enabled"]; assert s["fingertip_neural_enabled"]; print("COLOR_NECK_DEPLOY_PASS")' "$backup/status_after.json"; then
            sha256sum -c "$backup/hardware.sha256"
            committed=1
            printf 'BACKUP=%s\n' "$backup"
            exit 0
        fi
    fi
    sleep 1
done
exit 1
