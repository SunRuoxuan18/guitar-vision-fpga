#!/bin/sh
set -eu

runtime=/home/xilinx/fingertip_candidate_20261026/first_fold/runtime
candidate=/tmp/vision_pc_bridge_fret_prime_20261030.py
stamp=$(date +%Y%m%d_%H%M%S)
backup=/home/xilinx/vision_backups/fret_measurement_prime_$stamp

test -f "$candidate"
mkdir -p "$backup"
cp -p "$runtime/vision_pc_bridge.py" "$backup/vision_pc_bridge.py"
cp "$candidate" "$runtime/vision_pc_bridge.py"
chown xilinx:xilinx "$runtime/vision_pc_bridge.py"
chmod 0644 "$runtime/vision_pc_bridge.py"
python3 -m py_compile "$runtime/vision_pc_bridge.py"
systemctl restart fretlens-av.service
systemctl is-active --quiet fretlens-av.service

printf 'FRET_MEASUREMENT_PRIME_INSTALLED backup=%s\n' "$backup"
sha256sum "$backup/vision_pc_bridge.py" "$runtime/vision_pc_bridge.py"
