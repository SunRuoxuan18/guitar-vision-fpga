#!/bin/sh
set -eu

runtime=/home/xilinx/fingertip_candidate_20261026/first_fold/runtime
stage=/home/xilinx/staging/endpoint_20261030
stamp="previous_endpoint_$(date +%Y%m%d_%H%M%S)"
backup="$runtime/$stamp"

mkdir -p "$backup"
cp -a \
    "$runtime/fretboard_tracker.py" \
    "$runtime/roi_fretboard.py" \
    "$runtime/vision_pc_bridge.py" \
    "$backup/"

install -o root -g root -m 0644 \
    "$stage/fretboard_tracker.py" "$runtime/fretboard_tracker.py"
install -o root -g root -m 0644 \
    "$stage/roi_fretboard.py" "$runtime/roi_fretboard.py"
install -o root -g root -m 0644 \
    "$stage/vision_pc_bridge.py" "$runtime/vision_pc_bridge.py"

cd "$runtime"
/usr/local/share/pynq-venv/bin/python3 -m py_compile \
    fretboard_tracker.py roi_fretboard.py vision_pc_bridge.py
/usr/local/share/pynq-venv/bin/python3 -c \
    'import fretboard_tracker, roi_fretboard, vision_pc_bridge; print("BOARD_IMPORT_OK")'

printf 'BACKUP=%s\n' "$backup"
