#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
exec python3 -u -B pynq/audio_music_server.py --dev-mem --host 0.0.0.0 --port 8081 "$@"
