#!/bin/sh
set -eu

production=/home/xilinx/fingertip_candidate_20261026
stage=/home/xilinx/staging/fingertip_reviewed_20261030
graph_sha=7252cfa9cecc48907b392a8fff5ccd6d6c10af77b156112a785c2680e0093ea6
weights_sha=279a1504af8f773bcd680286e531f089ee2d8cc01a03cb6b7eef85f8fa32fb10

require_sha() {
    path=$1
    expected=$2
    actual=$(sha256sum "$path" | awk '{print $1}')
    if [ "$actual" != "$expected" ]; then
        echo "hash mismatch: $path" >&2
        echo "expected: $expected" >&2
        echo "actual:   $actual" >&2
        exit 1
    fi
}

require_sha "$stage/graph.json" "$graph_sha"
require_sha "$stage/weights.npz" "$weights_sha"

timestamp=$(date +%Y%m%d_%H%M%S)
backup="$production/backup_reviewed_$timestamp"
mkdir -p "$backup"

for name in graph.json weights.npz best.pt fingertip_heatmap_candidate.onnx \
            validation_replay.npz installation.json
do
    if [ -f "$production/$name" ]; then
        cp -a "$production/$name" "$backup/$name"
    fi
done

for name in graph.json weights.npz best.pt fingertip_heatmap_candidate.onnx \
            validation_replay.npz
do
    install -m 0644 "$stage/$name" "$production/$name"
done

require_sha "$production/graph.json" "$graph_sha"
require_sha "$production/weights.npz" "$weights_sha"

cat > "$production/installation_reviewed_20261030.json" <<EOF
{
  "version": "fingertip-reviewed-20261030",
  "source": "$stage",
  "backup": "$backup",
  "graph_sha256": "$graph_sha",
  "weights_sha256": "$weights_sha",
  "board_benchmark": "$stage/board_benchmark_fused.json",
  "board_benchmark_passed": true,
  "board_pl_calls": 21,
  "board_coordinate_shift_max": 0.0,
  "board_median_total_ms": 782.4079655001697,
  "board_inference_fps": 1.2781055972004711
}
EOF

echo "$backup"
