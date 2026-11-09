"""Find safe depthwise-Clip-pointwise streaming-fusion candidates in an ONNX graph."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
YOLO_DEPS = ROOT / "diagnostics" / "yolo_deps"

MAIN_LIMITS = {
    "max_input_channels": 144,
    "max_output_channels": 24,
    "max_height": 32,
    "max_width": 32,
    "max_width_times_channels": 4096,
    "kernels": [3, 5],
    "strides": [1, 2],
}
LEGACY_LIMITS = {
    "max_input_channels": 96,
    "max_output_channels": 32,
    "max_height": 32,
    "max_width": 32,
    "max_width_times_channels": 4096,
    "kernels": [3, 5],
    "strides": [1, 2],
}


def _attribute_map(node, helper):
    return {attribute.name: _json_safe(helper.get_attribute_value(attribute)) for attribute in node.attribute}


def _json_safe(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _shape_from_value(value):
    tensor = value.type.tensor_type
    if not tensor.HasField("shape"):
        return None
    shape = []
    for dimension in tensor.shape.dim:
        if not dimension.HasField("dim_value") or dimension.dim_value <= 0:
            return None
        shape.append(int(dimension.dim_value))
    return shape


def _conv_fields(node):
    attrs = node.get("attrs", {})
    weight_shape = node.get("weight_shape")
    if not weight_shape or len(weight_shape) != 4:
        return None
    return {
        "group": int(attrs.get("group", 1)),
        "kernel": [int(value) for value in attrs.get("kernel_shape", weight_shape[2:])],
        "stride": [int(value) for value in attrs.get("strides", [1, 1])],
        "input_channels": int(node["input_shape"][1]) if node.get("input_shape") and len(node["input_shape"]) == 4 else None,
        "output_channels": int(weight_shape[0]),
    }


def _is_clip_0_6(node):
    attrs = node.get("attrs", {})
    return node["op"] == "Clip" and attrs.get("min") == 0.0 and attrs.get("max") == 6.0


def _fits_limits(candidate, limits):
    dw = candidate["depthwise"]
    pw = candidate["pointwise"]
    input_shape = candidate["input_shape"]
    output_shape = candidate["output_shape"]
    if not input_shape or len(input_shape) != 4 or not output_shape or len(output_shape) != 4:
        return False
    _, input_channels, input_height, input_width = input_shape
    _, output_channels, output_height, output_width = output_shape
    return (
        dw["input_channels"] is not None
        and dw["input_channels"] <= limits["max_input_channels"]
        and pw["output_channels"] <= limits["max_output_channels"]
        and input_channels <= limits["max_input_channels"]
        and input_height <= limits["max_height"]
        and input_width <= limits["max_width"]
        and input_width * input_channels <= limits["max_width_times_channels"]
        and input_width * ((input_channels + 7) // 8) <= 512
        and output_channels <= limits["max_output_channels"]
        and output_height <= limits["max_height"]
        and output_width <= limits["max_width"]
        and output_width * output_channels <= limits["max_width_times_channels"]
        and dw["kernel"][0] == dw["kernel"][1]
        and dw["kernel"][0] in limits["kernels"]
        and dw["stride"][0] == dw["stride"][1]
        and dw["stride"][0] in limits["strides"]
    )


def find_candidates(nodes):
    """Return adjacent, single-consumer depthwise -> Clip[0,6] -> pointwise triplets."""
    consumers = {}
    for index, node in enumerate(nodes):
        for tensor in node.get("inputs", []):
            consumers.setdefault(tensor, []).append(index)

    candidates = []
    for index in range(len(nodes) - 2):
        depthwise, clip, pointwise = nodes[index:index + 3]
        dw_fields = _conv_fields(depthwise)
        pw_fields = _conv_fields(pointwise)
        if not dw_fields or not pw_fields or not _is_clip_0_6(clip):
            continue
        dw_output = depthwise["output"]
        clip_output = clip["output"]
        if consumers.get(dw_output) != [index + 1] or consumers.get(clip_output) != [index + 2]:
            continue
        if not clip.get("inputs") or clip["inputs"][0] != dw_output or not pointwise.get("inputs") or pointwise["inputs"][0] != clip_output:
            continue
        if (dw_fields["group"] != dw_fields["input_channels"]
                or dw_fields["output_channels"] != dw_fields["input_channels"]
                or pw_fields["group"] != 1 or pw_fields["kernel"] != [1, 1]):
            continue
        shape = depthwise.get("output_shape")
        if shape != clip.get("input_shape") or shape != clip.get("output_shape") or shape != pointwise.get("input_shape"):
            continue
        if not shape or len(shape) != 4:
            continue
        elements = int(np.prod(shape, dtype=np.int64))
        intermediates = [
            {"tensor": dw_output, "shape": shape, "elements": elements},
            {"tensor": clip_output, "shape": shape, "elements": elements},
        ]
        candidate = {
            "node_indices": {"depthwise": index, "clip": index + 1, "pointwise": index + 2},
            "node_names": {key: node.get("name", "") for key, node in zip(("depthwise", "clip", "pointwise"), (depthwise, clip, pointwise))},
            "input_shape": depthwise.get("input_shape"),
            "output_shape": pointwise.get("output_shape"),
            "intermediates": intermediates,
            "depthwise": dw_fields,
            "pointwise": pw_fields,
            "streaming": {
                "clip_range": [0.0, 6.0],
                "quantization": {
                    "proposed_fixed_scale": 6.0 / 32767.0,
                    "status": "requires_numerical_validation",
                    "note": "Fixed Clip[0,6] scale is proposed for streaming; current dynamic activation-scale behavior must be compared numerically before use.",
                },
                "ddr_traffic": {
                    "starts_before": 2,
                    "starts_after": 1,
                    "int64_sum_bytes": elements * 8,
                    "float_ps_bytes": elements * 4,
                    "int16_operand_bytes": elements * 2,
                    "bytes_saved_at_least": elements * (8 + 2),
                    "bytes_saved_if_float_ps_materialized": elements * (8 + 4 + 2),
                    "note": "The lower bound counts int64 sums plus int16 operands (10 bytes/element); materialized PS float adds 4 bytes/element.",
                },
            },
        }
        candidate["within_main_limits"] = _fits_limits(candidate, MAIN_LIMITS)
        candidate["within_legacy_smaller_limits"] = _fits_limits(candidate, LEGACY_LIMITS)
        candidates.append(candidate)
    return candidates


def _node_records(model, helper, numpy_helper):
    initializers = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    values = list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output)
    shapes = {value.name: _shape_from_value(value) for value in values}
    records = []
    for node in model.graph.node:
        attrs = _attribute_map(node, helper)
        if node.op_type == "Clip" and len(node.input) >= 3:
            attrs["min"] = float(np.asarray(initializers[node.input[1]]).item())
            attrs["max"] = float(np.asarray(initializers[node.input[2]]).item())
        weight_shape = list(initializers[node.input[1]].shape) if node.op_type == "Conv" and len(node.input) > 1 and node.input[1] in initializers else None
        records.append({
            "name": node.name,
            "op": node.op_type,
            "inputs": list(node.input),
            "output": node.output[0] if len(node.output) == 1 else None,
            "outputs": list(node.output),
            "attrs": attrs,
            "input_shape": shapes.get(node.input[0]) if node.input else None,
            "output_shape": shapes.get(node.output[0]) if node.output else None,
            "weight_shape": weight_shape,
        })
    return records, initializers


def analyze(source, destination):
    sys.path.insert(0, str(YOLO_DEPS))
    import onnx
    from onnx import helper, numpy_helper

    source = Path(source)
    destination = Path(destination)
    model = onnx.shape_inference.infer_shapes(onnx.load(str(source)))
    nodes, weights = _node_records(model, helper, numpy_helper)
    candidates = find_candidates(nodes)
    destination.mkdir(parents=True, exist_ok=True)
    np.savez(destination / "weights.npz", **weights)
    report = {
        "version": 1,
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "weights_sha256": hashlib.sha256((destination / "weights.npz").read_bytes()).hexdigest(),
        "shape_propagation": "onnx.shape_inference.infer_shapes",
        "main_limits": MAIN_LIMITS,
        "legacy_smaller_limits": LEGACY_LIMITS,
        "nodes": nodes,
        "candidates": candidates,
        "main_candidates": [item for item in candidates if item["within_main_limits"]],
    }
    (destination / "graph.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models/hand/handpose_estimation_mediapipe_2023feb.onnx")
    parser.add_argument("--output", type=Path, default=ROOT / "diagnostics/hand_fusion_analysis")
    args = parser.parse_args()
    report = analyze(args.model, args.output)
    print(json.dumps({"output": str(args.output), "candidates": len(report["candidates"]), "main_candidates": len(report["main_candidates"])}, indent=2))


if __name__ == "__main__":
    main()
