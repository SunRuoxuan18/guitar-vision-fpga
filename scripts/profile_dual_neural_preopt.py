"""Offline source-ONNX profiling; no model execution or hardware deployment."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
try:
    import onnx
except ModuleNotFoundError:
    sys.path.insert(0, str(ROOT / "diagnostics/yolo_deps"))
    import onnx
import numpy as np
from onnx import helper
sys.path.insert(0, str(ROOT / "scripts"))
from compile_fingertip_pl_candidate import _check_conv


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixed_shape(shapes, name):
    shape = shapes.get(name)
    if not shape or any(not isinstance(dim, int) or dim <= 0 for dim in shape):
        raise ValueError("Missing fixed positive shape: " + name)
    return shape


def conv_profile(node, shapes, initializers):
    attrs = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
    source = fixed_shape(shapes, node.input[0])
    target = fixed_shape(shapes, node.output[0])
    weights = initializers[node.input[1]]
    weight_shape = list(weights.shape)
    groups = int(attrs.get("group", 1))
    if (len(source) != 4 or len(target) != 4 or len(weight_shape) != 4
            or groups <= 0 or source[1] != weight_shape[1] * groups
            or weight_shape[0] % groups or target[1] != weight_shape[0]
            or target[0] != source[0]):
        raise ValueError("Inconsistent grouped NCHW Conv: " + node.name)
    gaps = []
    try:
        _check_conv(node, attrs, list(node.input), initializers, shapes)
    except ValueError as exc:
        gaps.append(str(exc))
    if max(source[2:]) > 256:
        gaps.append("Input spatial dimension exceeds production 256 cap")
    return {"name": node.name, "input": node.input[0], "output": node.output[0],
            "input_nchw": source, "output_nchw": target, "weight_shape": weight_shape,
            "groups": groups, "depthwise": groups == source[1] == target[1],
            "macs": math.prod(target) * math.prod(weight_shape[1:]),
            "weight_parameters": int(weights.size),
            "bias_parameters": int(initializers[node.input[2]].size) if len(node.input) > 2 else 0,
            "packed_input_elements": source[0] * math.prod(source[2:]) * ((source[1] + 7) // 8 * 8),
            "attrs": {key: value.decode() if isinstance(value, bytes) else value for key, value in attrs.items()},
            "production_conv_eligible": not gaps, "production_gaps": gaps}


def profile(path, plan_path=None):
    model = onnx.load(str(path))
    if any(node.domain or node.op_type == "PLConvolution" for node in model.graph.node):
        raise ValueError("Use original standard ONNX, not custom compatibility graph")
    bindings = []
    for value in model.graph.input:
        dims = value.type.tensor_type.shape.dim
        if dims and not dims[0].dim_value:
            bindings.append({"input": value.name, "original_batch": dims[0].dim_param, "bound": 1})
            dims[0].ClearField("dim_param")
            dims[0].dim_value = 1
    original = onnx.ModelProto()
    original.CopyFrom(model)
    from onnx.reference import ReferenceEvaluator
    folded = []
    for iteration in range(len(model.graph.node) + 1):
        model = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
        known_shapes = {}
        for value in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
            dims = value.type.tensor_type.shape.dim
            if dims and all(dim.dim_value > 0 for dim in dims):
                known_shapes[value.name] = [int(dim.dim_value) for dim in dims]
        constants = {item.name: onnx.numpy_helper.to_array(item) for item in model.graph.initializer}
        retained = []
        changed = False
        for node in model.graph.node:
            values = None
            if node.op_type == "Shape" and node.input[0] in known_shapes and not node.attribute:
                values = [np.array(known_shapes[node.input[0]], dtype=np.int64)]
            elif node.op_type in {"Constant", "Gather", "Add", "Sub", "Mul", "Div", "Cast", "Unsqueeze", "Squeeze", "Concat", "Slice"} and all(name in constants for name in node.input if name):
                values = ReferenceEvaluator(node).run(None, {name: constants[name] for name in node.input if name})
            if values is None:
                retained.append(node)
                continue
            changed = True
            folded.append({"name": node.name, "op": node.op_type})
            for name, value in zip(node.output, values):
                constants[name] = value
                model.graph.initializer.append(onnx.numpy_helper.from_array(value, name))
        del model.graph.node[:]
        model.graph.node.extend(retained)
        if not changed:
            break
    shapes, types = {}, {}
    for value in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        tensor = value.type.tensor_type
        if tensor.HasField("shape"):
            shapes[value.name] = [int(dim.dim_value) if dim.dim_value > 0 else None for dim in tensor.shape.dim]
            types[value.name] = tensor.elem_type
    initializers = {item.name: onnx.numpy_helper.to_array(item) for item in model.graph.initializer}
    convs = [conv_profile(node, shapes, initializers) for node in model.graph.node if node.op_type == "Conv"]
    names = {value.name for value in model.graph.input if value.name not in initializers}
    names.update(name for node in model.graph.node if node.op_type != "Constant" for name in node.output)
    activations, unknown = [], []
    for name in sorted(names):
        try:
            shape = fixed_shape(shapes, name)
        except ValueError:
            unknown.append(name)
            continue
        elements = math.prod(shape)
        activations.append({"name": name, "shape": shape, "elements": elements,
                            "source_bytes": elements * np.dtype(helper.tensor_dtype_to_np_dtype(types[name])).itemsize,
                            "hypothetical_int8_bytes": elements})
    counts = dict(sorted(Counter(node.op_type for node in original.graph.node).items()))
    original_initializers = {item.name: onnx.numpy_helper.to_array(item) for item in original.graph.initializer}
    plan = None
    if plan_path:
        data = json.loads(Path(plan_path).read_text(encoding="utf-8"))
        plan_counts = dict(Counter(node["op"] for node in data["nodes"]))
        plan = {"path": str(Path(plan_path).resolve()), "sha256": digest(plan_path),
                "source_hash_matches": data.get("source_sha256") == digest(path),
                "operator_counts": plan_counts, "model_contract": data.get("model_contract"),
                "conv_count_matches": plan_counts.get("PLConvolution", 0) == len(convs)}
        if not plan["source_hash_matches"] or not plan["conv_count_matches"]:
            raise ValueError("Compiled plan does not match source")
    return {"path": str(Path(path).resolve()), "sha256": digest(path),
            "validation_scope": "Original graph shape inference only; no checker, runtime or accuracy claim",
            "batch_bindings": bindings, "analysis_only_folded_shape_constants": folded,
            "op_counts": counts, "conv_count": len(convs),
            "conv_macs": sum(row["macs"] for row in convs),
            "depthwise_macs": sum(row["macs"] for row in convs if row["depthwise"]),
            "initializer_parameters": sum(int(value.size) for value in original_initializers.values()),
            "initializer_bytes": sum(value.nbytes for value in original_initializers.values()),
            "conv_weight_parameters": sum(row["weight_parameters"] for row in convs),
            "conv_bias_parameters": sum(row["bias_parameters"] for row in convs),
            "peak_single_activation": max(activations, key=lambda row: row["elements"]),
            "activation_scope": "Largest single logical tensor, not live-set, DDR peak or BRAM requirement; int8 hypothetical",
            "unknown_activation_shapes": unknown,
            "compiler_op_gaps": {op: count for op, count in counts.items() if op not in {"Conv", "Clip", "Add", "Constant", "Identity"}},
            "compiler_note": "Op-set comparison only; Constant-use and full graph contracts need compiler validation",
            "production_conv_gap_count": sum(not row["production_conv_eligible"] for row in convs),
            "convs": convs, "compiled_plan": plan}


def resources(path):
    rows = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        fields = [field.strip() for field in line.split("|")[1:-1]]
        if len(fields) == 10 and fields[0] in {"vision_pl_wrapper", "hand_conv_q16"}:
            values = [int(value) for value in fields[2:]]
            rows[fields[0]] = {"lut": values[0], "dsp": values[-1], "bram36_equivalent": values[-3] + values[-2] / 2}
    integrated, core = rows["vision_pl_wrapper"], rows["hand_conv_q16"]
    fixed = {key: integrated[key] - core[key] for key in integrated}
    capacity = {"lut": 53200, "dsp": 220, "bram36_equivalent": 140}
    return {"report": str(Path(path).resolve()), "sha256": digest(path), "integrated": integrated,
            "existing_core": core, "other_fixed": fixed, "device_capacity_xc7z020": capacity,
            "replacement_one_core_or_two_new_cores_combined_budget": {key: capacity[key] - fixed[key] for key in fixed},
            "additional_second_core_budget_keep_existing": {key: capacity[key] - integrated[key] for key in fixed},
            "new_core_resources": None,
            "scope": "Arithmetic budgets only; new cores unsynthesized; route/timing/AXI overhead and fit unproven"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    release = Path("D:/FPGA/datasets/guitar_fingertip_pilot/pl_candidate_release_20261026")
    parser.add_argument("--fingertip", type=Path, default=release / "fingertip_heatmap_candidate.onnx")
    parser.add_argument("--plan", type=Path, default=release / "graph.json")
    parser.add_argument("--neck", type=Path, default=ROOT / "models/neck/candidates/neck320.onnx")
    parser.add_argument("--resource-report", type=Path, default=ROOT / "output/hand_first_fold_20261026/hand_neural_utilization.rpt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "diagnostics/dual_neural_preopt_20261028")
    args = parser.parse_args()
    models = {"fingertip": profile(args.fingertip, args.plan), "neck": profile(args.neck)}
    bounds = {name: row["conv_macs"] / 6_400_000 for name, row in models.items()}
    report = {"models": models, "resources": resources(args.resource_report),
              "ideal_compute": {"clock_hz": 100_000_000, "output_lanes": 8, "input_simd": 8,
                                "macs_per_cycle": 64, "gmac_per_second": 6.4, "gops_per_second_2ops_per_mac": 12.8,
                                "conv_only_lower_bound_ms": bounds,
                                "shared_serial_conv_only_lower_bound_ms": sum(bounds.values()),
                                "two_independent_cores_pair_conv_only_lower_bound_ms": max(bounds.values()),
                                "assumptions": ["II=1, full 64 MAC/cycle utilization even for depthwise",
                                                "No padding/tails, memory, transfers, PS ops, pipeline, launch or scheduling costs",
                                                "Two-core bound assumes two identical cores concurrently; fit unproven"],
                                "interpretation": "Ideal arithmetic lower bounds, NOT predicted or measured latency"},
              "software_boundary": ["Head decoding and NMS remain PS, even when NMS absent from graph",
                                    "SiLU Sigmoid/Mul, Resize, Concat, MaxPool and head/shape ops lack this compiler path",
                                    "Quantization accuracy, scaling, bandwidth, synthesis and routing require separate evidence"],
              "tool_versions": {"onnx": onnx.__version__, "numpy": np.__version__}}
    lines = ["# Offline dual-neural pre-optimization profile", "", "Source ONNX shape/accounting only. No deployment or measured-latency claim.", ""]
    for name, row in models.items():
        peak = row["peak_single_activation"]
        lines.append(f"- {name}: {row['conv_count']} Conv; {row['conv_macs']:,} MAC ({row['depthwise_macs']:,} depthwise); {row['conv_weight_parameters']:,} Conv weights; {row['initializer_parameters']:,} initializer elements; peak single tensor {peak['elements']:,} elements / {peak['source_bytes']:,} source bytes; {row['production_conv_gap_count']} Conv gaps; ideal bound {bounds[name]:.4f} ms.")
        lines.append(f"  Compiler op gaps: {json.dumps(row['compiler_op_gaps'], sort_keys=True)}")
        for conv in row["convs"]:
            if conv["production_gaps"]:
                lines.append(f"  {conv['name']}: {'; '.join(conv['production_gaps'])}")
    lines += ["", "100 MHz x 8 output lanes x 8 input SIMD = 6.4 GMAC/s, or 12.8 GOP/s at 2 ops/MAC.",
              f"Ideal Conv-only serial shared bound {sum(bounds.values()):.4f} ms; concurrent two-core pair bound {max(bounds.values()):.4f} ms.",
              "Ideal lower bounds only, not latency predictions or measurements. Depthwise utilization is optimistic.", "",
              "LUT / DSP / BRAM36 equivalents: integrated 36446 / 212 / 125; existing core 9942 / 141 / 49; fixed remainder 26504 / 71 / 76.",
              "Replace one core: budget 26696 / 149 / 64. Serial sharing uses that one core.",
              "Keep existing + add second core: additional budget 16754 / 8 / 15. Replace with two new cores: combined budget 26696 / 149 / 64.",
              "New core resources unknown. Arithmetic fit does not prove placement/routing/timing; integration overhead excluded.", "",
              "SiLU, resize/concat/pooling, head decode and NMS remain PS/software boundaries. Existing compiler cannot compile YOLO as-is.",
              "Peak is one tensor, not live storage. Initializer count includes nonlearned constants; Conv parameters reported separately.",
              "See profile.json for hashes, batch=1 binding, plan correspondence and every Conv geometry/MAC/support gap."]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "profile.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summary = "\n".join(lines) + "\n"
    (args.output_dir / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
