"""Compile the trained fingertip ONNX graph into the existing PS/PL graph format."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


SUPPORTED = {"Conv", "Clip", "Add"}
MAX_TENSOR_ELEMENTS = 1_048_576
MAX_CHANNELS = 1024
MAX_SPATIAL = 256


def _shape_map(model):
    values = list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output)
    result = {}
    for value in values:
        tensor = value.type.tensor_type
        if not tensor.HasField("shape"):
            continue
        result[value.name] = [int(dim.dim_value) if dim.dim_value else None
                              for dim in tensor.shape.dim]
    return result


def _check_conv(node, attrs, inputs, initializers, shapes):
    stride = attrs.get("strides", [1, 1])
    dilation = attrs.get("dilations", [1, 1])
    pads = attrs.get("pads", [0, 0, 0, 0])
    auto_pad = attrs.get("auto_pad", b"NOTSET")
    if (len(inputs) not in (2, 3) or len(stride) != 2 or stride[0] != stride[1]
            or stride[0] not in (1, 2) or dilation != [1, 1]
            or auto_pad not in (b"NOTSET", "NOTSET") or len(pads) != 4
            or pads[0] != pads[2] or pads[1] != pads[3]):
        raise ValueError("Unsupported PL convolution geometry: " + node.name)
    if any(name not in initializers for name in inputs[1:]):
        raise ValueError("Conv weights and bias must be static initializers: " + node.name)
    weight_shape = initializers[inputs[1]].shape
    input_shape = shapes.get(inputs[0])
    output_shape = shapes.get(node.output[0])
    if (len(weight_shape) != 4 or input_shape is None or output_shape is None
            or len(input_shape) != 4 or len(output_shape) != 4
            or any(dim is None for dim in input_shape + output_shape)
            or input_shape[0] != 1 or output_shape[0] != 1):
        raise ValueError("PL convolution needs inferred fixed batch-one NCHW shapes: " + node.name)
    _, channels, height, width = input_shape
    outputs, per_group, kernel_h, kernel_w = weight_shape
    groups = int(attrs.get("group", 1))
    top, left, bottom, right = map(int, pads)
    if (kernel_h != kernel_w or kernel_h not in (1, 3, 5)
            or channels != per_group * groups or groups not in (1, channels)
            or (groups != 1 and outputs != channels)
            or min(channels, outputs, height, width) < 1
            or max(channels, outputs) > MAX_CHANNELS or max(height, width) > MAX_SPATIAL
            or min(pads) < 0 or max(pads) > 4):
        raise ValueError("Conv exceeds hand_conv_q16 shape limits: " + node.name)
    out_h, out_w = output_shape[2:]
    packed_channels = (channels + 7) // 8 * 8
    output_channels_packed = (outputs + 7) // 8 * 8
    if (height * width * packed_channels > MAX_TENSOR_ELEMENTS
            or outputs * out_h * out_w > MAX_TENSOR_ELEMENTS):
        raise ValueError("Conv activation exceeds hand_conv_q16 DDR buffers: " + node.name)
    if groups == channels == outputs:
        packed_weights = kernel_h * kernel_w * output_channels_packed
    else:
        packed_weights = outputs * kernel_h * kernel_w * packed_channels
    if packed_weights > 8 * MAX_TENSOR_ELEMENTS:
        raise ValueError("Conv weights exceed hand_conv_q16 DDR buffer: " + node.name)


def compile_graph(source, destination):
    import onnx
    from onnx import helper, numpy_helper

    source = Path(source).resolve()
    destination = Path(destination).resolve()
    model = onnx.load(str(source))
    onnx.checker.check_model(model)
    if len(model.graph.input) != 1:
        raise ValueError("Fingertip graph must have one model input")
    batch_dim = model.graph.input[0].type.tensor_type.shape.dim[0]
    batch_dim.ClearField("dim_param")
    batch_dim.dim_value = 1
    model = onnx.shape_inference.infer_shapes(model)
    initializers = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    constants = {}
    aliases = {}
    constant_uses = {}
    for node in model.graph.node:
        for input_index, input_name in enumerate(node.input):
            constant_uses.setdefault(input_name, []).append((node.op_type, input_index))
        if node.op_type == "Identity":
            if len(node.input) != 1 or len(node.output) != 1:
                raise ValueError("Malformed Identity alias: " + node.name)
            aliases[node.output[0]] = node.input[0]

    def resolve(name):
        seen = set()
        while name in aliases:
            if name in seen:
                raise ValueError("Cyclic Identity aliases in ONNX graph")
            seen.add(name)
            name = aliases[name]
        return name

    for node in model.graph.node:
        if node.op_type == "Identity":
            continue
        if node.op_type != "Constant":
            continue
        if len(node.output) != 1 or len(node.attribute) != 1:
            raise ValueError("Only scalar Constant nodes used by Clip are supported: " + node.name)
        value = helper.get_attribute_value(node.attribute[0])
        if node.attribute[0].name != "value":
            raise ValueError("Unsupported Constant encoding: " + node.name)
        constants[node.output[0]] = numpy_helper.to_array(value)

    for name in constants:
        if any(op != "Clip" or index not in (1, 2)
               for op, index in constant_uses.get(name, [])):
            raise ValueError("Constant nodes may only provide Clip bounds: " + name)

    nodes = []
    shapes = _shape_map(model)
    conv_count = clip_count = add_count = 0
    for node in model.graph.node:
        if node.op_type in ("Constant", "Identity"):
            continue
        if node.op_type not in SUPPORTED or len(node.output) != 1:
            raise ValueError("Unsupported operator in fingertip graph: " + node.op_type)
        attrs = {item.name: helper.get_attribute_value(item) for item in node.attribute}
        node_inputs = [resolve(name) for name in node.input]
        if node.op_type == "Conv":
            _check_conv(node, attrs, node_inputs, initializers, shapes)
            attrs.pop("auto_pad", None)
            attrs["_pl_op"] = "PLConvolution"
            conv_count += 1
            inputs = node_inputs
        elif node.op_type == "Clip":
            if not 1 <= len(node.input) <= 3:
                raise ValueError("Unsupported Clip input arity: " + node.name)
            bounds = [float(-np.finfo(np.float32).max), float(np.finfo(np.float32).max)]
            for index, input_name in enumerate(node_inputs[1:]):
                value = constants.get(input_name, initializers.get(input_name))
                if value is None or np.asarray(value).size != 1:
                    raise ValueError("Clip bounds must be static scalars: " + node.name)
                bounds[index] = float(np.asarray(value).item())
            attrs["min"], attrs["max"] = bounds
            inputs = [node_inputs[0]]
            clip_count += 1
        else:
            if len(node.input) != 2:
                raise ValueError("Add must have two runtime inputs: " + node.name)
            inputs = node_inputs
            add_count += 1
        nodes.append({"op": attrs.pop("_pl_op", node.op_type), "inputs": inputs,
                      "output": node.output[0], "attrs": attrs})

    if (conv_count != 21 or clip_count != 14 or add_count != 3
            or len(nodes) != conv_count + clip_count + add_count):
        raise ValueError("Unexpected fingertip graph structure: Conv={}, Clip={}, Add={}".format(
            conv_count, clip_count, add_count))
    inputs = [resolve(value.name) for value in model.graph.input if value.name not in initializers]
    outputs = [resolve(value.name) for value in model.graph.output]
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("Fingertip graph must have one input and one output")
    input_shape, output_shape = shapes.get(inputs[0]), shapes.get(outputs[0])
    if (input_shape is None or output_shape is None or len(input_shape) != 4
            or len(output_shape) != 4 or input_shape[:2] != [1, 3]
            or output_shape[:2] != [1, 4] or input_shape[2] != input_shape[3]
            or input_shape[2] % 8 or output_shape[2:] != [input_shape[2] // 8] * 2):
        raise ValueError("Expected batch-one square RGB input and 4 heatmaps at output stride 8")

    destination.mkdir(parents=True, exist_ok=True)
    weights_path = destination / "weights.npz"
    np.savez(weights_path, **initializers)
    plan = {
        "version": 1,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "inputs": inputs,
        "outputs": outputs,
        "output_order_sha256": hashlib.sha256("\0".join(outputs).encode("utf-8")).hexdigest(),
        "nodes": nodes,
        "weights_sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
        "model_contract": {"input_nchw": input_shape, "output_nchw": output_shape,
                           "points": ["index", "middle", "ring", "little"],
                           "preprocess": "RGB float32; MMPose ImageNet mean/std"},
        "operator_counts": {"PLConvolution": conv_count, "Clip": clip_count, "Add": add_count},
    }
    (destination / "graph.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    plan = compile_graph(args.source, args.destination)
    print(json.dumps({"status": "compiled_candidate", "destination": str(args.destination.resolve()),
                      "nodes": len(plan["nodes"]), "operator_counts": plan["operator_counts"]}))


if __name__ == "__main__":
    main()
