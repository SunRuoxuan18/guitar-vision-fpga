"""Export the fixed PL hand graph without requiring ONNX on the board."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'diagnostics/yolo_deps'))
import onnx
from onnx import helper, numpy_helper

SUPPORTED = {'PLConvolution', 'PLDense', 'Transpose', 'Clip', 'Add', 'Pad',
             'MaxPool', 'GlobalAveragePool', 'Flatten', 'Sigmoid', 'PRelu',
             'Upsample', 'Reshape', 'Concat'}


def output_order_sha256(outputs):
    """Make the requested OpenCV output order part of the graph contract."""
    return hashlib.sha256('\0'.join(outputs).encode('utf-8')).hexdigest()


def export(source, destination):
    model = onnx.load(str(source))
    nodes = []
    for node in model.graph.node:
        if node.op_type not in SUPPORTED or len(node.output) != 1:
            raise ValueError('Unsupported hand graph operation: '+node.op_type)
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        attrs = {key: val.decode() if isinstance(val, bytes) else val for key, val in attrs.items()}
        nodes.append(dict(op=node.op_type, inputs=list(node.input), output=node.output[0], attrs=attrs))
    arrays = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    outputs = [x.name for x in model.graph.output]
    plan = dict(version=1, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                inputs=[x.name for x in model.graph.input if x.name not in arrays],
                outputs=outputs, output_order_sha256=output_order_sha256(outputs), nodes=nodes)
    destination.mkdir(parents=True, exist_ok=True)
    np.savez(destination/'weights.npz', **arrays)
    plan['weights_sha256'] = hashlib.sha256((destination/'weights.npz').read_bytes()).hexdigest()
    (destination/'graph.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    print(json.dumps(dict(nodes=len(nodes), parameters=len(arrays), destination=str(destination))))


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    models = root/'models/hand'
    export(models/'handpose_estimation_mediapipe_2023feb_pl.onnx',
           models/'handpose_execution')
    export(models/'palm_detection_mediapipe_2023feb_pl.onnx',
           models/'palm_execution')
