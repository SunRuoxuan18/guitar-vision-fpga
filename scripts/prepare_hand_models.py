"""Export fixed-shape OpenCV 4.2 compatible graphs; preserve upstream weights."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'diagnostics/yolo_deps'))
import numpy as np
import onnx
from onnx import helper, numpy_helper

root = Path(__file__).resolve().parents[1]
for name in ('palm_detection_mediapipe_2023feb', 'handpose_estimation_mediapipe_2023feb'):
    model = onnx.shape_inference.infer_shapes(onnx.load(root/'models/hand'/(name+'.onnx')))
    constants = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
              for v in list(model.graph.value_info)+list(model.graph.input)}
    nodes = []
    for node in model.graph.node:
        if node.op_type == 'Resize':
            size = constants[node.input[3]]
            scales = np.asarray(size / np.asarray(shapes[node.input[0]]), np.float32)
            node = helper.make_node('Upsample', [node.input[0]], list(node.output),
                                    name=node.name, mode='bilinear', scales=scales.tolist())
        elif node.op_type == 'Pad' and len(node.input) > 1:
            pads = constants[node.input[1]].tolist()
            node = helper.make_node('Pad', [node.input[0]], list(node.output),
                                    name=node.name, pads=pads, mode='constant', value=0.)
        elif node.op_type == 'Clip' and len(node.input) > 1:
            node = helper.make_node('Clip', [node.input[0]], list(node.output), name=node.name,
                                    min=float(constants[node.input[1]]), max=float(constants[node.input[2]]))
        elif node.op_type == 'Squeeze' and len(node.input) > 1:
            if constants[node.input[1]].tolist() != [2,3]:
                raise ValueError('Unexpected squeeze axes')
            node = helper.make_node('Flatten', [node.input[0]], list(node.output), name=node.name, axis=1)
        elif node.op_type == 'Gemm':
            trans_b = next((a for a in node.attribute if a.name == 'transB'), None)
            if trans_b is not None and trans_b.i == 0:
                key = node.input[1]+'_transposed'
                model.graph.initializer.append(numpy_helper.from_array(constants[node.input[1]].T.copy(),key))
                node.input[1] = key
                trans_b.i = 1
        if node.op_type == 'Conv':
            pads = next((a for a in node.attribute if a.name == 'pads'), None)
            if pads is not None and list(pads.ints[:2]) != list(pads.ints[2:]):
                top, left, bottom, right = pads.ints
                padded = node.name+'_explicit_padding'
                nodes.append(helper.make_node('Pad', [node.input[0]], [padded],
                    name=padded, pads=[0,0,top,left,0,0,bottom,right], mode='constant', value=0.))
                node.input[0] = padded
                del pads.ints[:]
                pads.ints.extend([0,0,0,0])
        nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    used = {i for n in nodes for i in n.input}
    kept = [t for t in model.graph.initializer if t.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    path = root/'models/hand'/(name+'_cv42.onnx')
    onnx.save(model, path)
    print(path)
