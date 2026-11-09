"""Route pretrained Conv/Gemm layers through the board's PL custom layers."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'diagnostics/yolo_deps'))
import onnx
from onnx import helper

def export(source, destination):
    model = onnx.load(source)
    counts = {'Conv':0,'Gemm':0}
    for node in model.graph.node:
        if node.op_type == 'Conv':
            node.op_type = 'PLConvolution'
            counts['Conv'] += 1
        elif node.op_type == 'Gemm':
            attrs = {a.name:helper.get_attribute_value(a) for a in node.attribute}
            if attrs.get('transA',0) != 0 or attrs.get('transB',0) != 1 or attrs.get('alpha',1.) != 1. or attrs.get('beta',1.) != 1.:
                raise ValueError('Unsupported dense weights: '+node.name)
            node.op_type = 'PLDense'
            del node.attribute[:]
            counts['Gemm'] += 1
    onnx.save(model,destination)
    return counts

if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]/'models/hand'
    for name in ('palm_detection_mediapipe_2023feb','handpose_estimation_mediapipe_2023feb'):
        print(name,export(root/(name+'_cv42.onnx'),root/(name+'_pl.onnx')))
