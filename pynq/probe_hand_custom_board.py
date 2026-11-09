"""Verify the installed OpenCV custom-layer importer, without PL claims."""
from pathlib import Path
import cv2
import numpy as np
from hand_neural_pl import register

class ProbeEngine:
    def convolve(self,x,weights,bias,stride,pads,groups):
        return x*weights.reshape(-1)[0]+bias.reshape(-1)[0]

register(ProbeEngine())
directory = Path(__file__).resolve().parent
net = cv2.dnn.readNetFromONNX(str(directory/'custom_layer_probe.onnx'))
net.setInput(np.ones((1,1,2,2),np.float32))
np.testing.assert_array_equal(net.forward(),np.full((1,1,2,2),5,np.float32))
for filename in ('palm_pl.onnx','handpose_pl.onnx'):
    net = cv2.dnn.readNetFromONNX(str(directory/filename))
    print(filename, 'imported',len(net.getLayerNames()),flush=True)
register(None)
print('BOARD_CUSTOM_LAYER_IMPORT_PASS',cv2.__version__,flush=True)
