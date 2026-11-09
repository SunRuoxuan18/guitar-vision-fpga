"""Package a verified ONNX candidate and its PS bridge adapter, excluding PL."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import io

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads(args.metadata.read_text(encoding='utf-8'))
    digest = hashlib.sha256(args.model.read_bytes()).hexdigest()
    if metadata.get('onnx_sha256') != digest:
        raise ValueError('Model hash differs from metadata')
    files = {name: (ROOT/'pynq'/name).read_bytes() for name in
             ('vision_pc_bridge.py', 'neck_model.py', 'neck_color.py', 'yolo_detector.py', 'yolo_dfl.py',
              'neck_region.py', 'roi_fretboard.py', 'fretboard_tracker.py', 'hand_region.py',
              'vision_pl.py', 'vision_hls.py', 'tracking_native.py')}
    files.update({'model.onnx': args.model.read_bytes(),
                  'native/neck_region.cpp': (ROOT/'pynq/native/neck_region.cpp').read_bytes(),
                  'native/tracking_cv.cpp': (ROOT/'pynq/native/tracking_cv.cpp').read_bytes(),
                  'native/neck_color.cpp': (ROOT/'pynq/native/neck_color.cpp').read_bytes(),
                  'native/neck_color_ridges.cpp': (ROOT/'pynq/native/neck_color_ridges.cpp').read_bytes(),
                  'native/neck_color_candidates.cpp': (ROOT/'pynq/native/neck_color_candidates.cpp').read_bytes(),
                  'model.json': args.metadata.read_bytes(),
                  'source.json': args.model.with_name('source.json').read_bytes(),
                  'install_neck_model.sh': (ROOT/'scripts/install_neck_model.sh').read_bytes(),
                  'neck-model.conf': b'[Service]\nEnvironment=FRETLENS_NECK_MODEL=/home/xilinx/av_system/neck_models/model.onnx\nEnvironment=FRETLENS_NECK_METADATA=/home/xilinx/av_system/neck_models/model.json\nEnvironment=FRETLENS_FLOW_POINTS=32\n'})
    files['SHA256SUMS'] = ''.join(hashlib.sha256(data).hexdigest()+'  '+name+'\n'
                                 for name, data in files.items()).encode('ascii')
    with args.archive.open('xb') as stream, tarfile.open(fileobj=stream, mode='w:gz') as package:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            package.addfile(info, io.BytesIO(data))
    print(json.dumps({'archive': str(args.archive.resolve()), 'sha256': digest}))


if __name__ == '__main__':
    main()
