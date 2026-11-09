"""Validate real PL neural layer output and PS/PL latency on the same inputs."""
import argparse
import json
import hashlib
from pathlib import Path
import time

import cv2
import numpy as np

from vision_pl import VisionPL
from hand_neural_pl import create_neural_engine, register
from hand_tracking import HandModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bitfile',required=True)
    parser.add_argument('--image',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--load-overlay',action='store_true')
    args = parser.parse_args()
    directory = Path(args.bitfile).resolve().parent
    cv2.setNumThreads(1)
    if args.load_overlay:
        from av_system import activate
        activate(args.bitfile,directory/'manifest.json')
    processor = VisionPL(args.bitfile,download=False)
    engine = create_neural_engine(processor,profile=True)
    if engine is None:
        raise RuntimeError('Benchmark requires an actual PL neural core')
    report = dict(passed=False,backend=engine.backend_name,graphs=[],
                  bit_sha256=hashlib.sha256(Path(args.bitfile).read_bytes()).hexdigest(),
                  image_sha256=hashlib.sha256(Path(args.image).read_bytes()).hexdigest(),
                  hardware=processor.fingerprint)
    report['software_sha256'] = {name: hashlib.sha256((directory/name).read_bytes()).hexdigest()
        for name in ('hand_neural_pl.py', 'hand_tracking.py', 'handpose_model.py',
                     'roi_fretboard.py', 'native/hand_q16.c', 'libhand_q16.so')
        if (directory/name).is_file()}
    try:
        register(engine)
        image = cv2.imread(args.image)
        if image is None:
            raise ValueError('Missing benchmark image')
        rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        for name,size,count in (('palm',192,53),('handpose',224,51)):
            x = (cv2.resize(rgb,(size,size)).astype(np.float32)/255.)[None]
            reference = cv2.dnn.readNetFromONNX(str(directory/(name+'.onnx')))
            neural = cv2.dnn.readNetFromONNX(str(directory/(name+'_pl.onnx')))
            names = reference.getUnconnectedOutLayersNames()
            reference.setInput(x)
            started = time.monotonic()
            expected = reference.forward(names)
            ps_ms = (time.monotonic()-started)*1000
            before_calls,before_kernel = engine.calls,engine.kernel_ms
            neural.setInput(x)
            started = time.monotonic()
            actual = neural.forward(names)
            total_ms = (time.monotonic()-started)*1000
            errors = [float(np.max(np.abs(a-b))) for a,b in zip(actual,expected)]
            entry = dict(model=name,ps_ms=ps_ms,ps_pl_ms=total_ms,
                         pl_layer_calls=engine.calls-before_calls,
                         pl_kernel_ms=engine.kernel_ms-before_kernel,
                         max_absolute_errors=errors,speedup=ps_ms/total_ms)
            report['graphs'].append(entry)
            print(json.dumps(entry),flush=True)
            if engine.calls-before_calls != count:
                raise AssertionError('Neural layers did not all execute on PL')
            if engine.backend_name == 'pl_q16_conv_ps_graph':
                # Palm regressors are in 192-pixel model coordinates, logits
                # are not probabilities. Quantized accuracy uses task units.
                if name == 'palm':
                    raw_index = next(i for i,a in enumerate(actual) if a.shape[-1] == 18)
                    score_index = next(i for i,a in enumerate(actual) if a.shape[-1] == 1)
                    sigmoid = lambda a: 1/(1+np.exp(-np.clip(a,-80,80)))
                    score_error = float(np.max(np.abs(sigmoid(actual[score_index])-sigmoid(expected[score_index]))))
                    entry['quantized_limits'] = dict(model_coordinate_px=.5,score_probability=.01)
                    entry['score_probability_error'] = score_error
                    if errors[raw_index] > .5 or score_error > .01:
                        raise AssertionError('Q16 palm coordinate/score discrepancy')
                else:
                    entry['quantized_limits'] = dict(landmark_coordinate_px=.5,other_outputs=.01)
                    if errors[0] > .5 or any(error > .01 for error in errors[1:]):
                        raise AssertionError('Q16 pose output discrepancy')
            else:
                for a,b in zip(actual,expected):
                    np.testing.assert_allclose(a,b,rtol=.005,atol=.01)
        rgba = cv2.cvtColor(image,cv2.COLOR_BGR2RGBA)
        normal = HandModel(directory)
        accelerated = HandModel(directory,engine)
        started = time.monotonic()
        expected = normal.detect(rgba)
        report['hand_ps_ms'] = (time.monotonic()-started)*1000
        started = time.monotonic()
        actual = accelerated.detect(rgba)
        report['hand_ps_pl_ms'] = (time.monotonic()-started)*1000
        if len(actual) != len(expected) or not actual:
            raise AssertionError('Expected the reference hand in the replay image')
        error = float(np.max(np.abs(actual[0]['landmarks']-expected[0]['landmarks'])))
        if error > .01:
            raise AssertionError('Hand landmark discrepancy exceeds one percent')
        report.update(hand_count=len(actual),landmark_error_normalized=error,
                      confidence=actual[0]['score'],pl_calls=engine.calls)
        if engine.backend_name == 'pl_q16_conv_ps_graph':
            report['roi_repeats'] = []
            proposal = expected[0]['landmarks']
            for _ in range(2):
                started = time.monotonic()
                roi_expected = normal.detect(rgba,proposal)
                ps_ms = (time.monotonic()-started)*1000
                before_calls = engine.calls
                started = time.monotonic()
                roi_actual = accelerated.detect(rgba,proposal)
                elapsed = (time.monotonic()-started)*1000
                if not roi_expected or not roi_actual:
                    raise AssertionError('Warm ROI replay hand missing')
                error = float(np.max(np.abs(roi_actual[0]['landmarks']-roi_expected[0]['landmarks'])))
                if error > .01 or engine.calls-before_calls != 51:
                    raise AssertionError('Warm ROI landmarks/call count failed')
                report['roi_repeats'].append(dict(ps_ms=ps_ms,ps_pl_ms=elapsed,
                    landmark_error_normalized=error,pl_layer_calls=engine.calls-before_calls))
            report['pl_calls'] = engine.calls
        report['passed'] = True
    except Exception as exc:
        report['error'] = str(exc)
        raise
    finally:
        report['layer_profile'] = engine.layer_profile
        by_kind = {}
        for row in engine.layer_profile:
            summary = by_kind.setdefault(row['kind'],dict(calls=0,kernel_ms=0.,total_ms=0.))
            summary['calls'] += 1
            summary['kernel_ms'] += row['kernel_ms']
            summary['total_ms'] += row['total_ms']
        report['by_kind'] = by_kind
        report['slowest_layers'] = sorted(engine.layer_profile,
            key=lambda row:row['kernel_ms'],reverse=True)[:12]
        Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
        register(None)
        engine.close()
        processor.close()
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
