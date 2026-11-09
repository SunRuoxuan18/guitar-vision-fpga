"""Compare measured neural runs; reject mismatched inputs or failed checks."""
import argparse
import json
from pathlib import Path


def compare(before, after):
    if not before.get('passed') or not after.get('passed'):
        raise ValueError('Both hardware benchmarks must pass accuracy checks')
    supported = ('pl_fp32_conv_ps_graph','pl_q16_conv_ps_graph')
    if before.get('backend') not in supported or after.get('backend') not in supported:
        raise ValueError('Both benchmarks must execute a verified PL neural backend')
    if before.get('image_sha256') != after.get('image_sha256') or not after.get('image_sha256'):
        raise ValueError('Both benchmarks must identify the same input image')
    result = {'baseline_bit':before['bit_sha256'],'candidate_bit':after['bit_sha256'],
              'baseline_backend':before['backend'],'candidate_backend':after['backend'],
              'precision_changed':before['backend'] != after['backend'],
              'image_sha256':after['image_sha256'],'graphs':[]}
    old = {row['model']:row for row in before['graphs']}
    if set(old) != {row['model'] for row in after['graphs']}:
        raise ValueError('Graph sets differ')
    for row in after['graphs']:
        previous = old[row['model']]
        if previous['pl_layer_calls'] != row['pl_layer_calls']:
            raise ValueError('PL layer call counts differ')
        result['graphs'].append(dict(model=row['model'],
            previous_pl_ms=previous['ps_pl_ms'],candidate_pl_ms=row['ps_pl_ms'],
            measured_ps_ms=row['ps_ms'],
            speedup_over_previous_pl=previous['ps_pl_ms']/row['ps_pl_ms'],
            speedup_over_ps=row['ps_ms']/row['ps_pl_ms']))
    result.update(previous_hand_ms=before['hand_ps_pl_ms'],
                  candidate_hand_ms=after['hand_ps_pl_ms'],
                  speedup_over_previous_pl=before['hand_ps_pl_ms']/after['hand_ps_pl_ms'],
                  speedup_over_ps=after['hand_ps_ms']/after['hand_ps_pl_ms'],
                  landmark_error_normalized=after['landmark_error_normalized'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline',type=Path)
    parser.add_argument('candidate',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args = parser.parse_args()
    result = compare(json.loads(args.baseline.read_text()),json.loads(args.candidate.read_text()))
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
