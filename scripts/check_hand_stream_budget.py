"""Fail closed before adding the stream core to the deployed Zynq-7020 design."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def inspect(report):
    root = ET.parse(report).getroot()
    resources = root.find('AreaEstimates/Resources')
    if resources is None:
        raise ValueError('Missing HLS resource estimates')
    used = {key: int(resources.findtext(key)) for key in ('BRAM_18K', 'DSP', 'LUT')}
    # Existing routed design: 125 BRAM36, 212 DSP, 36061 LUT.
    # HLS BRAM18 estimates can pack imperfectly; this is only a rejection gate.
    budgets = dict(BRAM_18K=30, DSP=8, LUT=17139)
    period = float(root.findtext('PerformanceEstimates/SummaryOfTimingAnalysis/EstimatedClockPeriod'))
    uncertainty = float(root.findtext('UserAssignments/ClockUncertainty'))
    exceeded = {key: dict(estimated=used[key], remaining=budget)
                for key, budget in budgets.items() if used[key] > budget}
    return dict(resources=used, remaining_budget=budgets,
                estimated_period_ns=period, target_period_ns=10.,
                clock_uncertainty_ns=uncertainty,
                hls_uncertainty_budget_met=period <= 10.-uncertainty,
                exceeded=exceeded, preliminary_fit=not exceeded and period <= 10.,
                note='Passing is not routed timing/resource closure or permission to deploy.')


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=repo/'hls/hand_dw_pw_stream/build/solution1/syn/report/hand_dw_pw_stream_csynth.xml')
    parser.add_argument('--output', type=Path, default=repo/'diagnostics/hand_stream_budget.json')
    args = parser.parse_args()
    result = inspect(args.report)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
    return 0 if result['preliminary_fit'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
