"""Combine offline accuracy and synthesized resource evidence without a fit claim."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
PROBES = ROOT / "diagnostics/hand_conv_preopt_20261028"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ooc_resources(path):
    labels = {"Slice LUTs*": "lut", "DSPs": "dsp", "Block RAM Tile": "bram36_equivalent"}
    result = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        cells = [cell.strip() for cell in line.split("|")]
        if len(cells) > 3 and cells[1] in labels:
            result[labels[cells[1]]] = float(cells[2])
    if set(result) != set(labels.values()):
        raise ValueError("Incomplete synthesized utilization report: " + str(path))
    return result


def hls_resources(path):
    root = ET.parse(path).getroot()
    resources = root.find("AreaEstimates/Resources")
    if resources is None:
        raise ValueError("Missing HLS estimates")
    return {
        "lut": int(resources.findtext("LUT")), "dsp": int(resources.findtext("DSP")),
        "bram18_estimate": int(resources.findtext("BRAM_18K")),
        "estimated_period_ns": float(root.findtext("PerformanceEstimates/SummaryOfTimingAnalysis/EstimatedClockPeriod")),
        "clock_uncertainty_ns": float(root.findtext("UserAssignments/ClockUncertainty")),
        "target_period_ns": float(root.findtext("UserAssignments/TargetClockPeriod")),
    }


def resource_budget(fixed, candidate, capacity, copies=1):
    if copies < 1:
        raise ValueError("Positive candidate count required")
    total = {key: fixed[key] + copies * candidate[key] for key in capacity}
    remaining = {key: capacity[key] - total[key] for key in capacity}
    return {"core_copies": copies, "total": total, "remaining": remaining,
            "arithmetic_within_capacity": all(value >= 0 for value in remaining.values()),
            "scope": "Resource arithmetic only; combines existing routed fixed logic and isolated synthesis; integration, AXI, placement and routing unverified"}


def summarize(profile, accuracy, variants):
    hardware = profile["resources"]
    output = {
        "scope": "Offline pre-optimization; production bitstream/runtime/weights not changed; no board speed or dual-model deployment claim",
        "model_cost": {key: {"convolutions": value["conv_count"], "conv_macs": value["conv_macs"],
                             "conv_weight_parameters": value["conv_weight_parameters"]}
                       for key, value in profile["models"].items()},
        "existing_resource_evidence": hardware,
        "accuracy_probe": {"sample_count": accuracy["sample_count"], "visible_points": accuracy["visible_points"],
                           "results": accuracy["results"], "scope": "Offline fake quantization, not a deployed bit-exact network; same-person held-out recording, not chord accuracy"},
        "variants": {},
    }
    for variant in variants:
        build = PROBES / variant
        xml_path = build / "solution1/syn/report/hand_conv_preopt_csynth.xml"
        csim_path = build / "solution1/csim/report/hand_conv_preopt_csim.log"
        row = {"hls": hls_resources(xml_path), "hls_report": str(xml_path), "hls_sha256": digest(xml_path)}
        csim = csim_path.read_text(encoding="utf-8")
        row["csim"] = {"passed_cases": len(re.findall(r"^PASS ", csim, re.MULTILINE)),
                       "zero_errors": "CSim done with 0 errors" in csim, "path": str(csim_path)}
        cosim_path = build / "solution1/sim/report/hand_conv_preopt_cosim.rpt"
        if cosim_path.exists():
            cosim = cosim_path.read_text(encoding="utf-8")
            row["cosim"] = {"path": str(cosim_path), "verilog_pass": bool(re.search(r"Verilog\s*\|\s*Pass", cosim)),
                            "scope": "Five small integer convolution regressions; not full-model accuracy/latency"}
        ooc_path = build / "ooc/utilization.rpt"
        if ooc_path.exists():
            candidate = ooc_resources(ooc_path)
            row["ooc"] = {"resources": candidate, "path": str(ooc_path), "sha256": digest(ooc_path), "stage": "Synthesized, not placed/routed"}
            row["shared_core_budget"] = resource_budget(hardware["other_fixed"], candidate, hardware["device_capacity_xc7z020"])
            row["two_identical_cores_budget"] = resource_budget(hardware["other_fixed"], candidate, hardware["device_capacity_xc7z020"], copies=2)
            row["delta_vs_existing_routed_core"] = {key: candidate[key] - hardware["existing_core"][key] for key in candidate}
        output["variants"][variant] = row
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", default=["q16_v2", "q8_v2", "w8a16_v2", "w8a16_spatial_v2"])
    parser.add_argument("--output", type=Path, default=PROBES / "summary.json")
    args = parser.parse_args()
    if any(not re.fullmatch(r"[a-zA-Z0-9_]+", variant) for variant in args.variants):
        parser.error("Invalid variant name")
    profile = json.loads((ROOT / "diagnostics/dual_neural_preopt_20261028/profile.json").read_text(encoding="utf-8"))
    accuracy = json.loads((ROOT / "diagnostics/fingertip_preopt_20261028/report.json").read_text(encoding="utf-8"))
    result = summarize(profile, accuracy, args.variants)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "variants": {name: {"hls": row["hls"], "ooc": row.get("ooc"), "shared_core_budget": row.get("shared_core_budget")}
                      for name, row in result["variants"].items()}}, indent=2))


if __name__ == "__main__":
    main()
