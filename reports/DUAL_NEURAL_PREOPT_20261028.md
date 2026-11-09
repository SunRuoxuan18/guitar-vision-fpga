# Offline fingertip pre-optimization and neck YOLO PL feasibility

Date: 2026-10-28. Target: PYNQ-Z2, XC7Z020CLG400-1.

## Decision

There is measured synthesis evidence for a smaller DSP footprint while
preserving 16-bit fingertip activations. The recommended next integration
candidate is one W8A16 convolution core shared by both models, with regular
spatial multiplications mapped to fabric. Resource arithmetic fits the
device. Full graph execution, integration, routing and board latency remain
unverified; this is not a dual-model deployment result.

Two copies of the tested W8A8 core exceed the available DSP, BRAM and LUT
budgets. This rejects those particular two-core configurations, not every
possible independently optimized pair of accelerators.

## Accuracy evidence

The existing released 208x208 fingertip model was evaluated on the same
98 take03 validation frames, containing 356 visible fingertip points. No
training data, architecture, labels or production weights were changed.

| Numerical candidate | PCK@0.05 | PCK@0.10 |
| --- | ---: | ---: |
| Released float / Q16 numerical reference | 72.75% | 89.89% |
| W8A16 fake quantization | 71.35% | 89.04% |
| W8A8 fake quantization | 67.70% | 87.36% |

PCK uses visible points only and counts a point as correct when
sqrt((dx / ROI_width)^2 + (dy / ROI_height)^2) is at most the threshold.
It is a distance-based fingertip metric, not chord recognition accuracy.
This held-out recording is from the same person; generalization to other
people, camera setups or sessions is not established.

W8A16 uses 8-bit per-output-channel weights and 16-bit dynamic per-tensor
Conv inputs. W8A8 also reduces those inputs to 8 bits. Both probes dequantize
and run FP32 convolutions; they are not bit-exact integer hardware network
evaluations. The Q16 reference uses exact integer-valued FP64 sums followed
by the current FP32 scale/bias operation. Native PS rounding is not fully
emulated. Small convolution RTL checks and network accuracy checks provide
different evidence and must not be conflated.

W8A16 loses 1.40 percentage points at the stricter threshold; W8A8 loses
5.06 points. W8A16 is therefore the preferred accuracy-preserving starting
point. Quantization alone does not improve the existing model's accuracy.
Raw convolution weight storage halves from 105,664 Q16 bytes to 52,832 W8
bytes, excluding scales, padding and bias; this does not imply BRAM halves.

## Resource evidence

The existing routed full design uses 212 DSP, 125 BRAM36 equivalents and
36,446 LUT. Its hand_conv_q16 core accounts for 141 DSP, 49 BRAM36 and 9,942
LUT. Subtracting that hierarchy leaves 71 DSP, 76 BRAM36 and 26,504 LUT for
the other existing logic. Device capacity is 220 DSP, 140 BRAM36 and 53,200
LUT. BRAM36 equivalent means RAMB36 + RAMB18 / 2.

New-core values below come from Vivado 2026.1 out-of-context RTL synthesis,
not HLS estimates and not placed-and-routed implementation:

| Core | DSP | BRAM36 | LUT | Strict fingertip PCK probe |
| --- | ---: | ---: | ---: | ---: |
| Existing routed Q16, comparison | 141 | 49 | 9,942 | 72.75% Q16 reference |
| q8_v2, W8A8 | 82 | 39 | 15,150 | 67.70% fake |
| w8a16_v2, default mapping | 154 | 49 | 11,806 | 71.35% fake |
| w8a16_spatial_v2, spatial products in fabric | 82 | 49 | 20,422 | 71.35% fake |

The final variant keeps the same W8A16 arithmetic while changing the
multiplier implementation. Relative to the current routed hand core, it
uses 59 fewer DSPs, unchanged BRAM, and 10,480 more LUTs. Precision reduction
and mapping changes do not guarantee an identical saving after integration.

| Replacement scenario, including other fixed logic | DSP / 220 | BRAM36 / 140 | LUT / 53,200 |
| --- | ---: | ---: | ---: |
| One q8_v2 shared core | 153 | 115 | 41,654 |
| One w8a16_v2 shared core | 225 | 125 | 38,310 |
| One w8a16_spatial_v2 shared core | 153 | 125 | 46,926 |
| Two q8_v2 cores | 235 | 154 | 56,804 |
| Two w8a16_spatial_v2 cores | 235 | 174 | 67,348 |

The selected shared variant leaves 67 DSP, 15 BRAM36 and 6,274 LUT in this
arithmetic estimate. LUT occupancy is about 88.2%; integration and routing
headroom need care. Extra interconnect, graph operators and scheduling
hardware are not included. Default W8A16 mapping exceeds DSP capacity by 5.
Adding a second unmodified accelerator to the current design is not viable
from this budget: the existing full design leaves only 8 DSP and 15 BRAM36.

## Model coverage and performance limits

| Source graph | Conv layers | Conv MACs per inference | Conv weight parameters |
| --- | ---: | ---: | ---: |
| Fingertip, 208x208 | 21 | 98,241,728 | 52,832 |
| Neck YOLO, 320x320 | 64 | 1,010,476,800 | 3,000,640 |

YOLO has about 10.3 times the convolution work. Its W8 Conv weights alone
would be 3,000,640 bytes, too large to reside entirely in on-chip BRAM; the
proposed tiled engine therefore depends on external DDR and data reuse.
Reducing fingertip weights alone does not make an entire second network
resident on-chip. A shared accelerator loads each model's tiles and
weights from DDR without requiring bitstream reconfiguration between them.

The probe removes the current input-size restriction for YOLO's 320x320
first layer and its DFL Conv [1,16,4,2100] -> [1,1,4,2100]. This covers
convolution geometry only. The existing production compiler does not
compile the full YOLO graph. SiLU/Sigmoid/Mul, pooling, concat, resize,
shape/head operations, decode and NMS still require a supported graph
execution path. Small operations may remain on PS; CPU/PL transfers and
requantization must be measured to ensure they do not consume the saving.

At an optimistic 100 MHz and 64 useful MAC/cycle, convolution-only arithmetic
lower bounds are 15.35 ms for fingertip and 157.89 ms for YOLO, or 173.24 ms
for the serial pair. These are not latency predictions. They ignore
underutilization, depthwise efficiency, regular spatial path parallelism,
DDR, tile loading, output transfers, PS operators and launch overhead.
Actual execution will require a measured model-level schedule. The present
core configuration does not support a claim of both models running at
30 full inferences per second.

The deployment direction is PL convolution for both models, with fingertip
work prioritized and YOLO used for initial acquisition and periodic
reacquisition; a cheaper tracker can update neck geometry between YOLO
inferences. This is a scheduling recommendation, not an implemented
real-time performance result. PC remains a display/transport client.

## Verification

- q16_v2, q8_v2, w8a16_v2 and w8a16_spatial_v2 each passed 13 C simulation
  cases, including signed values, channel tails, asymmetric padding,
  striding, depthwise kernels, maximum channels, long accumulation,
  320x320 input, 4x2100 DFL input and output guard words.
- q8_v2, w8a16_v2 and w8a16_spatial_v2 passed Verilog RTL cosimulation for
  five small cases each. Large shapes were checked in C simulation only;
  their RTL tests need sufficient AXI simulation depth beyond 4096.
- Those three variants completed isolated RTL synthesis and utilization
  reporting. The 100 MHz OOC internal timing estimate passes, but external
  I/O delays and integrated placement/routing are not validated. The final
  variant's HLS period estimate is 9.275 ns with 2.7 ns clock uncertainty,
  so full timing closure must not be inferred from HLS frequency alone.
- Python regression results: 6 quantization checks, 5 graph/resource checks
  and 2 report checks passed. A separate read-only code review found no
  confirmed material bug in the scoped numerical/budget logic.
- No board access, camera restart, runtime replacement or production
  bitstream/weight deployment was performed for this offline experiment.
  Production kernel SHA256 remains
  c14ba2d73599c0b35f3dc111fda8f2efdaee0b8cf453c4f6a5c16599368afe77.

## Reproduction and evidence

Run from D:/FPGA/ZynqAudioLab:

```powershell
D:\FPGA\hand_fingertip_env\Scripts\python.exe -B vision/preopt_fingertip_quantization.py --threads 2
vision\.venv\Scripts\python.exe -B scripts/profile_dual_neural_preopt.py
vision\.venv\Scripts\python.exe -B scripts/summarize_neural_preopt.py
```

For a new HLS build, use an unused revision name: completed builds are
preserved. Example with v3 (the recorded results above are v2):

```powershell
$env:FRETLENS_PREOPT_VARIANT = 'w8a16_spatial'
$env:FRETLENS_PREOPT_REV = 'v3'
Remove-Item Env:FRETLENS_PREOPT_COSIM -ErrorAction SilentlyContinue
& 'D:\AMD\2026.1\Vitis\bin\vitis-run.bat' --mode hls --tcl hls/hand_conv_preopt/run_probe.tcl
$env:FRETLENS_PREOPT_COSIM = '1'
& 'D:\AMD\2026.1\Vitis\bin\vitis-run.bat' --mode hls --tcl hls/hand_conv_preopt/run_probe.tcl
& 'D:\AMD\2026.1\Vivado\bin\vivado.bat' -mode batch -source hls/hand_conv_preopt/synth_ooc.tcl -tclargs w8a16_spatial_v3
```

Key saved artifacts:

- diagnostics/fingertip_preopt_20261028/report.json: metrics, numerical
  method, input hashes and prediction drift.
- diagnostics/fingertip_preopt_20261028/candidate_w8_folded_weights.npz:
  research-only W8 export with scales and FP32 bias; not a production graph.
- diagnostics/dual_neural_preopt_20261028/profile.json: standard ONNX Conv
  shapes, operation accounting, source hashes and existing resource report.
- diagnostics/hand_conv_preopt_20261028/summary.json: combined evidence,
  source report hashes and capacity arithmetic for each variant.
- diagnostics/hand_conv_preopt_20261028/w8a16_spatial_v2/: Csim, csynth,
  cosim, isolated utilization/timing reports and synthesized checkpoint.
- hls/hand_conv_preopt/: isolated candidate source and testbenches.

## Remaining deployment work

1. Implement and verify W8 weight packing, W8A16 scaling and driver support
   against exact integer network outputs, preserving production fallback.
2. Add a supported YOLO graph execution path and calibrate/measure its
   quantization accuracy on neck images. Fingertip PCK says nothing about
   neck detector accuracy after quantization.
3. Integrate the shared accelerator and scheduling into a new full design,
   verify large-shape RTL memory transactions, then place/route and inspect
   actual resource and timing margins.
4. Run both models on the board and measure end-to-end latency, tracking
   continuity and chord-level outcomes. Only those measurements establish
   usable real-time performance.
