# FPGA Fingertip Acceleration - 2026-10-27

## Result And Scope

Deployed and verified an incremental acceleration of the existing four-tip
network and current-frame tracking pipeline on the actual Zynq-7020 board.
The learned convolution MACs still execute in PL. The board ARM PS schedules
layers, processes geometry, and fuses visual positions with audio. The PC
provides video transport and display; no PC neural inference was introduced.

This release does **not** achieve 30 neural inferences per second. It does not
establish live guitar fingertip or chord accuracy. The final camera scene had
no guitar, so live NN updates and current projected notes were both zero.

## Deployed Changes

- Native monotonic MMIO waits release the Python GIL between core polls, while
  retaining timeout handling, exclusive ownership and fault quarantine.
- Compact local-socket overlay identity checks avoid repeated large identity
  serialization, and fail closed when identity cannot be verified.
- Vision thread nice=-5, fingertip inference nice=0; audio remains operational.
- Four independent PL pyramidal LK fingertip tracks with forward/backward and
  patch-quality validation. A hand affine transform does not fabricate fingers.
- Bounded retained-frame catch-up and a validated direct catch-up path align
  delayed NN results with current frames. Failed or expired tracks are cleared.
- Native batched patch metrics preserve float32 threshold decisions, including
  exact fallback near decision boundaries.
- Batched four-tip perspective/interpolation reduces ARM projection overhead.
  Current fingertip projection is reused rather than recomputed by the bridge.
- New bridge stage `visual_projection_ms` separates geometry from preview work.
- Color mode requests coordinates only, avoiding diagnostic preview encoding.
- Web overlays use the corresponding analyzed image and stream/generation/
  sequence identity. Old/faulted results are not shown as current evidence.
- Current measured string/fret geometry and explicit calibration map tracked
  tips to MIDI notes. Only current positions participate in audio fusion.

## Measurements

| Measurement | Result | Conditions |
| --- | --- | --- |
| Current camera capture | 7.61 FPS median; 6.79-8.88 range | Live C922, active web analysis, 61.4 seconds |
| Current analysis submissions/results | 5.72 Hz | Same live run; no guitar or active NN |
| Camera without active web analysis | 13.30 FPS median | Separate 60.7-second observation; browser had stopped submitting |
| Four-tip NN call | 1104 ms median | Recorded real image, supplied neck quad, camera disabled, 25-second run |
| Analysis with concurrent NN | 5.41 FPS | Same recorded-image run; not live accuracy |
| Independent PL tip tracking | 30.54 ms median | Moving synthetic textures, synthetic neural seed |
| Note projection/history | 6.57 ms median | Same synthetic moving/calibrated test |
| Audio fusion | 0.410 ms median | Same synthetic test; not measured chord accuracy |

Live observation: 29/29 successful samples, every JPEG decoded, increasing
capture and analysis sequences, fresh analysis, no capture failures, no PL
fault and healthy advancing audio. Capture sequence 13429->13880; analysis
sequence 454->793. Boot ID: `ea135025-f06c-453f-8b57-d5e3ebd0fe98`.

Paired ARM projection benchmarks, exact same numerical results:

| Measured fret boundaries | Before | After | Time reduction |
| --- | --- | --- | --- |
| 5 | 15.55 ms | 4.29 ms | 72.4% |
| 9 | 18.48 ms | 4.48 ms | 75.8% |
| 17 | 23.06 ms | 4.66 ms | 79.8% |
| 25 | 30.21 ms | 5.26 ms | 82.6% |

The 5.41 FPS replay contains 136 accepted frames, 20 completed NN results and
128 independently tracked four-tip frames after initialization. It contains
**zero projected MIDI-note frames** because this fixture did not detect a
usable string/fret lattice. A `current` identity is not proof of a valid note.

The earlier 3481 ms NN run had different camera/background load. Do not use
3481->1104 ms as a controlled speedup claim. The last PS projection change
also did not materially speed up the CNN: the comparable preceding replay
measured 1081 ms NN and 5.31 FPS, versus 1104 ms and 5.41 FPS afterward.

## Accuracy And Functional Limits

Existing validation: 98 images, 356 visible tips, PCK@0.05=72.75%,
PCK@0.10=89.89%, normalized ROI coordinate error. These are point-distance
thresholds, not chord recognition percentages. No new weights were trained.

The actual-PL moving test verifies independent motion, partial occlusion,
delayed-seed alignment, a calibrated string-5/fret-2 -> B2/MIDI47 transition,
and audio support. Its synthetic seed/grid/audio do not establish live model
accuracy. Four image tips cannot determine pressure, all open/muted strings,
barres or the complete sounding voicing. Absolute fret and string orientation
must be calibrated using actual measured wires.

## Hardware Limit And Next Acceleration

Retained hardware: 100 MHz neural/vision clock, DSP212/220, BRAM125/140,
LUT36446. Routed WNS=+0.008 ns; no overclock was applied.

The current graph invokes 21 PL convolutions and fuses 14 transport edges.
Summed measured PL neural transactions remain approximately 395-404 ms,
before remaining graph scheduling, transfers and preprocessing. Therefore
software polling changes alone cannot produce 33 ms full NN updates.

The next substantial improvement requires a separately validated small
four-heatmap student network and calibrated low-bit inference, with layer
fusion/streaming and on-chip reuse to reduce DDR round trips. Keep the current
network as the accuracy reference; first measure same-load per-layer compute
and transfer costs, train/distill and quantize a candidate, then compare full
validation and moving-clip accuracy before replacing the live model. A useful
initial engineering target is <=100 ms NN and <=50 ms tracking, with bounded
queues; those are acceptance targets, not achieved or promised rates.

## Open Source Evidence

Sources inspected for this work and prior kernel experiments:

- FINN MVAU input reuse, folding and SIMD/PE structure:
  https://github.com/Xilinx/finn-hlslib/blob/master/mvau.hpp
- FINN depthwise/vector-vector unit:
  https://github.com/Xilinx/finn-hlslib/blob/master/vvau.hpp
- hls4ml fixed-point and streaming architecture project:
  https://github.com/fastmachinelearning/hls4ml
- AMD official HLS examples, burst transfers and local array partitioning:
  https://github.com/Xilinx/Vitis-HLS-Introductory-Examples

These inform the design; the release is not a FINN/hls4ml-generated DPU.
Installing an upstream library does not accelerate the unchanged graph.

## Verification And Rejected Candidates

- 60 bridge tests pass on host and board, including current-projection reuse,
  historical fallback, identity isolation and invalid-request handling.
- 86 board tests passed for tracking, notes, projection and native metrics.
- Host fingertip regression: 100 passed, one ARM-only skip.
- Web matched-frame tracking: 84 checks; preview: 32 checks; board overlay
  desktop/mobile fixture: passed. Browser fixtures are not live guitar proof.
- ARM/x86 test fixtures now use resolved texture and explicit frame timing;
  the original six-pixel accuracy threshold remains unchanged.
- Serial native helpers had exact identical heatmaps but regressed isolated
  call time 780->854 ms; retained `FRETLENS_NEURAL_SERIAL=0`.
- Sixteen-scanline JPEG experiment was byte-identical but about 0.1% faster;
  not installed, original one-scanline source restored.
- Lowering input from 208 to 160/128 reduced PCK@0.05 to 58.43%/40.17%;
  retained the 208-input model.

## Recovery And Reproduction

Board runtime: `/home/xilinx/fingertip_candidate_20261026/first_fold/runtime`.
Model root: `/home/xilinx/fingertip_candidate_20261026`.
Final bridge rollback directory: `previous_bridge_projection_1638122218`.
Previous PS files: `previous_ps_candidates_1638120523`.
Board clock is in 2021; backup timestamps do not denote the PC work date.

Camera is currently C922, YUYV640x480, native-fast JPEG, quality70. Existing
runtime autosuspend prevention is retained. An ordinary authorized full-board
software reboot restored capture; recovery also included unexpected repeated
boots recorded in `final_boot_monitor.log`. UVC -110 is initialization timeout,
not proof of a power fault. Do not repeat speculative controller/PHY resets or
temporary USB device-tree overlays on a working camera. No permanent SD/boot
change was applied. Future recovery is not guaranteed by this observation.

For the same startup stall, preserve status/kernel evidence, perform the
authorized normal reboot through the established UART console, and verify a
new boot ID, YUYV mode, fresh advancing images and zero errors at both board
and local proxy. Camera health and NN/PL health require separate checks.

Saved release under `reports/neural_accel_20261027/` includes source/tests,
benchmarks, live evidence, service units, the actual board runtime, model
weights, bitstream/HWH and camera dependencies. Inspect archive entries before
restoration; it is not an SD image or a cross-board installer. Restoring a
whole runtime needs controlled service stop/start and identity verification.

Bit SHA256: `b17829b68d04ebdb9d7ee3ba2dce78952cf2b73cd3e9c13f3d175a74f4509b4f`.
HWH SHA256: `a34fa04331ef2884036dd73f2640e048ef72d882186d2da20e0debf98fa0d64a`.
Board archive SHA256: `781f4336847e838bad1101e887d882aedc37d457958ba742218578ba80216c37`.
All ten changed runtime Python files matched the retrieved board checksums.

Live web: http://127.0.0.1:8001/?camera=board
