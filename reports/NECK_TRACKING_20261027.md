# Neck Tracking Deployment - 2026-10-27

## Result

Deployed bounded geometric neck acquisition and retained PL pyramidal LK
tracking with PS RANSAC and current-frame validation. This is an incremental
improvement, not acceptance of the complete real-time fingertip/chord system.

## Changes

- `pynq/neck_region.py`: one 448-pixel-wide edge/Hough pass per acquisition job,
  capped at 384 segments, using the installed native geometry backend on board.
  Alternate 30/80 and 25/70 edge thresholds after negative jobs. Preserve border,
  transverse-fret and negative-grid checks.
- `pynq/vision_pc_bridge.py`: remove expensive synchronous full-image fallback
  from default fast acquisition. CPU neural neck fallback is opt-in through
  `FRETLENS_NECK_MODEL_FALLBACK=1`. Retry a completed negative after 150 ms.
  Preserve asynchronous local lattice measurements, delayed-result alignment,
  appearance/spread checks, and clearing/reacquisition after loss. Back off
  unsuccessful whole-neck extent refreshes from 2 to 4 to 8 seconds.
- `web/vision.js`: live board-camera analysis uses the PL bridge; stop redundant
  legacy `auto=1` acquisition requests to the camera server. Manual tracking is
  retained. Reload the page to load the changed JavaScript.
- Regression fixtures use 640x480 and verify sequence-aligned geometry. JPEG
  regressions use the actual PL integer grayscale formula, not BGR2GRAY.

The neck geometry detector runs on PS. LK flow and image preprocessing use PL.
The existing fingertip neural worker remains enabled. The camera configuration,
neural weights, FPGA bitstream, and audio implementation were not replaced.

## Deployment And Recovery

Board runtime: `/home/xilinx/fingertip_candidate_20261026/first_fold/runtime`.
Deployed through `scripts/deploy_neck_realtime.sh` with hash, import, native
backend and service-readiness validation. Only the AV service was restarted;
the camera service remained running. No USB reset or board reboot was needed.

Backup of the two replaced board Python files:
`/home/xilinx/fingertip_candidate_20261026/first_fold/runtime/previous_neck_20211128_134348`.
The board clock is incorrect, hence the 2021 directory name. The deployment
script restores this backup automatically if deployment validation fails.

SHA256:

```text
a108c54e7243c02e345199219364eee0822ec6ae4d593891382274e095a97c85  pynq/neck_region.py
6badbb0130913a4def80c3195a79901eda2b0cc55386dd3b88291fba139297d5  pynq/vision_pc_bridge.py
f4e5f47e9d4b00b6a72692a173b1f7b94666c4cde40eb2723499cb3ca7851d13  web/vision.js
```

Deployment evidence: `diagnostics/console_Probe_20261027_122259.log` contains
`NECK_IMPORT_PASS`, `NECK_DEPLOY_PASS`, and zero command return status.

## Actual Board Replay

Recorded guitar image, synthetic translations of 0 through 22 pixels, four
blank frames, then reacquisition. Frames were precompressed at the stated JPEG
quality and sent as lossless RGBA to exercise deployed PL preprocessing.
These are limited replay measurements, not live-camera detection accuracy or
generalization estimates. Requests were paced by an additional 70 ms.

| Measurement | JPEG 70 (camera quality) | JPEG 95 |
| --- | --- | --- |
| First lock | 2.829 s | 3.172 s |
| Moving frames retaining neck ROI | 12/12 | 12/12 |
| Moving frames with hand region | 12/12 | 12/12 |
| Moving request latency median | 257.5 ms | 235.0 ms |
| Moving request latency p95 | 450.95 ms | 330.05 ms |
| Blank frames with stale neck ROI | 0/4 | 0/4 |
| First lock after blank frames | 344 ms | 500 ms |
| Reacquisition tracked frames | 16/16 | 16/16 |

Relative corner drift below 0.04 px measures synthetic translation consistency;
it is not absolute neck-boundary accuracy. A tracked neck ROI does not establish
six measured strings, absolute frets, finger contact, or chord identity.

Reports:
- `diagnostics/neck_realtime_http_20261027_jpeg70.json`
- `diagnostics/neck_realtime_http_20261027_jpeg95.json`
- `diagnostics/console_Probe_20261027_122137.log` (real integer-gray threshold probe)

## Verification

- Neck-region unittest discovery: 16 passed.
- Bridge tests: 57 passed.
- ROI fretboard tests: 21 passed.
- Browser ROI pipeline and board hand overlay: both passed, no page errors.
- `node --check web/vision.js`: passed.
- Moving browser fixture passed at desktop 1440 and mobile 360 widths; this is
  synthetic fixture verification, not board throughput.
- Native geometry was exercised on the actual board; the Linux shared library
  is unavailable to Windows-only native tests.

Final health check: camera sequence advanced 58711 to 58738 across two seconds;
capture 13.05 fps, zero capture failures, frame age 95.85 ms, not stale. Local
web proxy returned bridge ready, no error, no neural PL fault. The bridge's
analysis was the old replay (177 seconds old), not a fresh live camera result.

## Remaining Limits

Full request processing is approximately four updates/second before pacing,
not 30 fps. Pure neck-flow measurements are typically 37-50 ms, but preprocessing,
feature renewal, hand tracking and competing jobs add latency. Fingertip neural
inference still takes seconds and stale outputs are rejected. Do not present
those points as current-frame fingertip tracking.

The live-camera image inspected during this run did not contain a guitar neck,
so live acquisition and fast physical guitar motion still need validation.
Follow-up performance work should profile PL overlay-validation IPC and neural
transport/kernel cost, retaining hardware-identity and stale-result protections.

Page: http://127.0.0.1:8001/?camera=board
