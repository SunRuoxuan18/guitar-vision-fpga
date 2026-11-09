# Sparse LK HLS experiment

This isolated accelerator is a first hardware stage for neck tracking, not a
replacement for the production pyramidal tracker. It performs one linearized
Lucas-Kanade solve for up to 64 integer feature coordinates in two grayscale
frames (maximum 320x200). A 9x9 patch supplies symmetric spatial derivatives
for a 7x7 normal-equation window. Texture/aperture, boundary and motion guards
reject invalid outputs. Good results on small synthetic translations do not
establish accuracy on a moving, occluded guitar neck.

## Reference and adaptation

Reference supplied by the user:
https://gitee.com/Vickyiii/fpgachina25-amd/tree/master/reference/LucasKanade

The reference report and source demonstrate local LK sums, bounded integer
types, window storage, array partitioning and pipelining. This implementation
is independently written: sparse feature patches instead of dense whole-frame
flow, memory-mapped AXI instead of the reference FIFO top-level interface, and
integer division instead of its floating-point solver. No reference performance
claim is reused. There is no claim of affiliation or official template status.

## Build and verify

PowerShell from this folder (AMD 2026.1 installed):

```powershell
$env:LK_HLS_COSIM = '1'
& D:/AMD/2026.1/Vitis/bin/vitis-run.bat --mode hls --tcl run_hls.tcl
& D:/AMD/2026.1/Vivado/bin/vivado.bat -mode batch -source D:/FPGA/ZynqAudioLab/scripts/build_lk_hls_overlay.tcl
```

The testbench derives images from an analytic texture at known translated
coordinates, rather than duplicating the hardware arithmetic as a golden model.
It checks static, signed subpixel and one-pixel translation, flat images,
aperture ambiguity and out-of-window coordinates. HLS uses a 10ns clock
constraint for xc7z020clg400-1. Synthesized timing is an estimate; only routed
timing and physical end-to-end measurements establish implemented performance.

The build produces an experimental standalone `output/lk_hls/lk_hls.bit` and
matching `.hwh`. It does not include audio or the existing Sobel overlay.
Downloading it replaces those PL functions. It never initializes PS or DDR.

## ABI

AXI-Lite registers: control at 0x00; 64-bit addresses of previous frame, current
frame, points and results at 0x10, 0x1c, 0x28 and 0x34; unsigned width, height
and count at 0x40, 0x48 and 0x50. Frames are contiguous row-major uint8.
Each point is uint32 with x in bits 15:0 and y in 31:16.
Each result is uint64: signed Q6 dx in 15:0, signed Q6 dy in 31:16, valid in
bit 32; remaining bits are zero. Q6 represents units of 1/64 pixel, not an
accuracy guarantee. Valid motion is limited to +/-4 pixels per component;
one-step LK is only validated here for shifts up to one pixel.

PS driver: `pynq/lk_hls.py`. Board experiment:
`sudo env XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 diagnostics/benchmark_lk_hls.py --download`.
The driver requires explicit download, validates dimensions/coordinates, reuses
contiguous buffers, flushes/invalidate caches and bounds hardware waits. A
timeout quarantines outstanding memory until the process exits; keep the
process alive until the overlay has been replaced/reset if an AXI operation
remains outstanding. Never retry a faulted driver in place.

## Integration gate

Production tracking retains pyramids, forward/backward checking, RANSAC,
spatial coverage and appearance checks. This single-scale kernel does not
implement those. Its benchmark compares against OpenCV iterative single-scale
LK and explicitly reports that these algorithms differ. It includes copying,
cache maintenance, register writes, polling and output decoding in total time.
It is not automatically enabled in the web camera service. Integrate only
after realistic motion/occlusion testing and a measured end-to-end benefit.

## Measured result (2026-10-20)

Clean C simulation and RTL co-simulation passed; the repaired 100MHz overlay
also passed on PYNQ-Z2: all 48 points accepted on seven known translations,
maximum error 0.09111 pixel, blank/border rejected. Routed usage: 6563 LUT,
9409 registers, 21 DSP, two BRAM36 tiles; setup slack +0.817ns and hold
slack +0.021ns, zero DRC errors.

Complete calls measured 30.43..30.81ms; OpenCV iterative single-scale LK
measured 3.86..4.10ms. Kernel dispatch-to-done measured 0.46..0.47ms;
PYNQ overlay identity checks consumed approximately 27ms per call. These
checks use PL-server IPC, not just local properties. Copying takes about
1ms. The current driver is therefore slower overall, despite a short
hardware computation. It remains experimental, disabled in production.
Evidence: `diagnostics/lk_hls_board.json` and `output/lk_hls/*.rpt`.

## PS driver follow-up

The per-call guard now queries the programming timestamp once, instead of
calling both `Overlay.is_loaded()` and `device.timestamp`. Both previously
queried the same PL-server metadata. A nonempty initial timestamp is required;
changed overlays are still rejected before MMIO. Five host tests pass,
including changed-overlay rejection and AXI timeout memory retention.

A subsequent isolated metadata-only test on ARM measured 112.89ms for the
old two-query guard and 56.34ms for the one-query guard (10 measured samples
after two warmups). These timings were obtained after a board reboot under
different conditions from the earlier 27ms result; do not combine them into
an estimated complete LK time. Full updated LK measurement did not complete:
the board rebooted during the trial, with cause unconfirmed. Base overlay
was confirmed active afterward; camera reopen returned UVC -110. No updated
end-to-end acceleration or production integration is claimed.
