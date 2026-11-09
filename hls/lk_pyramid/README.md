# Pyramidal sparse Lucas-Kanade HLS kernel

`lk_pyramid` tracks at most 64 subpixel feature positions through two 8-bit
grayscale frames, up to 320x200. It stores both complete four-level image
pyramids on chip (320x200, 160x100, 80x50, 40x25) with 5x5 Gaussian reduction,
uses a 21x21 patch, reference
image inverse-compositional gradients, bilinear sampling, and at most ten
iterations at each coarse-to-fine level.

## Locked ABI

The top function is:

```cpp
void lk_pyramid(const ap_uint<8>* previous, const ap_uint<8>* current,
                const float* points, float* result,
                unsigned width, unsigned height, unsigned count);
```

AXI-Lite `control` offsets are fixed: `previous` 0x10, `current` 0x1c,
`points` 0x28, `result` 0x34, `width` 0x40, `height` 0x48, and `count` 0x50.
The pointer ports are separate AXI master bundles. `previous` and `current`
are contiguous `uint8` row-major frames. `points` is `float[count * 2]`
containing `[x, y]` pairs. `result` is `float[count * 4]` containing
`[tracked_x, tracked_y, valid, mean_abs_residual]`; valid is 0.0 or 1.0.

The C testbench includes an independent CPU float LK/pyramid reference using
the same integer-rounded Gaussian pyramid, 4-level/21x21/10-iteration parameters. It checks subpixel points and
known 1, 4, 8 pixel, and fractional translations with no output error above
0.7 pixels, as well as blank and boundary rejection.

The current Csim accepts 316 of 320 textured synthetic tracks across eight
known translations (1, 4, 8, and fractional pixels); the other four are
explicit safe-boundary/conditioning rejections. Every accepted track is within
0.7 pixel of its known translation and agrees with the standalone CPU reference
within 0.001 pixel.

`cpu_lk_pyramid.cpp` and `cpu_lk_pyramid.h` are the plain-C++ exact algorithm
reference for ARM comparison. They contain no HLS/ap types and export:

```cpp
extern "C" void cpu_lk_pyramid(const uint8_t*, const uint8_t*, const float*,
                                float*, unsigned, unsigned, unsigned);
```

Build it on the target with `g++ -O3 -std=c++14 -fPIC -shared
cpu_lk_pyramid.cpp -o libcpu_lk_pyramid.so`. It uses the identical buffer ABI,
integer-rounded Gaussian pyramid, per-point safe starting level, patch, iteration cap,
acceptance rule, and result fields as the HLS C model.

## Build

```powershell
& D:/AMD/2026.1/Vitis/bin/vitis-run.bat --mode hls --tcl run_hls.tcl
```

Artifacts are written below `build/solution1`, including the exported IP at
`build/solution1/impl/ip`.

## Final HLS result

The optimized exported IP is in `build_optimized/solution1/impl/ip` (archive:
`build_optimized/solution1/impl/export.zip`). AMD Vitis HLS 2026.1, targeting
`xc7z020clg400-1` at 10 ns, reports 9.162 ns (109.15 MHz), 145 BRAM18K,
41 DSP, and 31,809 LUT. The 145 BRAM18K estimate plus the current ROI
candidate's 78 BRAM18K is 223 of the device's 280 BRAM18K blocks.

The optimized HLS datapath caches gradients, converts bilinear sample phase
once per level/iteration, and uses four-lane reductions at II=4. The CPU
reference is intentionally unchanged by this scheduling optimization. ARM
production validation of the exact CPU implementation passed 11/14 sequences,
matching the baseline; reported forward/backward medians were 86.52/88.13 ms.

## Routed Hardware And Board Measurements

The combined design is implemented at 100 MHz and deployed on PYNQ-Z2.
`output/vision_flow/vision_pl.bit` includes the existing gray/Sobel datapath,
this LK core and the optional ROI core. Final setup/hold slack is
+0.005/+0.015 ns, with no unrouted nets. Whole-design usage is 27,896 LUT,
33,499 FF, 98 BRAM36 equivalents and 74 DSP; these routed numbers supersede
the HLS resource estimates above.

The standalone 48-point test takes 14.23-18.83 ms per PL call including copies,
versus 21.12-35.86 ms for PS OpenCV and 53.82-64.56 ms for the identical
algorithm's native CPU reference. The maximum accepted known-motion error
is 0.0123 pixels on these bandlimited translation fixtures. PL/reference
coordinates and validity agree under the microbenchmark tolerance.

The same-bitstream, 14-frame application comparison measures forward/backward
medians of 25.30/25.45 ms on PL versus 37.03/37.85 ms for PS OpenCV. Both keep
13 tracking frames after the initial acquisition frame. Whole board request
processing is only 366.19 versus 373.99 ms, and HTTP roundtrip is 700.89 versus
688.74 ms: this is a local compute improvement, not demonstrated end-to-end
latency improvement. PS retains the small ROI warp because the measured ROI
hardware is slower. See `diagnostics/vision_acceleration_comparison.json` and
`diagnostics/vision_hls_microbench.json` for raw measurements and scope.

## Reverse Pyramid Cache Candidate

`run_hls_cached.tcl` builds an isolated candidate in `build_cached`. The
optional AXI-Lite register `reuse_reverse` at 0x58 swaps the retained pyramid
pair for one reverse call, avoiding duplicate image transfers and pyramid
generation. The cache is consumed after that call. A missing cache or changed
dimensions yields invalid points instead of stale coordinates. The Python
driver uses this register only when the loaded HWH advertises it and the same
image objects are supplied in reverse order.

C simulation compares cached and freshly generated reverse results, including
residuals and validity, and rejects a second cache consumption. HLS estimates
9.162 ns, 145 BRAM18K, 41 DSP and 32,809 LUT. These are synthesis estimates,
not a routed timing result or a measured application speedup.

The application also supports `FRETLENS_FLOW_POINTS=32` (default remains 48
unless configured by the service). On the original combined bitstream the
32-point fixture test retained 13/14 tracking frames and measured 195.19 ms
board processing / 254.89 ms HTTP median; forward plus reverse flow was about
39 ms instead of 51 ms. This fixture does not validate real guitar occlusion.

Network ROI transport is experimental (`?roiTransport=1` in the browser).
The current PL ABI still processes full frames; CPU reconstruction offset
the reduced network payload in measurement (204.57 / 257.32 ms). It is not
enabled by default and is not described as hardware ROI acceleration.

The cache candidate is now routed and board-validated. Artifact:
`output/vision_cached/vision_pl.bit`, SHA256
`4cb150b62c83e12a27722216bd382ac1e82b80bfc4f4cefec4d12d9817b3de0d`.
At 100 MHz setup/hold slack is +0.181/+0.012 ns, DRC has no errors,
and routed usage is 28,550 LUT, 33,922 FF, 98 BRAM36 equivalents and 74 DSP.
Parallel synthesis initially exhausted host RAM; the single-thread retry
and subsequent implementation completed successfully.

On-board identical-input reverse calls measure 12.92 ms fresh versus 8.24 ms
cached (1.57x); coordinates, validity and residuals agree within 0.001.
See `diagnostics/vision_cache_microbench.json`. The complete sequence still
retains 13/14 tracking frames with identical string-line errors, but its
217.36 ms processing / 273.26 ms HTTP medians do not improve on the prior
32-point run's 195.19 / 254.89 ms. Refresh scheduling and remaining PS work
dominate: this is a demonstrated core speedup, not a demonstrated overall
speedup or 30 tracking updates per second.
