# ROI Projection HLS

`roi_projection` produces one packed `uint32` word for every 320 x 128 destination pixel.

## Stable Integration Artifact

The current packaged IP is `build/solution1/impl/ip/component.xml` and
`build/solution1/impl/ip/xilinx_com_hls_roi_projection_1_0.zip`. Its source is
`roi_projection.cpp`, the 16-row tagged DDR cache implementation. The original
direct-DDR package was preserved before this export as
`direct_ddr_baseline_xilinx_com_hls_roi_projection_1_0.zip`.

## ABI

Top function: `roi_projection`

| AXI-Lite offset | Argument |
| --- | --- |
| `0x10` | `image`, `uint8[640 * 480]` |
| `0x1c` | `inverse_homography`, `float[9]`, destination-to-source |
| `0x28` | `packed`, `uint32[320 * 128]` |

Each `packed[y * 320 + x]` is:

| Bits | Value |
| --- | --- |
| `[7:0]` | raw bilinear rectified `uint8` |
| `[18:8]` | `abs(gx)` from Sobel on rounded Gaussian output |
| `[29:19]` | `abs(gy)` from Sobel on rounded Gaussian output |
| `[31:30]` | zero |

Warp interpolation has a constant-zero border. Gaussian 3x3 uses weights
`[1 2 1; 2 4 2; 1 2 1] / 16`, rounded to `uint8`; Gaussian and Sobel use
`BORDER_REFLECT_101`. A zero projective denominator emits a raw rectified zero.
Its downstream Gaussian/Sobel values can be nonzero when neighboring pixels
contribute through the reflect101 filter border.

## Verification

`test_roi_projection.cpp` independently computes the warp, rounded Gaussian,
and Sobel packing. AMD 2026.1 C simulation passed identity, fractional
translation, and projective homographies.

## Measured Candidates

| Candidate | BRAM18 | DSP | LUT | HLS minimum latency at 100 MHz | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Direct DDR (preserved ZIP) | 70 | 17 | 11.8k | not a performance candidate | fits memory |
| 16-row tagged cache (active IP) | 78 | 18 | 11,419 | 36.88 ms | stable first-bit only |
| Pipelined cache experiment (`build_fast`, rejected) | 150 | 22 | 25,914 | 48 ms | reject: exceeds combined memory and is slower |

The combined bitstream has been routed, downloaded and tested on PYNQ-Z2.
For the standalone projective and full-frame fixtures, the PL call takes
64.7-69.7 ms including copies, versus 18.6-18.8 ms for PS OpenCV. Rectified
pixel mean absolute error is approximately 0.029 intensity units and directional
gradient error approximately 0.19-0.20. Evidence is recorded in
`diagnostics/vision_hls_microbench.json` at the project root.

Production therefore keeps the small-ROI warp on PS and enables PL pyramidal
LK instead. The ROI IP remains in the bitstream for the explicitly selected
`--hls-roi` comparison, not as a claimed speed improvement. DDR sampling and
cache misses make this implementation slower despite its HLS pipeline.
