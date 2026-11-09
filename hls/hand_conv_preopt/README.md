# Offline shared-convolution pre-optimization probe

This isolated copy of `../hand_conv_q16/hand_conv_q16.cpp` evaluates resource
tradeoffs. The source snapshot SHA256 is
`c14ba2d73599c0b35f3dc111fda8f2efdaee0b8cf453c4f6a5c16599368afe77`.
It does not replace the production IP, driver, weights or bitstream.

The q16 reference, q8 candidate and q8_fabric candidate share the same
parallelism and tiling. Q8 changes signed operands from 16 to 8 bits, packed
words from 128 to 64 bits, and accumulators from 48 to 32 bits. The largest
supported reduction is 1024*5*5 terms: 8-bit products cannot overflow the signed
32-bit accumulator. The q8_fabric variant additionally binds multiplications
to fabric to measure the LUT/DSP tradeoff; it is not assumed to be smaller or
faster before synthesis.

All variants allow 320x320 input and a bounded 4x4096 pointwise head, covering
the current YOLO first layer and 4x2100 DFL convolution. C simulation checks
these shapes as well as grouped/depthwise, channel tails, signed values,
strided/asymmetric boundaries, accumulation range, and output guard words.

Run `vitis-run --mode hls --tcl run_probe.tcl` from this directory after setting
`FRETLENS_PREOPT_VARIANT` to `q16`, `q8`, or `q8_fabric`. Results go to
`diagnostics/hand_conv_preopt_20261028/<variant>`. A completed build is not
overwritten. Set `FRETLENS_PREOPT_COSIM=1` to cosimulate an existing build with
five small regression cases.

`w8a16` preserves 16-bit activations but uses signed 8-bit weights, a separate
64-bit weight port and a 40-bit accumulator. It is an alternative to W8A8
when the latter loses too much accuracy. `FRETLENS_PREOPT_REV=v2` selects
versioned build directories; v2 fixes the DFL shape to 4x2100. Earlier probes
checked only 1x2100 and are superseded.

`w8a16_spatial` additionally binds the regular spatial convolution products
to fabric. Pointwise products keep their default DSP mapping. This trades
LUTs for DSPs without changing operand or accumulator precision. Select it
with `FRETLENS_PREOPT_VARIANT=w8a16_spatial` and the v2 revision. Run the
build and cosimulation sequentially; cosimulation reopens that same build.

For a completed probe, run Vivado with `-source synth_ooc.tcl -tclargs <name>`
to synthesize its RTL in isolation. This improves the resource estimate but
does not perform placement/routing or integrate audio and tracking.

Q8 requires a new packer/driver. These are integer convolution probes; bias,
scaling, nonlinearities, graph scheduling and decoding are not implemented in
this core. HLS estimates are not full-design placed-and-routed acceptance or
end-to-end speed measurements. Fake-quantized model metrics are separate
evidence, not proof of bit-exact hardware network execution.

## Completed v2 results, 2026-10-28

Vivado isolated RTL synthesis, before placement/routing:

| Core | DSP | BRAM36 equivalent | LUT |
| --- | ---: | ---: | ---: |
| Existing routed Q16 core, comparison only | 141 | 49 | 9,942 |
| q8_v2 | 82 | 39 | 15,150 |
| w8a16_v2 | 154 | 49 | 11,806 |
| w8a16_spatial_v2 | 82 | 49 | 20,422 |

Each v2 C simulation passed 13 cases, including 320x320 and 4x2100. The
three new synthesized cores each passed RTL cosimulation for the first
five small cases only. Interface simulation depth is 4096; large-shape
RTL validation requires a separate setup with sufficient interface depth.

Adding the existing fixed logic (71 DSP, 76 BRAM36, 26,504 LUT) gives
153 DSP, 125 BRAM36 and 46,926 LUT for one w8a16_spatial_v2 shared core.
This is within arithmetic device capacity, not proof of integrated fit.
Two q8_v2 copies already exceed all three budgets. Sharing one core between
the two model graphs is the next integration candidate; it does not imply
concurrent execution or that unsupported YOLO operators are implemented.

See `../../reports/DUAL_NEURAL_PREOPT_20261028.md` and the machine-readable
`../../diagnostics/hand_conv_preopt_20261028/summary.json` for accuracy,
validation scope, resource arithmetic and reproduction commands.
