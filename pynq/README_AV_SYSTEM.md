# Unified Audio And Vision System

## Fingertip Tracking And Event Scheduling

The hand pipeline tracks13selected joints (four pressing fingertips, index
direction and palm/crop anchors) plus at most16texture features, instead of
21joints plus32features. The existing21-point neural network is unchanged.
Untracked joints become drawing-only affine estimates with validity=false;
the page omits their dots/segments. Invalid tips cannot become string evidence.
Forward/backward flow, patch agreement, RANSAC and age/identity checks remain.

After a successful neural refresh, stable tracking waits up to1.5seconds before
another request, with a4second source-anchor refresh deadline. Independent
finger motion, fewer than three valid pressing fingertips or loss trigger
immediate requests when the single model worker is free. Verified rigid hand
translation alone does not force a new pose inference. Requests are not queued
behind a busy model. `hand_detection.scheduler` exposes these decisions.

Same-image sequential board replay measured median hand PL flow22.95->14.45ms,
hand processing60.44->48.80ms and ROI-frame HTTP roundtrip314.59->275.03ms.
An earlier candidate replay had only1.9percent roundtrip improvement, so the
end-to-end gain is load-dependent. Neural ROI inference remains about2.26seconds
in the final static run; cold full search reached5.87seconds. No100ms live
fingering claim or smaller trained network is implied. Initial hand acquisition
and some warmed frames are excluded from the ROI-frame medians. The saved
real image still has fingertips outside the observed fret run: geometry
completion/contact accuracy remains unresolved.
The final browser replay instead sampled2.866s neural inference,431.8ms HTTP
and992ms image-translation response. Tracking still works, but this run does
not demonstrate an improvement in overall visible response time.

See `diagnostics/fingertip_schedule_comparison.json` and the accompanying replay
and health records. Full-joint mode remains available as the explicit
`HandPipeline(..., keypoint_mode='full')` comparison path. A trained lightweight
fingertip detector and validated barre/contact classifier are future work;
retaining index joints does not itself implement either model.

## Unattended Simulation And Integrity Repairs

`harmonic_tracking_v4_av_integrity` adds a PS-monotonic observation timestamp
before audio inference. Fusion and motion-assisted confirmation reject audio
and video observation pairs more than300ms apart. Receipt/observation timestamps
are software association bounds, not synchronized exposure or acoustic onsets.
Reordered FFT snapshots cannot replace newer audio publications;32-bit rollover
remains valid. Weak and octave-uncertain notes cannot constrain visual mappings.
An unsupported finger projection prevents a joint-consistent verdict.

Run the reproducible, offline ablation with:

```powershell
.\vision\.venv\Scripts\python.exe diagnostics/simulate_av_comparison.py --output diagnostics/unattended_av_comparison.json
```

It compares audio-only and motion-assisted tracking on independently specified
guitar tabs, generated21-joint hands, projected wires and measured synthetic
waveforms. The16 low/high-register fixtures cover C7, Cadd9, Csus4, Cdim7, C9,
C7b9, Cadd11 and Am. The ideal10-observations/s rule timeline has300ms median
audio-only confirmation versus100ms with motion assistance; neither figure is
measured end-to-end board latency. Vision-only abstains because the automatic
grid has no independent absolute fret or string-direction anchor.

All16 waveform cases retain the expected interpretation among possible chord
names. Octave-doubled notes are not always separately identifiable: only8/16
recover the exact MIDI set. The common C7 tab `x32310` omits the fifth and remains
unknown under the complete-tone grammar. These failures/limitations are recorded,
not counted as live performance successes. The bounded harmonic split repair
prevents two true fundamentals being pooled into a false lower note when that
note has no measured fundamental and both replacements explain independent energy.

The real-image replay's four valid fingertips all lie beyond the measured wire
region; zero projected positions remain the correct outcome for that geometry.
The position diagnostics and page now distinguish this from a missing hand.
Fast live fingering, occluded-wire completion and sounding-string/contact
accuracy remain unverified. No hardware or neural architecture change is implied
by this software repair.

The exact software passed150ARM-native tests and was installed with a rollback
backup. Local verification passes361tests with17platform skips. The actual-board
image replay retains six strings/17visible fret wires, follows image translation
and clears on blank input with zero browser model requests. Sampled preview
29.88fps and board visual updates2.88/s are distinct measurements; ROI neural
inference3.173s and image-translation response999ms do not establish fast live
fingering. See `diagnostics/unattended_av_browser.json` for the full replay and
`diagnostics/unattended_av_final_health.json` for a read-only service snapshot.
The frontend preserves fresh board conflict/time-association feedback instead
of overwriting it with generic geometry text. Expired feedback cannot assert
agreement. PC remains responsible for capture, transport and display only.

## Labeled Playing Validation

From the project directory, run:

```powershell
.\scripts\run_av_validation.ps1
```

The script creates a unique `diagnostics/av_trial_*` session, counts down three
seconds, and records read-only board audio/vision API observations for40seconds.
It prompts C->Am->F->G, twice at3seconds/chord, twice at1second/chord, then
four times at0.5seconds/chord. Keep the existing web camera feed running with
the hand and fretboard visible; this recorder does not open a camera or run
inference on the PC. An unattended capture is diagnostic only.

For the high-position/extended-chord trial, use:

```powershell
.\scripts\run_av_validation.ps1 -Profile extended
```

This uses C7->Cadd9->Csus4->Cdim7 with the same speed phases. Use movable
voicings above fret7 and annotate the actual inversion and onset from independent
media. The cue does not verify that a particular position was played.

Record independent video/audio of the hand, actual sound, and terminal/screen,
including the `SESSION START` marker. Map the video timeline to the printed
session time; annotate actual sounding-chord onset, not the cue schedule.
If sounding intervals include silence, omit the gaps instead of labelling them
as sustained chords. Use the actual inversion label if applicable; chord labels
are compared exactly. Report alignment/onset uncertainty in milliseconds.

Edit the generated `annotations.template.json` into a separate annotations file:
set `verified` to true only after checking the recording, `kind` to
`independent_media_annotation`, `reference` to the local recording path,
`media_sha256` to its SHA256, `timing_uncertainty_ms` to the measured uncertainty,
and replace segment boundaries/labels with the actual performance. This is a
human annotation declaration, not automatic proof that the media labels are true.
PowerShell `Get-FileHash -Algorithm SHA256 <recording>` provides its hash;
use lowercase hex in the annotations. Score with:

```powershell
.\vision\.venv\Scripts\python.exe diagnostics/av_validation.py score --session diagnostics/av_trial_<id> --annotations <annotations.json>
```

`diagnostic.json` contains interface availability, unique observed result counts,
neural completion counts, and fusion statuses without claiming accuracy.
`scored.json` adds per-phase missed acquisitions, time-weighted correctness,
onset-to-stable-label delay, and false stable labels during sustain. A correct
label must hold for100ms. Missing/error/stale/test-mode observations never score;
the maximum assumed hold is250ms, less board-reported age and conservatively
the whole request RTT. Misses count against the within200ms fraction, which
also includes annotation uncertainty. Browser rendering latency is excluded.
Use independent video to measure actual sound-to-screen delay and hand movement.

The recorder binds logs to a session and SHA256, rejects unverified cue templates,
and verifies the annotated media hash. It does not certify finger-contact accuracy
or camera freshness from status snapshots. The board status endpoint reports
`analysis_age_seconds` since processing completion; this is not camera exposure
age. The recorder counts hands/positions only when this age plus the request RTT
is within the observation limit. A fixed frame number is not a live video test.
No independent vision-only chord endpoint or synchronized same-recording ablation
currently exists. `complete_audiovisual_verified` therefore remains false even
when audio labels or audio-constrained visual mappings are correct.

## Extended Chords And Moving Positions

The installed software `harmonic_tracking_v4_av_extended` names36 complete
observed pitch-class sets, including dominant/major/minor sevenths, add9/add11,
sus2/sus4, diminished/half-diminished/dim7, altered dominants and ninth/eleventh
extensions. It retains inversions and equivalent names. Missing chord tones,
intentional omissions and rootless voicings are not silently filled in.
The MIDI40-88 search covers standard-tuning guitar through fret24. Naming is
octave-invariant; voicing and string/fret association use exact MIDI pitches.

Current verified PL joint motion detects both one-finger changes and whole-hand
shifts. It can shorten chord switching to two independent reliable audio results
separated by at least80ms; actual processing and transport add latency. Decaying
tones without valid visual motion retain the conservative sustain rules.
Fusion bounds PS input-receipt age to750ms and processing-completion age to300ms
when both timestamps exist; legacy records retain the300ms input limit. This
prevents a fresh completed result from already being rejected solely because
processing took over300ms, without accepting an arbitrarily old input frame.
The reported input age remains visible; this does not shorten acquisition delay
or establish synchronized exposure/audio clocks.

The PS fusion module vectorizes100 fret/orientation hypotheses. Two different
measured pitch collections can constrain a persistent alignment of a tracked
grid, including high positions. Stream/grid changes, contradictions or expired
support invalidate it. This remains an audio-constrained hypothesis, not an
independent nut/fret marker or proof of finger pressure. Matching current fingers
and sounding pitch classes produce a separate `joint_decision`; stale audio
labels produce `awaiting_audio_transition` rather than a joint success claim.
The main video labels a matching aligned candidate with `?` and rejects mismatched
visual sequence/grid identities. No recognition is moved to the PC.

Background PS acquisition now tries up to six distinct border/width proposals
before the full-image fallback. Each must pass the existing six-string and
equal-tempered fret checks, and its result is revalidated on the current frame.
This repairs a reproduced browser-canvas scaling case where the highest-scoring
region alone was too wide. It does not fill in hidden strings or unseen frets.

Board candidate tests, waveform benchmarks and deployment manifests are recorded
under `diagnostics/extended_av_*`. Synthetic harmonic input proves bounded
regressions, not accuracy on fast live guitar. The existing Q16 PL neural core
is unchanged; whole-network inference and independent absolute fret anchoring
remain limitations.

## Latest Software Verification (2026-10-21)

The installed hardware is the verified dedicated-depthwise Q16 release below.
Latest same-image full hand inference measures4.714s; warm ROI measures
1.883-2.073s. The preceding run measured4.662s and1.890-2.053s, so the newest
PS conversion change does not establish a whole-model speedup. All104 neural
layers and final hand/ROI coordinates pass board
numerical checks. PS-only remains faster for complete hand/ROI inference;
the isolated palm graph alone is slightly faster on this PS/PL core.

PC capture -> transport -> board PS/PL -> browser rendering remains the only
production route. Gigabit links can send raw RGBA to avoid PS JPEG decoding;
other links retain JPEG. PS libjpeg direct RGBA decoding measured49.018ms
versus68.019ms OpenCV on identical pixels. Returned display data can contain
two actual PL320x240 gray/edge planes, avoiding PS preview JPEG encoding.
This changes display transport only; tracking still consumes640x480 PL output.

Fixed an OpenCV4.2 custom-layer return-container tensor retention leak, bounded
stream admission and lattice reacquisition, and cached CMA tensor handoffs.
Background lattice results must validate against the current frame. A current
PL LK keypoint is explicitly distinct from fresh neural inference.

Neck proposals now use repeated transverse wires to avoid cropping outer
strings between two inner long lines. Intensity ridges supplement directional
Sobel peaks, with six independently supported lines and fret-spacing checks.
The PS C ridge helper matches the NumPy reference exactly in six fixture cases
and measures2.93ms for the last case. The same replay now returns six string
candidates and15 visible frets on89/90 frames after temporal confirmation;
absolute fret numbers and finger pressure remain unknown.

Direct ARM NEON byte packing now writes B,G,R,0 directly into the cached DMA
input, without an intermediate OpenCV image. Nine independent byte cases and
five actual-PL DMA golden cases pass, including noncontiguous RGBA and arbitrary
alpha. Same-board transport microbenchmark18.79->7.09ms; concurrent replay input
packing median5.89ms. Gray/Sobel arithmetic still executes in PL.

Opposite neck-border segments no longer need a long shared visible interval;
their union proposes a region, followed by the same string/fret validation.
Short visible fret runs use scale-relative curvature rather than a fixed
pixel cutoff. Seventeen board geometry regression tests pass. A signed-ridge
fret fallback suppresses double edges and tolerates partial row occlusion;
only two end outliers may be removed, never interior missing wires. A public
300px hand/guitar image now confirms six strings and six fret/end wires.
Latest actual-board replay has59/60 lattice frames and33 relative-position
frames,2.895 board updates/s and298.43ms median HTTP;
maximum six-string cross-section error versus supplied annotations is2.318px
in the original image (annotation error is from the earlier evaluation).
This single still is not contact/fingering accuracy
or a live recognition benchmark. Evidence:public_lattice_evaluation.json.
Latest replay:`diagnostics/hand_ps_transport_public.json`.

Latest150-frame replay:3.121 board updates/s,280.98ms median round trip and
432.19ms p95. Fresh neural inference median2.286s under concurrent work,
completing0.287 new inferences/s. The previous3.104Hz/276.50ms run is similar;
there is no demonstrated overall throughput gain. Memory
plateau check passes;149/150 lattice candidate frames still yield zero finger
position frames because the partial visible lattice stops before the hand.
Latest browser
replay:26fps preview,1446ms translated-image response, blank clearing, real PL
neural/LK calls and zero PC model requests. Preview FPS is not recognition FPS.
No labeled rapid fingering sequence or complete audiovisual chord decision
has passed; `diagnostics/realtime_acceptance.json` intentionally reports failure.
An audio-constrained grid mapping is not an independent visual chord verdict.

Evidence: `diagnostics/hand_realtime_native_lattice.json`,
`diagnostics/ridge_native.json`, `diagnostics/lattice_native_tests.log`,
`diagnostics/hand_realtime_browser_final.log`, `diagnostics/hand_tiled_neon.json`.
Latest evidence: `diagnostics/hand_ps_transport_installed.json`,
`diagnostics/hand_ps_transport_numerical.json`, `diagnostics/pl_hand_browser.json`.
The browser sample measures1.64 board updates/s and608ms round trip with actual
web rendering; raw transport replay and browser figures are distinct workloads.
The preceding browser sample was2.02Hz/494ms with636ms translated-image response;
this run does not demonstrate improved interactive latency.

## Current Dedicated-Depthwise Hardware (2026-10-21)

`output/hand_neural_dw_bounded` is installed in `/home/xilinx/av_system`.
Bit SHA256: `ec50df2447175bb26f245811d8cc3d73feaa45e96879e361b961e9d5ec2f8889`.
Backup: `/home/xilinx/av_before_dw_bounded_20261021`.
Depthwise weights are cached across spatial tiles; pointwise results use
explicit64-bit bursts on a separate results AXI master from128-bit operands.
All26 raw integer/guard tests, full model/landmarks, five DMA pixel cases,
three known LK translations and real PL A4/C/Am audio checks pass. Local seeded
joint motion of1-3pixels is followed independently of a stationary surrounding
region in35-39ms. These fixtures do not prove neural fast-fingering accuracy.

Routed setup/hold slack:+0.000/+0.050ns, zero failing constrained endpoints;
DRC and bus skew pass. Usage:36,490LUT,42,053FF,74RAMB36+78RAMB18,212DSP.
Setup margin rounds to zero and the design has limited DSP headroom. Existing
I/O/CDC methodology warnings remain. Functional acceptance is bounded to the
tested hardware/settings. Evidence: `hand_dw_bounded_*` and `pl_joint_motion.json`.
The continuous-read pointwise cache passed C/RTL, physical,26 integer/guard,
full-model and combined board checks but regressed full inference to5.149s
and ROI to2.067-2.235s. Both pointwise and depthwise kernel groups became
slower; it was rejected. It is not part of this installed release.

The PS peak selector uses array comparisons with the same scalar thresholds,
stable candidate order and merge rules. All288 independent oracle cases and
11 board geometry tests pass; per-call median2.983->1.777ms. Native PS patch
correlation and direct cached-CMA neural packing remain enabled. PC performs
capture/transport/rendering only.

The latest PS-only transport/scheduling changes are installed with backup
`/home/xilinx/av_before_ps_transport_20261021`. Direct word deinterleaving keeps
output conversion bit-identical to the previous implementation across60
signed-boundary/random cases, while an independent cast oracle checks the
legal HLS accumulation range below2**45. Microbenchmark30.35->28.73ms.
Cached fixed-length least-squares projections and one batched coordinate
transform preserve210 fret-spacing decisions and two replay geometries;
same-input PS geometry median89.66->76.77ms. These local gains are not an
end-to-end recognition speedup. Evidence:`hand_unpack_candidate.json`,
`lattice_scheduling_board.json`. Deployment checks numerical evidence against
the exact tested runtime hashes and retains a rollback copy.

`hand_stage_profile.json` separates actual board stages on the original
1390x774 screenshot: warm ROI crop233-250ms, pose graph1.607-1.611s,
of which51 PL calls total1.285s and kernels0.677s. The crop figures must not
be applied to the640x480 production camera stream. Whole-graph tensor
handoffs and execution remain the primary neural acceleration work.

## Original Packed Q16 Release (Superseded)

`output/hand_neural_q16` is retained as the rollback reference.
Bit SHA256: `d3e75fce8cbb4b8a69ca9db54e3dc83461638653f8a39322c43d14b8ef6ebb6b`.
Board backup: `/home/xilinx/av_before_q16_20261021.tar.gz`.

All 100 convolution and 4 dense layers execute on PL with packed signed16
operands, per-output weight scales and48-bit accumulators. PS owns graph
scheduling, tensor quantization/dequantization, activations/pooling, crop/decode
and geometry validation. ARM NEON only converts tensor layouts/numbers; neural
MACs stay in PL. PC captures, encodes/transports and renders only.

Same-image full hand:9.321 ->5.802 seconds (1.61x,37.8% less time).
Same-run PS-only:4.094 seconds, so PL remains slower. Browser warm ROI is
approximately2.93 seconds. This is not real-time neural recognition. Maximum
full-hand normalized coordinate difference from floating reference:0.004343;
numerical agreement is not a measured guitar-fingering accuracy rate.

Routed setup/hold slack:+0.035/+0.005ns, zero failing constrained endpoints;
DRC and bus skew pass. Usage:37,746LUT,42,876FF,67RAMB36+38RAMB18,170DSP.
The hold margin is small; existing I/O/CDC methodology warnings remain.
HLS passes eight independent integer reference cases.

Tracking uses32 palm-region features plus21 independent joints with PL
forward/backward pyramidal LK, PS RANSAC and local appearance checks. Invalid
joints stay invalid until a new neural result. Catch-up uses at most three
image pairs, each checked. PS projects fresh or currently flow-validated
fingertips onto confirmed
current string/fret lines. Outputs are relative position candidates, with
pressure/sounding unknown. Missing nut/fret-number evidence cannot become
absolute notes or a chord. Full audiovisual fingering recognition is incomplete.
Current flow candidates require independently valid joints, current sequence,
actual PL flow evidence and a neural anchor no older than8seconds. They retain
the `optical_flow` source and neural age; they never become fresh neural output.

Quality92 JPEG cuts this replay upload from675,008 deflate bytes to88,447bytes.
Thin-wire JPEG accuracy still needs live validation. Append `transport=lossless`
to retain the lossless path. Bandwidth savings are not CPU decode acceleration.

Deployed browser replay passes:29.88fps preview,1.88board updates/s,22.92ms PL LK
pair,767ms translation response, blank-frame clearing, zero PC model requests.
Separate44-frame replay:423.85ms median HTTP,327.70ms processing. Median latency
regresses versus preceding JPEG/FP32 run329.82/244.99ms: faster CNN still competes
for ARM/DDR. These are screenshot replays, not live30fps recognition evidence.
Old diagnostic thumbnails are already burned into the source screenshot.

The final compact supervisor identity channel reads the same live PYNQ server
state in1.60ms median, instead of40-60ms of full metadata IPC. Version mismatch
and unavailable owner fail closed. Global servers keep the original check.
Final44-frame replay:2.14responses/s,367.76ms median HTTP,278.93ms processing.
This improves the first Q16 deployment, but does not beat all earlier FP32
median measurements. Latest browser acceptance:29.99fps preview,1.41updates/s
at the sampled instant,2.99s ROI inference and1081ms translation response.
Results fluctuate; these are not deterministic latency guarantees.

Verification:110 Python tests (five platform skips on Windows), four compact
identity tests separately pass on board Linux;13 proxy tests, eight HLS
cases, native ARM tensor checks, hardware model/landmark checks and deployed
browser replay pass. Historical browser suites with obsolete hidden controls
are not reported as passing. Evidence:
`diagnostics/hand_neural_q16_final_profile.json`,
`diagnostics/hand_neural_q16_final_comparison.json`,
`diagnostics/hand_realtime_q16_compact.json`, `diagnostics/pl_hand_browser.json`.
The earlier browser result is retained in `diagnostics/hand_q16_browser.json`.

Build with `vitis-run --mode hls --tcl hls/hand_conv_q16/run_hls.tcl`, then
`HAND_NEURAL_Q16=1` for `build_hand_neural.tcl` and
`implement_hand_neural.tcl`. `scripts/package_av_system.py` packages the release;
`install_av_system.sh` compiles the native helper on ARM. Normal boot service
remains enabled. Further acceleration requires reducing inter-layer ARM/DDR
traffic, fusing more graph operations in PL and validating a smaller or
lower-precision model. Neither perfection nor real-time chord fusion is proven.

## Previous measured release: buffered FP32 (2026-10-21)

The deployed `output/hand_neural_buffered` supersedes the initial neural release
and the historical PS-only sections below. All 100 convolution and 4 dense
layers run on PL; PS retains graph scheduling, activations, crop and decoding.
PC captures, transports and displays only. No PC neural fallback is enabled.

Dedicated pointwise/dense weight caching, eight partial-sum banks, buffered
input tiles and continuous 64-pixel compute reduce full-image hand inference
from 12.448 s to 9.321 s (1.34x, 25.1% less time). Same-run PS takes 4.106 s:
the PL version is STILL slower than PS. This is not real-time finger recognition.
Depthwise weight caching and II=1 reduce that kernel aggregate by about 36%.

Nine HLS reference cases, full-model output comparison, all 104 PL layer calls
and 21 landmark checks pass. Normalized landmark difference: 3.67e-6. Routed
setup/hold slack: +0.006/+0.012 ns; no failing endpoints; DRC and bus skew pass.
The setup margin is small. Usage: 37,049 LUT, 46,923 FF, 103 RAMB36 + 55 RAMB18,
167 DSP. Existing audio and LK remain in the same bitstream.

Evidence: `diagnostics/hand_neural_buffered_profile.json`,
`diagnostics/hand_neural_buffered_comparison.json`, `output/hand_neural_buffered/`.
An earlier direct-AXI pointwise experiment regressed to 13.805 s and was rejected.
Rebuild with HAND_CONV_BUILD set to hls/hand_conv/build_buffered for HLS and
HAND_NEURAL_BUFFERED=1 for the prepare/implementation scripts. Existing isolated
build directories are never overwritten by the prepare script.

The 2026-10-21 hand-neural release in `output/hand_neural` additionally contains
`hand_conv` at `0x43c30000`, clocked at100MHz. It executes all100Conv and4Gemm
layers of the pretrained hand models; PS runs graph scheduling, activations,
pooling and crop/decode. Missing or faulted hardware does not fall back to PC.
The board benchmark passed numerical checks and21landmarks, but this generic
FP32 implementation is slower:12.51s full detection versus4.11s on PS alone.
Do not describe this release as a neural acceleration success or30fps system.
Actual browser replay shows1.44boardupdates/s with PL LK between detections.
The original `output/av_system` release remains available as a verified backup.

The AVS1 bitstream contains the clean AUD4 audio path, 4096-point Hann FFT,
grayscale/Sobel stream and cached four-level iterative LK accelerator together.
Vision uses FCLK0 at 100 MHz; audio uses FCLK1 at 125 MHz. An AXI clock converter
connects the audio control port to the 100 MHz PS control bus. Audio FFT capture
uses BRAM, while vision DMA/LK uses the PS DDR HP port. The unused ROI projection
accelerator is omitted; geometry/consistency checks remain in PS software.

| Hardware | Address | Clock |
| --- | --- | --- |
| Vision DMA | `0x40400000` | 100 MHz |
| Iterative pyramidal LK | `0x43c00000` | 100 MHz |
| AUD4 audio/FFT registers | `0x43c20000` | 125 MHz |

PC USB camera -> browser RGBA -> Ethernet -> PS bridge -> PL grayscale/Sobel
and LK -> PS fretboard geometry -> coordinates -> browser overlay.
Guitar line input -> codec/I2S -> PL PCM/FFT -> PS polyphonic recognition ->
browser audio/visual evidence. PC camera capture does not bypass PL processing.
The browser may display 30 camera frames/s while board tracking updates less
often. These are separate rates. This design does not promise 30 tracking/s.

## Build

Run from the repository root with Vivado 2026.1 installed:

```powershell
vivado -mode batch -source scripts/build_av_system.tcl
vivado -mode batch -source scripts/implement_av_system.tcl
vision/.venv/Scripts/python.exe scripts/package_av_system.py
```

The build uses `diagnostics/av_system_build` and exports into `output/av_system`.
Export stops if setup/hold timing is negative, DRC has an error/critical warning,
or the codec I2C tristate enable is constant. Inspect the CDC report as well.
The routed AVS1 meets setup+0.024ns and hold+0.019ns. The CDC report still flags
inherited audio reset crossings (25 CDC-1 and one CDC-10); it is not CDC-clean.
The added 100/125MHz AXI control crossing uses three-stage XPM synchronizers.
Audio reset CDC hardening remains separate from this latency/integration change.
The packaged SHA256 manifest prevents mixed `.bit`/`.hwh` releases.
Linux owns PS/DDR initialization; never run a debugger `ps7_init` during loading.

## Run On The Board

Extract `diagnostics/av_system_deploy.tar.gz` into `/home/xilinx/av_system` and
install `fretlens-av.service` and `fretlens-av.target` into
`/etc/systemd/system`. Stop old audio/vision overlay owners first. Then run:

```sh
sudo systemctl daemon-reload
cd /home/xilinx/av_system
sudo systemctl stop fretlens-av.service
gcc -O3 -ffp-contract=off -shared -fPIC -o libaudio_fit.so.new native/audio_fit.c -lm
mv -f libaudio_fit.so.new libaudio_fit.so
sudo systemctl start fretlens-av.target
systemctl --no-pager status fretlens-av
```

The persistent supervisor programs once, configures both running PS clocks, validates
addresses, reads the AUD4 identity/metadata, and records `/run/fretlens-av.json`.
It forks audio/vision workers after loading so they inherit the PYNQ Device
overlay state. This board's PYNQ state does not persist between independent
processes; separate attach-only services cannot reliably use is_loaded().
Workers never reprogram PL. A worker failure stops both and reports the failure;
there is no automatic reload loop. To reload PL, restart `fretlens-av.service`.
To run at power-on after validation, enable `fretlens-av.target`.
The conflicting legacy camera services must stay disabled.

Audio HTTP is port 8081; PC-to-PL video HTTP is port 8084. On the PC, run
`node serve_board.mjs` from `web` and open `http://127.0.0.1:8001/`.
Choose the PC camera and connect board audio. No board USB camera is required.

## Latency Evidence

The web audio poll period is 100 ms instead of 500 ms, scheduled from request
start without overlapping requests. The audio service polls at 50 ms and skips
unchanged FFT sequences before requesting freeze or reading all spectrum bins.
No recognition thresholds or chord hold rules are weakened for faster numbers.
The audio unit enables `FRETLENS_AUDIO_NATIVE=1`: a small ARM C library runs the
existing sparse template dot products and bounded coordinate-descent NNLS.
Python still performs harmonic/chord decisions and temporal filtering. The
NumPy reference remains available with the variable unset; enabling native
without its compiled library fails explicitly. Fast-math is not used.
On the same ARM synthetic harmonic C-major input, profiling measured about
443 ms with Python versus 224 ms with native arithmetic and prepare-once buffer
validation. These profiled CPU-only numbers exclude capture, MMIO and display;
use concurrent hardware acceptance results for the deployed pipeline.

Audio API `timing` separates coherent snapshot reads, inference and PS processing.
An FFT window spans 85.33 ms; polling, chord confirmation, transport and rendering
add delay. Its processing metric is not capture-to-display latency.
`age_seconds` reports PS result age, not hardware sample timestamps.

For hardware acceptance, the supervisor `--self-test` checks real PL A4/C/Am
sources before starting workers and restores live input. Stop the supervisor
before running standalone `test_board_music.py --dev-mem --base-address 0x43c20000`.
Do not run multiple freeze/test-mode writers. `benchmark_av_audio.py`
also owns the audio port and restores live mode. Compare the same video fixture
sequence alone and with this diagnostic source running. This proves simultaneous
hardware processing, not real guitar fingering accuracy or synchronized acquisition.
Supervisor `--serve --self-test --benchmark-audio-seconds 45` runs a bounded
diagnostic C-major recognition worker alongside real video PL, writes
`audio_joint_benchmark.json`, restores live mode and exits. Restart the normal
service afterwards. The normal service always selects live mode0.

## Deployed Acceptance Results

The installed AVS1 release passed real PL A4, C and Am source checks, restored
live input, and passed a browser test through the actual board for automatic
acquisition, motion tracking, loss rejection and reacquisition without corner clicks.
These use diagnostic tones/rendered fretboards, not labelled live guitar ground truth.
The enabled boot target starts one supervisor and its two forked workers; the
earlier three independent service design was replaced because this PYNQ version
keeps overlay ownership in process memory. Power-cycle startup is configured but
has not been power-cycle tested in this acceptance run.

The ARM C library also reads aligned volatile 32-bit MMIO words while the existing
freeze/acknowledgement/sequence checks hold the FFT snapshot coherent. This is not
an ordinary memcpy of device memory. Snapshot reads dropped from about49ms to
about3ms. A45.43s joint run published246 C-major results (5.42/s), with170.20ms
median PS processing, no reported errors and a hardware drop counter of0 throughout.
The preceding read-loop run had212.82ms median processing and2 recorded drops.
Different test lengths and load mean these are bounded measurements, not a hard
real-time guarantee. Templates are warmed before the audio HTTP port opens;
allow roughly12s at service startup instead of stalling the first played chord.

With active chord recognition, the latest video run had203.45ms median processing
and269.30ms median HTTP round trip for tracking frames. Only35/42 frames were
tracking (the preceding joint run had41/42); geometry error remained at most2.43px
on accepted synthetic frames. The latest run fails tracking-continuity acceptance,
so it is explicitly not presented as an overall vision acceleration or30fps result.
PS geometry/lattice validation and recovery still limit the complete pipeline.
The raw measurements and quality gates are in diagnostics/av_vision_comparison.json,
diagnostics/av_audio_joint_benchmark.json and diagnostics/pc_auto_browser.json.
No camera-to-display latency or real fingering accuracy is asserted by these tests.

## Historical PS hand model and PL tracking (superseded)

This section records the earlier PS-only neural release. It is not the current
Q16 deployment; the current PL neural execution and measurements are above.

PC camera input remains on USB at the computer; it only captures/transports the
pixels. PS OpenCV DNN runs OpenCV Zoo MediaPipe palm and21-landmark networks.
The release includes OpenCV4.2 compatibility adapters and upstream license.
PS inference runs in one background worker and never accesses the PL directly.

The existing PL grayscale/Sobel and pyramidal LK cores remain in AVS1. A PS
detection is propagated through a bounded history of PL grayscale frames using
forward/backward PL LK and PS RANSAC. Unmatched stream/generation, lost feature
support, missing history, or an old detection cannot become a current hand box.
The browser receives results for the same uploaded capture and runs no model.
Inside the red hand box it composites the matching PL grayscale/edge planes.

Neural layers and weights do NOT run in PL. PL accelerates hand-region tracking,
not the hand neural network itself. Propagated joints are marked unobserved and
must not be used as fresh evidence of independent finger motion/string contact.
One hand is detected per inference. Full search is repeated periodically or
after failed ROI confidence; ROI inference uses the PL-tracked position.

Actual-board screenshot replay evidence is diagnostics/ps_hand_browser.json/png.
Typical measured PS full inference4.5s; ROI inference1.7s; hand PL forward/back
kernel15.8ms; total processing382ms/HTTP613ms, approximately1.6updates/s. Translation
response957ms in that replay; no live fast-guitar or30fps recognition claim.
