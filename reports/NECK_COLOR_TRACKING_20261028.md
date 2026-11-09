# Colour-Border Neck Acquisition And Recovery - 2026-10-28

## Result And Scope

Deployed automatic colour/border/crossbar candidate acquisition on the PYNQ-Z2
PS, followed by existing PL pyramidal LK and PS RANSAC. No manual ROI or colour
calibration is required. This is a board-tested improvement to neck candidate
acquisition/tracking, not acceptance of complete real-time chord recognition.

The inspected live camera scene contained no guitar. Guitar motion evidence
below is a replay of an existing recording through the actual board PL chain,
using the existing port-8001 proxy. It is not a physical live guitar trial.

## Confirmed Problems

- The previous configured acquisition path used PS OpenCV DNN YOLO. A live
  status observation reported 7946.7 ms inference and confidence 0.4258 below
  the configured 0.5 threshold. Its worker waits another 10 seconds after
  completion before a new inference. This explains long waits in that mode;
  it is not a statement that all earlier geometric algorithms had this cost.
- Strong features near the hand skewed GFTT spatial quantiles. Initialization
  could reject a visible long neck despite independently detected points
  spanning it. A bounded diverse subset now passes the original count, hull,
  occupancy and span checks before it is accepted.
- Subsequent motion still caused insufficient-spread, invalid-motion and
  appearance rejections. Recent-neck recovery now runs on the same failed
  frame, with feature alignment preferred over translation. Every recovered
  ROI still requires current-frame support; blank images do not recover.
- Response metadata includes an optional 208-pixel fingertip crop. It can grow
  beyond Node's default response-header limit. This was reproduced as
  `HPE_HEADER_OVERFLOW` and an HTTP 503 while camera capture and PL remained
  healthy. It was a separate transport fault, not a USB failure.

## Implementation

- `pynq/neck_color.py`: at most 320x240 colour analysis; bounded Hough line
  pairing; Lab contrast across both long borders; interior-colour support;
  observed transverse ridges with a smoothly spaced run. It rejects plain
  strips and sampled furniture negatives without claiming semantic identity.
  Output is explicitly `color_border_candidate`, `lattice_confirmed=false`.
- `pynq/fretboard_tracker.py`: on failed initial spatial support, select up to
  24 actual GFTT features distributed in board coordinates and recheck using
  the existing gates. No invented features or relaxed quality thresholds.
- `pynq/vision_pc_bridge.py`: optional `color_border` acquisition; Lab warmup;
  same-frame recent-neck recovery; feature-first alignment on recovery.
  Large optional preview thumbnails are omitted from the copied HTTP metadata
  above a 14 KiB budget. Point coordinates, identity, timing and tracking remain
  in the response. Original worker results retain their previews.
- `web/board_proxy.mjs`: bounded 128 KiB upstream header allowance for the next
  web-service launch. The running port-8001 process was not restarted: automatic
  approval rejected stopping/restarting it and starting another service with
  `blocked by policy`. Board-side preview compaction works with that process.
- Packaging lists include the new module. The environment override is saved
  in `pynq/neck-acquisition.conf` and installed as a systemd drop-in.

PS performs colour candidate acquisition and final geometric validation. PL
performs image preprocessing, sparse LK and the existing fingertip neural
convolutions. This update did not place a second YOLO network in PL.

## Actual Board Replay

Source recording: `WIN_20261026_01_17_27_Pro.mp4` (personal recording excluded from the public package).
Input: JPEG quality 90, 640x480, sequential HTTP requests, 40 ms test pacing
plus host decode/encoding. Phases: 12 still frames, 24 motion frames sampled at
0.1-second video intervals, 6 black frames, 12 still frames after return.

| Measurement | Header fix, before same-frame recovery | With same-frame recovery |
| --- | --- | --- |
| First tracked neck candidate | 2.281 s | 1.328 s |
| Moving frames with neck ROI | 20/24 | 23/24 |
| Moving frames with hand ROI | 20/24 | 23/24 |
| Moving request latency median | 273.5 ms | 266.0 ms |
| Moving request latency p95 | See raw samples | 812.85 ms |
| Black frames retaining neck ROI | 0/6 | 0/6 |
| Black frames retaining hand/tip tracking | 0/6 | 0/6 |
| First neck candidate after black frames | 2.750 s | 2.797 s |
| HTTP 503 / header-overflow failures | 0/54 | 0/54 |

The second motion phase completed 24 requests in 12.031 seconds, approximately
1.99 updates/second including pacing, host work and recovery. The 266 ms median
is request roundtrip, not an end-to-end display FPS claim. Several recovery
requests took approximately 0.81 seconds. Initial search timing varies with
competing tasks; the table contains two runs, not a latency distribution.

Reports: `diagnostics/neck_color_probe_20261028/replay_board_final.json` and
`replay_board_recovery.json`. The latest `replay_36.jpg` shows the neck borders;
`replay_24.jpg` deliberately has no lines because that frame failed validation.
No neck ground-truth labels were supplied. A tracked-frame count is availability,
not boundary accuracy or a detection-accuracy percentage.

The saved-image experiment sampled eight frames uniformly from each of seven
videos: all 48 sampled guitar frames produced a candidate, and all eight frames
of the short no-guitar clip produced none. Two additional saved room negatives
also produced none. These are correlated samples and do not establish broad
generalization or precise edge placement; some candidate borders are imperfect.

## Actual Live Camera Health

`live_negative_final.json` records 12 distinct C922 frames through the board
processing endpoint. The inspected scene had no guitar. All 12 requests
completed, no neck ROI/hand ROI/tip coordinates were published, and the camera
stream identity remained unchanged. The final observation reported:

- Capture: 12.55 fps, zero capture failures, frame age 126.24 ms, not stale.
- PL bridge: ready, error null, neural PL fault false, fingertips enabled.
- Audio: hardware source, stale false, error null, result age 0.0574 seconds.

`live_camera_after_deploy.jpg` is the actual current camera view. Browser tab
inspection repeatedly timed out at `Emulation.setFocusEmulationEnabled`, so
the visible overlay in the user's active tab was not verified in this run.
No claim of a successful live guitar/browser test is made.

## Deployment And Recovery

Runtime: `/home/xilinx/fingertip_candidate_20261026/first_fold/runtime`.
Camera service: `fretlens-vision.service`, kept running throughout updates.
Processing/audio service: `fretlens-av.service`, restarted by the validated
software deployment script. A short audio interruption is expected on restart.
Mode: `FRETLENS_NECK_ACQUISITION=color_border` in
`/etc/systemd/system/fretlens-av.service.d/neck-acquisition.conf`.

Latest prior-version backup:
`/home/xilinx/fingertip_candidate_20261026/first_fold/runtime/previous_color_neck_20211129_221737`.
It contains the header compaction fix before same-frame recovery. The board
clock is wrong; these 2021 names were produced during 2026-10-28 host work.

For original pre-colour restoration, the original bridge is in
`previous_color_neck_20211129_213434` and the original fretboard tracker is in
`previous_color_neck_20211129_215010`. Inspect each backup before restoring;
they are incremental, not full system images. `deploy_neck_color.sh` also
rolls back software/config automatically on failed service validation.

The existing bitstream, HWH and fingertip weights were not replaced.
Verified board SHA256:

```text
b17829b68d04ebdb9d7ee3ba2dce78952cf2b73cd3e9c13f3d175a74f4509b4f  av_system.bit
a34fa04331ef2884036dd73f2640e048ef72d882186d2da20e0debf98fa0d64a  av_system.hwh
d89527ba413b79e6d0fa69425d7355e5d5b9b603e40a165075fe3ef33ff265a0  neck_color.py
23e0d0c3f84e430395ac852f848b967017c870d0b1fd8371c2e59b53c831141d  fretboard_tracker.py
476838a4f6b28c3cd5c88c760930345154499e812ef3d9ab76d772b5372204cc  vision_pc_bridge.py
```

`scripts/snapshot_neck_color.ps1` saves the relevant source, evidence and this
record to a new immutable scoped archive with a verified SHA256 manifest. It
does not overwrite the previously accepted project release or contain all
dependencies/weights needed for an independent full-system installation.

## Verification And Remaining Work

- Bridge: 68 tests passed.
- Colour acquisition: 5 tests passed, including rotation/mirror, recorded JPEG
  and room negatives.
- ROI fretboard: 21 tests passed.
- Fretboard tracker: 17 passed, 2 board-native tests skipped on Windows.
- Board correlation/native-support tests: all 5 passed after explicitly using
  the deployed runtime as working directory. An initial test launch from the
  SSH home directory imported an older same-named module; its failures were a
  test-path error. Current runtime import path and checksums were verified.
- Node proxy: 14 tests passed, including 32 KiB metadata forwarding.

The neural worker still measured 1549.9 ms per inference in the final replay.
Only 5/24 moving frames had accepted fingertip tracking in that run, compared
with 13/24 in the preceding run. Scheduling and recovery interact; this update
does not establish a fingertip speed/accuracy gain. Its hand ROI remains the
existing lightweight skin/geometry proposal, not a newly trained neural box.

No replay frame established the complete supported six-string/fret lattice.
Absolute fret numbers, string contact, occlusion, open/muted strings and
accurate current fingertip locations remain necessary for dependable notes
and chords. Candidate crossbars are not published as numbered frets. Further
work needs a physical guitar trial, independent border/fret ground truth,
then measured fingertip and processing acceleration with the same test inputs.

Page: http://127.0.0.1:8001/?camera=board
