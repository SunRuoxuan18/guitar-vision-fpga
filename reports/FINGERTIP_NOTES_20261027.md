# Four Fingertips To Fretboard Notes - 2026-10-27

## Delivered

The board PS now converts the current four-fingertip neural model output into
string/fret position candidates and, with explicit wire calibration, MIDI,
note names and equal-tempered frequencies. Existing PL inference, bitstream,
camera configuration and audio processing were retained.

`pynq/visual_positions.py` reuses the existing perspective rectification and
measured-wire interpolation for four named tips, independently of the legacy
21-landmark hand model. It locates the two bounding fret wires at each tip's
rectified height, then finds nearby string wires at that longitudinal position.
It retains ambiguous adjacent-string and fret-wire candidates. It rejects tips
outside measured support and stale/unverified lattice propagation.

`pynq/fingertip_notes.py` retains at most 64 small geometry/calibration snapshots.
Delayed neural output is projected using its own sequence, stream and generation,
never the latest frame's geometry. Results older than eight seconds or without
cached source geometry are unavailable. Historical projections remain historical
and are not substituted into the current visual/audio fusion positions.

`pynq/vision_pc_bridge.py` validates the optional request calibration, records
source geometry, and publishes `analysis.fingertip_notes`. Only exact-current
neural positions become `visual_positions` for audio fusion. Existing relative
position and audio-constrained alignment behavior remains available.

## Calibration

Web page: http://127.0.0.1:8001/?camera=board

Reload the page after deployment. Under settings/details, open calibration.
After six strings and consecutive fret wires are detected, enter the actual fret
number of the first image-left detected boundary (zero only if that boundary
is the nut), choose the image-top string and the increasing-fret direction, then
confirm the measured strings and consecutive fret wires. Standard tuning is
E2 A2 D3 G3 B3 E4, image-top string 6 by default. Do not use zero merely because
the first visible fret number is unknown.

The wire indices are ordered in rectified board coordinates. The detector's
canonical orientation must agree with the visual left/top labels. Inspect the
numbered wire overlay after applying calibration. A reversed camera view is
supported by independently reversing strings and fret direction.

Calibration is bound to stream, generation and lattice epoch. A changed or lost
lattice requires explicit confirmation again. No automatic assignment of fret
numbers from ROI corners is performed. User confirmation assumes consecutive
measured fret wires; incorrect/missing-wire detection or wrong calibration can
still produce wrong notes.

The formula is `MIDI = open_string_MIDI + physical_fret`, with frequency
`440 * 2 ** ((MIDI - 69) / 12)`. For example, string 5 fret 2 is MIDI 47,
B2, approximately 123.471 Hz. Fret-wire numbers descend correctly when the nut
is to the right. Custom tuning is supported by the board request contract;
the current web controls use standard tuning and physical fret numbers.

The live overlay includes wire numbers after calibration. Current candidate
positions include note names. Delayed results appear as historical source-frame
note candidates in the position readout, separate from current position evidence.

## Limits

This is geometric projection, not pressure/contact detection. The heatmap model
does not provide calibrated visibility probabilities; raw heatmap peaks are not
used as confidence. Hidden or lifted fingertips can still be predicted and must
be treated as candidates. A fingertip outside the board yields no note. A tip
near a wire or between strings remains ambiguous. Open strings, muted strings,
barres, bends and the complete sounding voicing are not inferred from these four
points. No independent visual chord verdict is introduced.

Only a neck ROI without six measured strings and fret boundaries is insufficient.
The current slow neural worker often returns historical results, not current
ones. This change does not solve neural speed or wire-detection availability.
Live musical accuracy has not been established by the synthetic tests below.

## Verification And Deployment

- New projection/history tests: 8 passed, covering all strings, four finger names,
  reversal, nonuniform boundaries, perspective, boundary ambiguity, identity
  changes, missing/expired sources and invalid tips.
- Existing visual projection tests: 5 passed.
- Bridge tests including request calibration validation: 58 passed.
- Audio fusion tests: 11 passed; fingertip worker tests: 17 passed.
- Web board overlay, calibration forwarding/reset, current/historical note
  display, desktop/mobile layout: passed with no page errors.
- Existing ROI browser pipeline and fretboard tests: passed.
- Board Python import and service-readiness checks passed.
- Actual board synthetic-geometry smoke test returned
  `BOARD_NOTES_PROJECTION_PASS`, including B2 and history isolation.
- Camera after deployment: streaming, 13.10 fps, zero capture failures, not stale.
  Bridge ready, no neural PL fault.

Board deployment transcript:
`diagnostics/console_Probe_20261027_131954.log`.
Hashes: `diagnostics/fingertip_notes_SHA256SUMS`.
Deployment script: `scripts/deploy_fingertip_notes.sh`.
Board smoke script: `diagnostics/verify_fingertip_notes_board.py`.

Board runtime: `/home/xilinx/fingertip_candidate_20261026/first_fold/runtime`.
Previous Python files are saved in
`previous_notes_20211128_144041` under that directory. The board clock accounts
for the 2021 backup timestamp. Deployment restarted only the AV service and
preserved the running camera service. The script rolls back on validation failure.
PL bitstream SHA256 remains
`b17829b68d04ebdb9d7ee3ba2dce78952cf2b73cd3e9c13f3d175a74f4509b4f`.
