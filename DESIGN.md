# Design

## Source of truth
- Status: Active
- Last refreshed: 2026-10-15
- Overnight revision: 2026-10-16. Focus valid fretboard geometry with padded
  crop and a red perimeter. Keep inference in original-frame coordinates and
  reset the crop on stale or lost geometry. Gray/Sobel thumbnails explicitly
  identify browser processing; they do not claim deployed PL acceleration.
- Show observed fingertip projections and catalog chord shapes separately.
  A reference chart never establishes the player's actual voicing. Leave
  unknown/open/muted strings unknown unless independently supported. Scale
  diagrams are candidate pitch collections, not proof of tonic or hand shape.
- Primary product surfaces: browser audiovisual recognition workstation and PYNQ-Z2 audio pipeline
- Evidence reviewed: `README_AUDIO_CORE.md`, `rtl/`, `sim/`, and the existing `web/` implementation

## Brand
- Personality: focused rehearsal-room analysis instrument with a restrained rock identity
- Trust signals: explicit source labels, confidence, audiovisual agreement, camera/audio/FPGA connection state, and visible processing provenance
- Avoid: tuner-first layouts, decorative music-player clutter, fake hardware controls, and claims that placeholder recognition is real

## Product goals
- Goals: make the live fretboard image the primary workspace; combine visual fingering evidence with audio notes; show chord and scale/mode candidates; preserve waveform, spectrum, and hardware status as diagnostics.
- Non-goals: replace a DAW, provide guitar effects, or claim production-grade finger/chord recognition before the vision model and FPGA data bridge exist.
- Success signals: the user can open a camera, see a fret/string guide, start an audio input, understand the current visual and audio evidence, and read one fused musical conclusion.

## Personas and jobs
- Primary personas: FPGA learner building a demonstrable multimodal system; guitarist checking fingering and harmonic output.
- User jobs: frame the fretboard, verify detected finger positions, compare intended fingering with sounded notes, and identify likely chord and scale context.
- Key contexts of use: desktop browser during development, lab demonstration, and guitar practice.

## Information architecture
- Primary navigation: one workspace, no navigation tabs. Center the video,
  followed by chord, note, scale and fingertip results. Keep camera/audio
  connection, calibration and detection visible. Put alternative inputs and
  diagnostics in a closed disclosure; remove the processing-flow diagram.
- Core routes/screens: one responsive workstation for the first release.
- Content hierarchy: live fretboard image first; fused chord/scale conclusion second; audio and visual evidence third; waveform/spectrum and FPGA state last.

## Design principles
- Principle 1: make the centered video the primary working area, with the first
  results directly below it in the desktop viewport.
- Principle 2: distinguish real browser input, FPGA data, model output, and simulated demonstration data.
- Principle 3: show separate visual and audio evidence before their fused conclusion.
- Tradeoffs: this release uses native browser APIs and a guide overlay; robust hand landmarks, fretboard calibration, and audiovisual inference remain integration work.

## Visual language
- Color: light neutral page, white controls, dark text, green actions and cyan
  guides; preserve clear warnings and synthetic-data labels.
- Typography: readable system sans with fixed sizes and zero letter spacing.
- Spacing/layout rhythm: single centered column, large video, compact results.
- Shape/radius/elevation: flat sections, thin dividers, up to 4px radii, no shadows.
- Motion: short state transitions and subtle live indicators; honor reduced-motion preferences.
- Imagery/iconography: real camera frames are the primary imagery; overlay lines and landmarks must support inspection rather than decoration.

## Components
- Existing components to reuse: browser audio input, pitch estimator, waveform/spectrum canvases, and FPGA status adapter placeholder.
- New/changed components: centered video, essential connection and calibration
  controls, results band, collapsed calibration settings and debug disclosure.
- Variants and states: idle, camera-only, audio-only, live inputs, demo, unavailable, permission error, and awaiting model/FPGA data.
- Token/component ownership: `styles.css` owns visual tokens; `app.js` owns media streams, demo state, recognition state, and canvas rendering.

## Accessibility
- Target standard: WCAG 2.1 AA intent for contrast, focus, labels, and status announcements.
- Keyboard/focus behavior: all controls use native buttons/selects with visible focus rings and logical order.
- Contrast/readability: status never relies on color alone; video overlays use outlined labels for changing backgrounds.
- Screen-reader semantics: live status and fused result use announcement regions; canvases and video have text alternatives.
- Reduced motion and sensory considerations: nonessential transitions stop under reduced-motion preference.

## Responsive behavior
- Supported breakpoints/devices: desktop first, usable down to a 360px-wide phone.
- Layout adaptations: single video/results column on all viewports. On phones,
  chord/note share a row, scale uses a full row, diagnostics stack vertically.
- Touch/hover differences: controls remain at least 40px high and do not depend on hover.

## Interaction states
- Loading: camera/audio buttons show an explicit starting state.
- Empty: video stage explains how to start a camera and what is not yet being inferred.
- Error: distinguish browser support, permission denial, missing device, and unavailable FPGA/model data.
- Success: show active source, separate evidence, fused verdict, and confidence together.
- Fretboard acquisition: board camera defaults to automatic visible string and
  fret-wire detection. Show measured line counts and searching states; clear
  geometry on stale frames, loss or stream change. Absolute fret numbers and
  string identities remain unknown without independent evidence. Manual
  four-point calibration remains inside the settings disclosure as a fallback.
- Manual fretboard calibration: board camera supports a checked follow-neck control.
  Keep the neck still while selecting four perimeter corners. Show locking,
  tracking and lost states; clear fret/string evidence on lost or stale frames.
  Recalibration is explicit after loss, restart or source/size changes. Local
  camera and imported video retain fixed calibration; disable follow-neck there.
- Disabled: device selector and unsupported controls explain why they are unavailable.
- Offline/slow network, if applicable: the page has no network dependency after loading.

## Content voice
- Tone: concise, technical, honest, and suitable for a project demonstration.
- Terminology: use “指板画面”, “弦/品格”, “视觉指法”, “音频音符”, “视听融合”, “和弦”, and “音阶/调式”.
- Microcopy rules: label simulated values `DEMO`; label live camera without a model as “画面已连接，识别模型待接入”.

## Implementation constraints
- Framework/styling system: native HTML, CSS, and JavaScript; no build step or third-party runtime.
- Design-token constraints: keep colors and spacing in CSS variables; avoid oversized marketing heroes and one-hue palettes.
- Performance constraints: reuse media streams and canvases; keep one animation loop per active signal source.
- Compatibility constraints: camera/microphone require localhost or HTTPS plus user permission.
- Test/screenshot expectations: verify demo state, camera fallback, desktop layout, and mobile layout; ensure video and controls never overlap.

## Open questions
- [ ] Which vision runtime will produce hand landmarks: browser prototype, PS/Python OpenCV/MediaPipe, or PL accelerator? / project owner / affects the recognition adapter.
- [ ] Which FPGA-to-PS transport will expose FFT peaks and onset timestamps: AXI-Lite, DMA, or an initial UART bridge? / project owner / affects the audio adapter.
- [ ] What fixed camera angle and fretboard calibration target will be used for the final demo? / project owner / affects coordinate mapping.
