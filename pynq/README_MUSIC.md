# PS-side music recognition

Upgrade `harmonic_tracking_v3`: `music_tracking.py` adds causal chord hysteresis
and 12-second decaying pitch-class event history. Initial acquisition takes two
fresh observations above 0.6 chord confidence. Switching normally requires three
consecutive matching observations spanning at least 220 ms; bass/inversion-only
changes require four spanning 500 ms. These are additional observed-frame gates,
not promises of wall-clock latency on the ARM CPU.
The server
keeps raw `chord`, and adds `raw_chord`, `stable_chord`, `tracking`, and rolling
`scale_evidence` by default. A held pitch-class set counts once. Brief active
uncertainty retains the last supported chord for at most 750 ms. When only a
strict subset of its notes remains and RMS falls below 70% of the peak RMS of
the confirmed chord, sustain hold extends to 1.8 seconds; renewed attacks bypass
that extension. The peak reference accommodates gradual decay before a note drops.
Uncertain/subset frames never refresh the support deadline. Held results are
explicitly marked `holding` or `switching`, while raw notes/chords stay unchanged.
Silence, inactive input, source errors and context changes clear the label.
Duplicate sequences cannot supply votes or extend holds. This improves display
stability without inventing missing notes or resolving musical ambiguity.
All seven natural modes sharing a collection remain alternatives, even if only
one collection is compatible. These scores do not identify a proven tonic.

The dictionary now also includes sixth, added ninth and ninth chords. Fits
within a frame are cached and stationary envelope refinement stops early.
Missing-fundamental candidates are compared with their upper octave; close
alternatives retain an octave-ambiguity flag. See `../MUSIC_UPGRADE.md` for the
measured speed comparison, limits, references and real-guitar acceptance steps.

`audio_music.py` fits a mixture of guitar harmonic templates to a 4096-point,
48 kHz one-sided FFT power spectrum (2048 unsigned bins: DC included, Nyquist
excluded). Each note template explains up to ten partials together. Several
spectral-decay, pure-tone and weak/missing-fundamental variants, nonnegative
joint refits, lower-note penalties, replacement and pruning prevent strong overtones from becoming
independent notes. This is spectral inference, not a largest-bin note list.

The default search is E2 through E6 (MIDI 40..88), at most six notes, with tuning
templates at -25, 0 and +25 cents. Frequencies/cents describe the fitted template;
they are not precision tuner measurements. Use `MusicConfig` to change the
range, polyphony, fit thresholds, or optional scale evidence. Only NumPy and the
Python standard library are required. Python 3.8 / NumPy 1.20 are supported.
PYNQ is optional when using direct stdlib `/dev/mem` mode; imports remain lazy.

## Recognition output

`MusicRecognizer().analyze(power, sequence=...)` returns a JSON-safe dictionary.
`status` is `silence`, `unknown`, or `recognized`; `signal_active` distinguishes
an active but unsupported sound from silence. Silence clears all notes. An
unsupported or noisy spectrum returns no asserted notes. Confidence is a fit
heuristic, not a probability. Each note includes MIDI/name, fitted frequency,
relative strength, supported partials, and an octave-ambiguity flag.

Chord recognition uses complete observed pitch-class sets and the lowest
inferred note as bass. It covers major/minor/diminished/augmented triads;
dominant, major, minor, minor-major, half-diminished and diminished sevenths;
sus2/sus4, seventh suspended chords, and power chords. Inversions use slash
names, such as `C/E`. Equivalent suspended and symmetrical diminished/augmented
interpretations appear in `alternatives`. Unmatched sets, single notes and
octave-only sets have an `unknown` chord. Missing-root and incomplete chords
are not assigned a definite name. Optional major/natural-minor scale evidence
reports compatibility and ambiguity; a single frame cannot establish a key.

## FFT hardware contract

The AUD4 version map below is verified by the parent's 25061-check PL run and
hardware ID `0x41554434` read; FCLK is 125 MHz. API `fft.contract` is `AUD4`.
PS recognition from the deployed hardware remains a separate board smoke check.

`audio_fft.py` defaults to AUD4 at `0x43C00000`, range `0x4000`. An explicit
`FFTRegisterMap` can override the map. Format: 4096 points at 48000 Hz, 2048
unsigned linear-power bins, DC included, Nyquist excluded. Input is signed
PCM16 multiplied by a Q15 Hann window, denominator 4095. FFT is unscaled;
power approximates `abs(fft(windowed_pcm16))**2 >> 20` (divisor 1048576).

| Offset | Meaning |
| --- | --- |
| `0x00` | Core ID `0x41554434` |
| `0x04` | Status: busy bit 0, ready bit 1, frozen bit 2, error bit 3 |
| `0x08` | Published-frame sequence |
| `0x0C` | Control: freeze bit 0, analysis test mode bits 2:1 |
| `0x10`, `0x14` | FFT length 4096, sample rate 48000 |
| `0x18`, `0x1C` | Dropped frames, peak count (maximum 24) |
| `0x20`, `0x24` | Power divisor 1048576, Hann denominator 4095 |
| `0x100 + 8*i` | Peak bin, unsigned power at `+4` |
| `0x1000 + 4*k` | Unsigned power bin `k=0..2047` |

The snapshot reader preserves control/source bits, requests freeze, waits for
frozen AND busy-clear AND ready, checks sequence around all bins/metadata, retries incoherent
reads, and releases its own freeze in `finally`. A freeze that was already
owned by the caller stays asserted. The PL acknowledgement must mean the bin
memory is quiescent and contains a complete published frame. One driver owns
the register port; its lock cannot coordinate other programs using the same IP.
Initialization validates core ID, length, rate, power divisor and window
denominator. Metadata includes absolute scale, drops and hardware peak table.

`set_test_mode(0..3)` selects live input, A4, C major or A minor. It requests
freeze, waits for busy-clear (ready is not required), updates only mode bits
while frozen, waits at least 100 microseconds for source-change capture reset,
then releases its own freeze. A caller-owned freeze stays asserted. CLI
`--test-mode` defaults to 0/live. Test voices feed analysis only and do not
change physical listening.

`DevMemMMIO` uses `/dev/mem` mmap directly, checks aligned register bounds and
closes resources. The current board requires sudo. This bypasses PYNQ/XRT
discovery without changing system permissions. The CLI never constructs an
Overlay, parses `.hwh`, initializes PS/DDR/FCLK or downloads a bitstream. Parent
deployment owns SD boot, PL programming and the existing 125 MHz FCLK.

`minimum_rms` defaults to 0.001 normalized full-scale PCM: about 33 signed PCM16
RMS counts / -60 dBFS. Hardware CLI sets `MusicConfig(power_scale=1048576.0)`;
Parseval energy, FFT length, PCM16 scale and Hann mean-square gain correct raw
power into RMS. For normalized PC spectra leave `power_scale=None`. Output
includes `spectrum.rms`, `rms_pcm16`, `rms_dbfs` (null at zero),
`rms_quantization_upper` and `minimum_rms`. Integer-bin truncation contributes
to the upper bound. Nyquist is omitted and window correction assumes a
stationary signal, so this is not an exact transient RMS measurement. Calibrate
the gate above actual idle noise and below wanted notes for the input/gain.
`silence_power` is an additional raw non-DC floor, default `1e-10`. Broadband
noise is also rejected using a median-bin floor and harmonic fit quality.

## JSON server

`audio_music_server.py` runs one sampling thread and a stdlib threaded HTTP
server. `GET /api/audio` (aliases `/api/music`, `/api/state`) returns analysis, sequence,
configuration, age and stale/error state; `GET /healthz` returns 200 for fresh
data and 503 for starting/error/unavailable/stalled acquisition. An unchanged sequence does
not refresh the age. Read failures clear asserted notes. API requests never
read MMIO directly. CORS is disabled unless an explicit origin is supplied.

Schema version 1 fields: `schema_version`, `status` (starting,
hardware_unavailable, error, silence, unknown, recognized), `source` (hardware
or explicit demo), `hardware_available`, `signal_active`, `confidence`,
`sequence`, `updated_at` (Unix seconds), `age_seconds`, `stale`, `error`,
`notes`, `chord`, `fft`, `spectrum`, `scale_evidence`. Analysis states include
`config`, `reason`, `limitations`. Unavailable hardware clears notes/chord and
never generates fake live data. Stalled acquisition retains old data marked
stale; repeated sequences do not refresh age. `hardware_available` means a
successful mapped read, not that a stalled capture is fresh.

Note objects contain `midi_note`, `note_name`, `pitch_class`, `frequency_hz`,
`cents`, `confidence`, `relative_strength`, `harmonic_support` and
`octave_ambiguous`. Chord fields: `status`, `label`, `root`, `quality`, `bass`,
`inversion`, `confidence`, `ambiguous`, `alternatives`; unknown values are null.
Spectrum adds `power`, `noise_floor`, `explained_fraction`, RMS fields and
`bins:[number]` (all 2048 original power values in bin order, including DC)
and `peaks:[{bin,frequency_hz,power}]`. Hardware bins preserve unsigned integers,
including values above signed-32-bit range; demo bins are normalized floats.
No log scaling, normalization or truncation is applied to the API bin array.
Spectral peaks include harmonics and are not
independent recognized notes. FFT metadata includes `fft_size`,
`sample_rate_hz`, `bin_count`, `bin_spacing_hz`, `window`, `power_format`,
`power_shift`, `power_scale_factor`, `power_scale`, `window_denominator`,
`window_format`, `contract`, `drops`, `peaks`, `analysis_test_mode` (also
`test_mode`). FFT/spectrum are null before successful acquisition.

Local synthetic use (does not import PYNQ or access a board):

```powershell
python pynq/audio_music_server.py --demo --port 8081
```

Parent-managed board invocation after programming matching PL into SD boot:

```text
sudo python3 -B pynq/audio_music_server.py --dev-mem --base-address 0x43C00000 --address-range 0x4000 --host 0.0.0.0 --port 8081 --test-mode 0
```

Optional `--register-map path.json` keys match `FFTRegisterMap` (decimal values).
`--window` must match PL. The same server serves existing `../web` unchanged at
`/`, with `--web-root` override; default bind is localhost. No clocks, system
permissions or configuration are modified.

## Offline checks and limitations

```text
python -B -m unittest discover -s pynq/tests -p test_audio_music.py -v
```

Tests synthesize actual time-domain harmonic signals with deterministic random
phases, then apply a Hann-windowed FFT. Coverage includes mono guitar pitches,
weak/missing fundamentals, triads, inversions, sevenths, sus/power voicings,
unequal/detuned mixtures, silence, noise, octave ambiguity, fake-MMIO coherent
snapshots/retries/control cleanup, and a local HTTP API roundtrip. No board
process is started and no board or FPGA is accessed by these tests.

Additional tests cover pure-tone C/Am PL voices, Hann Q15/shift-20 unsigned
quantization, absolute RMS gates, mode-change freeze/reset sequencing and
anonymous-local-memory mmap. Synthetic waveforms are generated by the test
file; no separate binary fixtures are needed for UART deployment.

Parent-run real board smoke check (do not run concurrently with the server):

```text
sudo python3 -B pynq/test_board_music.py --dev-mem --output board_music_smoke.json
```

This script uses the real driver, selects analysis modes 1/2/3, waits for at
least three observed new-frame advances after each switch, checks A4/C/Am from
the real FFT and saves JSON evidence. It always attempts to restore mode 0
in `finally`; failures return exit code 1. The offline suite verifies this
control flow using a fake source, including restoration after a failed check.

At 48 kHz, 4096 samples span 85.3 ms and bin spacing is 11.71875 Hz. Low guitar
semitones cannot be resolved from a fundamental bin alone; higher partials
provide evidence. Exact octave doublings may merge with harmonics. Distortion,
inharmonic strings, attacks, weak overlapping notes, chords above six notes,
and timbres outside the guitar template family can fail or return unknown.
Synthetic checks are not a measurement of real-guitar accuracy.

Actual AUD4 board evidence is saved in `../diagnostics/audio_board_evidence`.
The real PL-to-PS A4/C/Am smoke passed on the final BRAM build. With the sparse
correlation cache, actual ARM warm C-major inference averaged 463.00 ms over
five calls (maximum 464.16 ms); first preparation took 12.58 s. This excludes
capture and web latency and does not establish fast-phrase recognition.
All template nonzero bins are retained; dense-equivalence tests verify the
correlation change. The final local suite has 43 tests.
