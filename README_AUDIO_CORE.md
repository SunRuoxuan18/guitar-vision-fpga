# ZynqAudioLab Audio Core

Current entry (2026-10-16): use `START_HERE_AUDIO.md` and `RETURN_CHECKLIST.md`.
The verified active audio baseline is clean AUD4 under `output/audio_music`,
not the historical AUD3 described below. The local UI is http://127.0.0.1:8001/.
Offline vision/fingering updates are prepared but have not been deployed after
the user powered off the board. `reports/EXPERIMENT_REPORT.docx` records the
implemented scope and remaining physical validation.

## Verified hardware baseline: AUD3 (2026-10-14)

The user verified the programmed AUD3 revision on the physical PYNQ-Z2:

- SW0=0, SW1=0: the test tone plays in both ears without periodic pops.
- SW0=1, SW1=0: decoded guitar playback has no excessive hiss or periodic pops.

The active bitstream is `output/audio_ps_clock_wrapper.bit`, with JTAG
USERCODE `41554433`. The matching Vivado project is
`diagnostics/ps_clock_test/audio_ps_clock.xpr`. It uses PS FCLK0 at 125 MHz,
rather than the H16 Ethernet PHY clock. The earlier `ZynqAudioLab.xpr` and
`output/pynq_audio_board_top.bit` are not this verified hardware baseline.

A frozen copy of the bitstream, RTL, tests, constraints, programming scripts
and hardware handoff is saved in `output/verified_audio_AUD3_20261014`.
See that directory's `RESTORE.md` for restoration and build instructions.
FPGA configuration is volatile: after board power-off, boot the existing
PYNQ SD image and run `scripts/program_audio_ps_clock.ps1` again.

This milestone verifies audio acquisition and playback. FFT processing,
spectral peaks, chord recognition and camera/audio fusion remain future work.

## Current signal path

```text
125 MHz clock
    -> audio_sample_tick
    -> audio_test_tone or external PCM input
    -> audio_gain_clip
    +-> audio_level_meter
    +-> zero_cross_pitch
    +-> audio_frame_buffer
    -> processed PCM output
```

## RTL modules

### audio_sample_tick.v

Generates a one-clock enable pulse at an average rate of 48 kHz from the
125 MHz system clock. It is used by the internal test source. A real I2S input
will provide its own sample-valid event.

### audio_test_tone.v

Generates a signed 16-bit PCM sine wave near 440 Hz. It uses a 32-bit phase
accumulator and a 64-entry sine lookup table. This source lets the complete
audio pipeline run before the board codec is connected.

### audio_gain_clip.v

Applies gains of 1x, 2x, 4x, 8x and higher by shifting the signed sample. It
uses saturation at -32768 and +32767, preventing arithmetic wraparound. The
`clipped` output reports an over-range sample.

### audio_level_meter.v

Measures the peak absolute sample value over a configurable window. It also
produces `signal_active` for basic sound/silence detection.

### audio_frame_buffer.v

Stores 1024 consecutive signed PCM samples. Vivado infers block RAM for this
memory. `frame_ready` pulses when a complete analysis window is available.

### zero_cross_pitch.v

Estimates frequency from the period between positive threshold crossings. It
is suitable for proving the pipeline with a clean single tone. Guitar pitch
accuracy will later require an autocorrelation or YIN detector because real
guitar signals contain harmonics, noise and decaying amplitude.

### audio_processing_core.v

Connects all current modules. It can select either the internal test tone or
external signed 16-bit PCM. Its external PCM port is the future connection
point for the ADAU1761 I2S receiver.

### audio_i2s_rx.v / audio_i2s_tx.v

Receive and transmit stereo audio in standard I2S format. The current board
wrapper uses 24-bit samples in 32-bit channel slots, matching a common
ADAU1761 configuration. The I2S loopback test verifies that left and right
samples are not shifted or exchanged.

### audio_sample_cdc.v

Transfers each PCM sample event between the audio bit-clock domain and the
125 MHz system-clock domain. This prevents the processing core from reading a
multi-bit sample while it is changing in another clock domain.

### pitch_note_decoder.v

Maps a measured frequency to the nearest equal-tempered MIDI note from E2
through E6. It outputs the MIDI number, pitch class, octave, target frequency
and signed frequency error. For example, approximately 440 Hz produces MIDI
69, pitch class 9 and octave 4 (A4).

### pynq_audio_stream_top.v

Connects the real-time board audio path:

```text
ADAU1761 I2S input
  -> 24-bit sample receive
  -> 16-bit PCM conversion and clock-domain crossing
  -> audio analysis core
  -> clock-domain crossing and 24-bit conversion
  -> ADAU1761 I2S output
```

This wrapper intentionally performs direct analysis/pass-through only. Delay,
echo, distortion and other audio effects are not part of the current design.

## Simulation

`tb_audio_processing_core.v` verifies:

- continuous PCM generation;
- two complete 1024-sample frames;
- sound-level detection;
- a pitch result near 440 Hz;
- block RAM readback;
- saturated clipping at high gain.

Expected result:

```text
PASS: complete audio processing core verified.
```

## Completed versus pending

Completed and synthesizable:

- 48 kHz test timing;
- signed PCM test tone;
- gain and saturated clipping;
- peak level and active-signal detection;
- 1024-sample BRAM frame buffer;
- simple pitch estimation;
- integrated core and self-checking simulation.
- 24-bit stereo I2S transmit/receive loopback;
- audio-clock/system-clock sample transfer;
- frequency-to-MIDI/note/octave conversion;
- complete I2S-to-analysis-to-I2S stream simulation.

Pending analysis and PS integration:

- Xilinx FFT IP and magnitude/peak processing;
- higher-accuracy autocorrelation or YIN pitch detector;
- Zynq PS, AXI control, DMA and Python interface;
- browser visualization.

## PYNQ-Z2 board self-test

The PL-only project described below is an earlier diagnostic. Use the
verified PS-clock AUD3 baseline above for current hardware development.

The project now also contains a self-contained PL-only board top named
`pynq_audio_board_top`. It generates the codec's 10 MHz MCLK, configures the
ADAU1761 over I2C, and sends the internal 440 Hz test tone to the headphone
output. This is the shortest hardware milestone before adding the Zynq PS.

Board controls:

```text
SW0/SW1 = 00  internal 440 Hz tone
SW0/SW1 = 01  raw I2S input-to-output loopback (no PCM processing)
SW0/SW1 = 10  decoded I2S loopback (no clock crossing or PCM processing)
SW0/SW1 = 11  complete PCM processing path (1x gain)
BTN0      restart the audio system
LED0      MCLK locked
LED1      codec configured
LED2      non-silent PCM detected
LED3      I2C configuration error
```

In decoded I2S loopback mode (SW0=1, SW1=0), LED0 means 32 BCLKs per
channel, LED1 means 16, and LED2 means another count. LED3 always reports
an I2C initialization error.

In test-tone mode (SW0=0, SW1=0), LED2 latches if the 10 MHz clock generator
loses lock after one second of stable codec operation, and LED3 latches if
BTN0 is pressed or its input
briefly goes high. Both latches clear on FPGA reprogramming. In this mode
LED3 is a reset diagnostic, not the I2C error indicator; check LED3 in any
other mode for an I2C error.

In full processing mode (SW0=1, SW1=1), LED1 shows activity on the right
input channel and LED2 shows activity on the left input channel. These
diagnostic indicators update about every 21 ms (1024 samples) and help
locate an intermittent loss of audio before the PCM processing path.

The generated board bitstream is copied to:

```text
D:/FPGA/ZynqAudioLab/output/pynq_audio_board_top.bit
```

### System clock requirement

The H16 system-clock input is supplied by the Ethernet PHY, rather than an
independent board oscillator. PYNQ maintainers and PYNQ-Z2 users report that
this nominal 125 MHz clock can become unstable without an Ethernet link.
Since this PL-only design derives codec MCLK from H16, a loss of this clock
can reset the codec initialization sequence and interrupt audio in every mode.

For this design, connect the board to a powered computer Ethernet port or a
powered network switch and verify that the Ethernet link is up. An IP address
or Internet access is not required for this clock-stability test. For a future
PS/PL design, use a configured PS FCLK instead of relying on the PHY clock.

Reference: [TUL reference manual, section 5](https://dpoauwgwqsy2x.cloudfront.net/Download/pynqz2_user_manual_v1_0.pdf),
[PYNQ clock-source discussion](https://discuss.pynq.io/t/bitstream-works-ok-when-programmed-by-jtag-but-not-from-pynq/858/5),
with a [PYNQ-Z2 confirmation](https://discuss.pynq.io/t/bitstream-works-ok-when-programmed-by-jtag-but-not-from-pynq/858/18).

During the earlier H16 test on 2026-10-14, the user reported small audio pops about every three seconds
and a latched MCLK-loss indicator; Windows showed the wired Ethernet adapter
disconnected. BTN0 debounce did not remove the pops. The active-link test is
was not verified, so it did not establish an Ethernet-link remedy. The later
PS-clock AUD3 listening test below passed without periodic pops.

### PS-clock diagnostic and restart checks (2026-10-14)

The first PS FCLK0 version removed periodic pops and kept the test-mode
clock-loss LED off, but right-only headphone sound and excessive input noise
remained at that stage. The final AUD3 listening test resolved those symptoms.

After a power cycle, restoring that bitstream did not restore audio. FCLK0
was observed changing from 125 MHz to 100 MHz after programming; the writer
has not been identified. More importantly, inspection of the placed PS
diagnostic netlist showed that out-of-context synthesis converted both
inferred I2C tri-state drivers into logic. Both pads had ordinary OBUFs,
without output enables, so codec configuration could not work reliably.

Revision AUD3 explicitly instantiates SCL/SDA IOBUFs, connects SDA feedback
to the I2C controller, disables the codec PLL before reconfiguration, and
sets both channels' digital volumes, analog bypass, converter rate and serial
pin modes explicitly. The build checks actual pad-buffer connectivity and
nonconstant output enables before exporting the bitstream. Its JTAG USERCODE
is `41554433`; a later mismatch means another design replaced this revision.

The codec command/NACK/reinitialization simulation and independent stereo
serial-bit test passed. The stereo test checked 128 frames at 48 kHz with
32 BCLKs per channel, with nonzero output in both channels. These digital
checks do not establish analog sound quality or codec PLL lock. The startup
still uses a fixed 25 ms PLL delay rather than lock-status polling.

Reprogram after board power-off with:

```powershell
& 'D:\FPGA\ZynqAudioLab\scripts\program_audio_ps_clock.ps1'
```

This requires the PS to have already initialized its IO PLL and PS/PL
interface. It preserves CPU/DDR/PLL/MIO settings and sets only FCLK0's rate.
Do not load a competing PYNQ overlay while testing. The final AUD3 listening
test passed: stereo test tone and decoded guitar playback were normal, with
no reported periodic pops or excessive guitar hiss. AUD3 built with setup slack +0.525 ns
and hold slack +0.129 ns. Final-netlist checks confirmed both open-drain pad
enables and SDA input feedback. JTAG programming read back USERCODE
`41554433`, and a subsequent PS readback confirmed FCLK0 still at 125 MHz.
External I2S input/output timing delays remain unspecified; the positive
internal timing slack is not a complete external-interface timing sign-off.
