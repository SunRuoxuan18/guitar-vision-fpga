`timescale 1ns / 1ps

module audio_processing_core #(
    parameter integer CLK_FREQ_HZ = 125_000_000,
    parameter integer SAMPLE_RATE_HZ = 48_000,
    parameter integer FRAME_SAMPLES = 1024,
    parameter integer FRAME_ADDR_WIDTH = 10,
    parameter integer LEVEL_WINDOW_SAMPLES = 1024
)(
    input  wire                         clk,
    input  wire                         rst,
    input  wire                         use_test_tone,
    input  wire signed [15:0]           external_sample,
    input  wire                         external_sample_valid,
    input  wire [2:0]                   gain_shift,
    output wire signed [15:0]           processed_sample,
    output wire                         processed_sample_valid,
    output wire                         clipped,
    output wire [15:0]                  peak_level,
    output wire                         signal_active,
    output wire                         level_valid,
    output wire [31:0]                  frequency_x100,
    output wire                         pitch_valid,
    output wire                         note_busy,
    output wire                         note_valid,
    output wire [6:0]                   midi_note,
    output wire [3:0]                   note_class,
    output wire [3:0]                   note_octave,
    output wire [31:0]                  target_frequency_x100,
    output wire signed [31:0]           frequency_error_x100,
    output wire                         frame_ready,
    output wire [31:0]                  frame_number,
    input  wire                         frame_read_enable,
    input  wire [FRAME_ADDR_WIDTH-1:0]  frame_read_address,
    output wire signed [15:0]           frame_read_data
);

wire sample_tick;
wire signed [15:0] tone_sample;
wire tone_valid;
wire signed [15:0] selected_sample;
wire selected_valid;

audio_sample_tick #(
    .CLK_FREQ_HZ    (CLK_FREQ_HZ),
    .SAMPLE_RATE_HZ (SAMPLE_RATE_HZ)
) sample_tick_inst (
    .clk         (clk),
    .rst         (rst),
    .sample_tick (sample_tick)
);

audio_test_tone tone_inst (
    .clk          (clk),
    .rst          (rst),
    .sample_tick  (sample_tick),
    .pcm_sample   (tone_sample),
    .sample_valid (tone_valid)
);

assign selected_sample = use_test_tone ? tone_sample : external_sample;
assign selected_valid  = use_test_tone ? tone_valid  : external_sample_valid;

audio_gain_clip gain_inst (
    .clk              (clk),
    .rst              (rst),
    .sample_valid_in  (selected_valid),
    .sample_in        (selected_sample),
    .gain_shift       (gain_shift),
    .sample_valid_out (processed_sample_valid),
    .sample_out       (processed_sample),
    .clipped          (clipped)
);

audio_level_meter #(
    .WINDOW_SAMPLES  (LEVEL_WINDOW_SAMPLES),
    .ACTIVE_THRESHOLD(16'd1000)
) level_inst (
    .clk          (clk),
    .rst          (rst),
    .sample_valid (processed_sample_valid),
    .sample_in    (processed_sample),
    .peak_level   (peak_level),
    .signal_active(signal_active),
    .level_valid  (level_valid)
);

zero_cross_pitch #(
    .SAMPLE_RATE_HZ(SAMPLE_RATE_HZ)
) pitch_inst (
    .clk            (clk),
    .rst            (rst),
    .sample_valid   (processed_sample_valid),
    .sample_in      (processed_sample),
    .frequency_x100 (frequency_x100),
    .pitch_valid    (pitch_valid)
);

pitch_note_decoder note_decoder_inst (
    .clk                  (clk),
    .rst                  (rst),
    .pitch_valid          (pitch_valid),
    .frequency_x100       (frequency_x100),
    .busy                 (note_busy),
    .note_valid           (note_valid),
    .midi_note            (midi_note),
    .note_class           (note_class),
    .octave               (note_octave),
    .target_frequency_x100(target_frequency_x100),
    .frequency_error_x100 (frequency_error_x100)
);

audio_frame_buffer #(
    .FRAME_SAMPLES(FRAME_SAMPLES),
    .ADDR_WIDTH   (FRAME_ADDR_WIDTH)
) buffer_inst (
    .clk          (clk),
    .rst          (rst),
    .sample_valid (processed_sample_valid),
    .sample_in    (processed_sample),
    .frame_ready  (frame_ready),
    .frame_number (frame_number),
    .read_enable  (frame_read_enable),
    .read_address (frame_read_address),
    .read_data    (frame_read_data)
);

endmodule
