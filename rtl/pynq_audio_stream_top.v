`timescale 1ns / 1ps

// Digital audio stream wrapper for the PYNQ-Z2 ADAU1761 interface.
// Codec register initialization over I2C is intentionally handled by the
// Zynq PS/Python layer. This module handles only the real-time I2S/PCM path.
module pynq_audio_stream_top #(
    parameter integer CLK_FREQ_HZ = 125_000_000,
    parameter integer SAMPLE_RATE_HZ = 48_000,
    parameter integer FRAME_SAMPLES = 1024,
    parameter integer FRAME_ADDR_WIDTH = 10,
    parameter integer LEVEL_WINDOW_SAMPLES = 1024
)(
    input  wire                         sys_clk,
    input  wire                         rst,
    input  wire                         use_test_tone,
    input  wire                         audio_bclk,
    input  wire                         audio_lrclk,
    input  wire                         audio_sdata_in,
    output wire                         audio_sdata_out,
    output wire                         audio_sdata_decoded,
    input  wire [2:0]                   gain_shift,
    output wire signed [15:0]           pcm_input_sample,
    output wire                         pcm_input_valid,
    output wire signed [15:0]           pcm_output_sample,
    output wire                         pcm_output_valid,
    output wire [15:0]                  peak_level,
    output wire                         signal_active,
    output wire                         right_signal_active,
    output wire [31:0]                  frequency_x100,
    output wire                         pitch_valid,
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

wire signed [23:0] i2s_left_sample;
wire signed [23:0] i2s_right_sample;
wire i2s_sample_valid;
wire clipped_unused;
wire level_valid_unused;
wire note_busy_unused;
wire signed [15:0] pcm_output_bclk;
wire pcm_output_bclk_valid_unused;
wire [15:0] right_peak_unused;
wire right_level_valid_unused;
wire signed [23:0] i2s_output_sample = {pcm_output_bclk, 8'b0};

audio_i2s_rx #(
    .SAMPLE_WIDTH(24),
    .SLOT_WIDTH  (32)
) i2s_rx_inst (
    .rst         (rst),
    .bclk        (audio_bclk),
    .lrclk       (audio_lrclk),
    .sdata_in    (audio_sdata_in),
    .left_sample (i2s_left_sample),
    .right_sample(i2s_right_sample),
    .sample_valid(i2s_sample_valid)
);

audio_level_meter #(
    .WINDOW_SAMPLES  (1024),
    .ACTIVE_THRESHOLD(16'd1000)
) right_level_inst (
    .clk          (audio_bclk),
    .rst          (rst),
    .sample_valid (i2s_sample_valid),
    .sample_in    (i2s_right_sample[23:8]),
    .peak_level   (right_peak_unused),
    .signal_active(right_signal_active),
    .level_valid  (right_level_valid_unused)
);

audio_sample_cdc #(
    .SAMPLE_WIDTH(16)
) input_cdc_inst (
    .src_clk   (audio_bclk),
    .src_rst   (rst),
    .src_valid (i2s_sample_valid),
    .src_sample(i2s_left_sample[23:8]),
    .dst_clk   (sys_clk),
    .dst_rst   (rst),
    .dst_valid (pcm_input_valid),
    .dst_sample(pcm_input_sample)
);

audio_processing_core #(
    .CLK_FREQ_HZ        (CLK_FREQ_HZ),
    .SAMPLE_RATE_HZ     (SAMPLE_RATE_HZ),
    .FRAME_SAMPLES      (FRAME_SAMPLES),
    .FRAME_ADDR_WIDTH   (FRAME_ADDR_WIDTH),
    .LEVEL_WINDOW_SAMPLES(LEVEL_WINDOW_SAMPLES)
) processing_core_inst (
    .clk                  (sys_clk),
    .rst                  (rst),
    .use_test_tone        (use_test_tone),
    .external_sample      (pcm_input_sample),
    .external_sample_valid(pcm_input_valid),
    .gain_shift           (gain_shift),
    .processed_sample     (pcm_output_sample),
    .processed_sample_valid(pcm_output_valid),
    .clipped              (clipped_unused),
    .peak_level           (peak_level),
    .signal_active        (signal_active),
    .level_valid          (level_valid_unused),
    .frequency_x100       (frequency_x100),
    .pitch_valid          (pitch_valid),
    .note_busy            (note_busy_unused),
    .note_valid           (note_valid),
    .midi_note            (midi_note),
    .note_class           (note_class),
    .note_octave          (note_octave),
    .target_frequency_x100(target_frequency_x100),
    .frequency_error_x100 (frequency_error_x100),
    .frame_ready          (frame_ready),
    .frame_number         (frame_number),
    .frame_read_enable    (frame_read_enable),
    .frame_read_address   (frame_read_address),
    .frame_read_data      (frame_read_data)
);

audio_sample_cdc #(
    .SAMPLE_WIDTH(16)
) output_cdc_inst (
    .src_clk   (sys_clk),
    .src_rst   (rst),
    .src_valid (pcm_output_valid),
    .src_sample(pcm_output_sample),
    .dst_clk   (audio_bclk),
    .dst_rst   (rst),
    .dst_valid (pcm_output_bclk_valid_unused),
    .dst_sample(pcm_output_bclk)
);

audio_i2s_tx #(
    .SAMPLE_WIDTH(24),
    .SLOT_WIDTH  (32)
) i2s_tx_inst (
    .rst         (rst),
    .bclk        (audio_bclk),
    .lrclk       (audio_lrclk),
    .left_sample (i2s_output_sample),
    .right_sample(i2s_output_sample),
    .sdata_out   (audio_sdata_out)
);

// Board diagnostic: exercise I2S decode/encode without clock crossing.
audio_i2s_tx #(
    .SAMPLE_WIDTH(24),
    .SLOT_WIDTH  (32)
) decoded_tx_inst (
    .rst         (rst),
    .bclk        (audio_bclk),
    .lrclk       (audio_lrclk),
    .left_sample (i2s_left_sample),
    .right_sample(i2s_right_sample),
    .sdata_out   (audio_sdata_decoded)
);

endmodule
