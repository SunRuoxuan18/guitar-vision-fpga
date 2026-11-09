`timescale 1ns / 1ps

// PYNQ-Z2 board self-test top level.
// SW0/SW1: 00 internal tone, 01 raw I2S loopback,
//          10 decoded I2S loopback, 11 complete PCM processing path.
module pynq_audio_board_top (
    input  wire       sys_clk,
    input  wire       rst_btn,
    input  wire       source_sw,
    input  wire       gain_sw,
    output wire       codec_mclk,
    input  wire       audio_bclk,
    input  wire       audio_lrclk,
    input  wire       audio_sdata_in,
    output wire       audio_sdata_out,
    inout  wire       audio_i2c_scl,
    inout  wire       audio_i2c_sda,
    output wire [1:0] codec_addr,
    output wire [3:0] led,
    output wire signed [15:0] analysis_pcm_sample,
    output wire analysis_pcm_valid,
    output wire analysis_reset
);

wire mclk_locked;
wire i2c_scl_drive_low;
wire i2c_sda_drive_low;
wire i2c_sda_feedback;
wire codec_init_busy;
wire codec_init_done;
wire codec_init_error;
wire [5:0] codec_debug_command;
wire stream_reset;
wire stream_sdata_out;
wire decoded_sdata_out;

reg [3:0] reset_pipe = 4'hF;
reg button_sync_1 = 1'b0;
reg button_sync_2 = 1'b0;
reg button_pressed = 1'b0;
reg [20:0] button_stable_count = 21'd0;
reg clock_loss_seen = 1'b0;
reg button_reset_seen = 1'b0;
reg [26:0] stable_clock_count = 27'd0;
reg clock_monitor_armed = 1'b0;
reg [7:0] bclk_per_channel = 8'd0;
reg [7:0] channel_bit_count = 8'd0;
reg lrclk_previous = 1'b0;

wire signed [15:0] pcm_input_sample;
wire pcm_input_valid;
wire signed [15:0] pcm_output_sample;
wire pcm_output_valid;
wire [15:0] peak_level;
wire signal_active;
wire right_signal_active;
wire [31:0] frequency_x100;
wire pitch_valid;
wire note_valid;
wire [6:0] midi_note;
wire [3:0] note_class;
wire [3:0] note_octave;
wire [31:0] target_frequency_x100;
wire signed [31:0] frequency_error_x100;
wire frame_ready;
wire [31:0] frame_number;
wire signed [15:0] frame_read_data;

// A noisy BTN0 edge must not restart the codec mid-stream.
always @(posedge sys_clk) begin
    button_sync_1 <= rst_btn;
    button_sync_2 <= button_sync_1;
    if (button_sync_2 == button_pressed) begin
        button_stable_count <= 21'd0;
    end else if (button_stable_count == 21'd1_249_999) begin
        button_pressed <= button_sync_2;
        button_stable_count <= 21'd0;
    end else begin
        button_stable_count <= button_stable_count + 1'b1;
    end
end

audio_mclk_gen mclk_gen_inst (
    .clk_125mhz (sys_clk),
    .rst        (button_pressed),
    .codec_mclk(codec_mclk),
    .locked     (mclk_locked)
);

// Ignore startup lock transitions; arm the sticky diagnostic after one
// second of uninterrupted codec operation.
always @(posedge sys_clk) begin
    if (codec_init_done && !codec_init_error && mclk_locked && !button_pressed) begin
        if (!clock_monitor_armed) begin
            if (stable_clock_count == 27'd124_999_999)
                clock_monitor_armed <= 1'b1;
            else
                stable_clock_count <= stable_clock_count + 1'b1;
        end
    end else if (!clock_monitor_armed) begin
        stable_clock_count <= 27'd0;
    end

    if (clock_monitor_armed && !mclk_locked && !button_pressed)
        clock_loss_seen <= 1'b1;
    if (button_pressed)
        button_reset_seen <= 1'b1;
end

always @(posedge sys_clk or posedge button_pressed) begin
    if (button_pressed)
        reset_pipe <= 4'hF;
    else if (!mclk_locked)
        reset_pipe <= 4'hF;
    else
        reset_pipe <= {reset_pipe[2:0], 1'b0};
end

adau1761_init codec_init_inst (
    .clk           (sys_clk),
    .rst           (reset_pipe[3]),
    .sda_in        (i2c_sda_feedback),
    .scl_drive_low (i2c_scl_drive_low),
    .sda_drive_low (i2c_sda_drive_low),
    .busy          (codec_init_busy),
    .done          (codec_init_done),
    .error         (codec_init_error),
    .debug_command_index(codec_debug_command)
);

// Explicit pad primitives preserve open-drain enables when this module is
// synthesized out of context inside the PS block design.
IOBUF codec_scl_iobuf (
    .I (1'b0),
    .T (~i2c_scl_drive_low),
    .O (),
    .IO(audio_i2c_scl)
);

IOBUF codec_sda_iobuf (
    .I (1'b0),
    .T (~i2c_sda_drive_low),
    .O (i2c_sda_feedback),
    .IO(audio_i2c_sda)
);
assign codec_addr    = 2'b11;
assign stream_reset  = reset_pipe[3] | !codec_init_done | codec_init_error;
assign analysis_pcm_sample = pcm_output_sample;
assign analysis_pcm_valid = pcm_output_valid;
assign analysis_reset = stream_reset;

// Count bit clocks in each LRCLK half-frame on the actual board.
always @(posedge audio_bclk) begin
    if (stream_reset) begin
        lrclk_previous    <= audio_lrclk;
        channel_bit_count <= 8'd0;
        bclk_per_channel  <= 8'd0;
    end else begin
        lrclk_previous <= audio_lrclk;
        if (audio_lrclk != lrclk_previous) begin
            bclk_per_channel  <= channel_bit_count + 1'b1;
            channel_bit_count <= 8'd0;
        end else if (channel_bit_count != 8'hff) begin
            channel_bit_count <= channel_bit_count + 1'b1;
        end
    end
end

// Raw loopback isolates the codec input path from PCM conversion/processing.
assign audio_sdata_out = gain_sw ? (source_sw ? stream_sdata_out : audio_sdata_in)
                                 : (source_sw ? decoded_sdata_out : stream_sdata_out);

pynq_audio_stream_top stream_inst (
    .sys_clk               (sys_clk),
    .rst                   (stream_reset),
    .use_test_tone         (!source_sw),
    .audio_bclk            (audio_bclk),
    .audio_lrclk           (audio_lrclk),
    .audio_sdata_in        (audio_sdata_in),
    .audio_sdata_out       (stream_sdata_out),
    .audio_sdata_decoded   (decoded_sdata_out),
    .gain_shift            (3'd0),
    .pcm_input_sample      (pcm_input_sample),
    .pcm_input_valid       (pcm_input_valid),
    .pcm_output_sample     (pcm_output_sample),
    .pcm_output_valid      (pcm_output_valid),
    .peak_level            (peak_level),
    .signal_active         (signal_active),
    .right_signal_active   (right_signal_active),
    .frequency_x100        (frequency_x100),
    .pitch_valid           (pitch_valid),
    .note_valid            (note_valid),
    .midi_note             (midi_note),
    .note_class            (note_class),
    .note_octave           (note_octave),
    .target_frequency_x100 (target_frequency_x100),
    .frequency_error_x100  (frequency_error_x100),
    .frame_ready           (frame_ready),
    .frame_number          (frame_number),
    .frame_read_enable     (1'b0),
    .frame_read_address    (10'd0),
    .frame_read_data       (frame_read_data)
);

// In test-tone mode LEDs 2/3 latch clock loss/button reset, respectively.
// In full processing mode LEDs 1/2 show right/left channel activity.
// During an I2C error LED2:LED0 show the low three bits of the command index.
// This keeps board diagnostics available without adding an ILA debug core.
// In decoded-loopback mode, LEDs 0/1/2 report 32/16/other BCLKs per channel.
assign led[0] = codec_init_error ? codec_debug_command[0] :
                (source_sw && !gain_sw) ? (bclk_per_channel == 8'd32) : mclk_locked;
assign led[1] = codec_init_error ? codec_debug_command[1] :
                (source_sw && !gain_sw) ? (bclk_per_channel == 8'd16) :
                (source_sw && gain_sw) ? right_signal_active :
                                         (codec_init_done && !codec_init_error);
assign led[2] = codec_init_error ? codec_debug_command[2] :
                (!source_sw && !gain_sw) ? clock_loss_seen :
                (source_sw && !gain_sw) ?
                    ((bclk_per_channel != 8'd32) &&
                     (bclk_per_channel != 8'd16)) : signal_active;
assign led[3] = (!source_sw && !gain_sw) ? button_reset_seen : codec_init_error;

endmodule
