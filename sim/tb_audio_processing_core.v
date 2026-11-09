`timescale 1ns / 1ps

module tb_audio_processing_core;

localparam integer CLK_FREQ_HZ = 4_800_000;
localparam integer SAMPLE_RATE_HZ = 48_000;
localparam integer CLK_PERIOD_NS = 10;

reg clk;
reg rst;
reg use_test_tone;
reg signed [15:0] external_sample;
reg external_sample_valid;
reg [2:0] gain_shift;
reg frame_read_enable;
reg [9:0] frame_read_address;

wire signed [15:0] processed_sample;
wire processed_sample_valid;
wire clipped;
wire [15:0] peak_level;
wire signal_active;
wire level_valid;
wire [31:0] frequency_x100;
wire pitch_valid;
wire note_busy;
wire note_valid;
wire [6:0] midi_note;
wire [3:0] note_class;
wire [3:0] note_octave;
wire [31:0] target_frequency_x100;
wire signed [31:0] frequency_error_x100;
wire frame_ready;
wire [31:0] frame_number;
wire signed [15:0] frame_read_data;

integer valid_sample_count;
integer frame_count;
integer pitch_count;
integer note_count;
integer level_count;
integer clip_count;
integer error_count;
integer timeout_count;
integer frequency_error;
reg signed [15:0] first_frame_sample;

audio_processing_core #(
    .CLK_FREQ_HZ        (CLK_FREQ_HZ),
    .SAMPLE_RATE_HZ     (SAMPLE_RATE_HZ),
    .FRAME_SAMPLES      (1024),
    .FRAME_ADDR_WIDTH   (10),
    .LEVEL_WINDOW_SAMPLES(1024)
) dut (
    .clk                  (clk),
    .rst                  (rst),
    .use_test_tone        (use_test_tone),
    .external_sample      (external_sample),
    .external_sample_valid(external_sample_valid),
    .gain_shift           (gain_shift),
    .processed_sample     (processed_sample),
    .processed_sample_valid(processed_sample_valid),
    .clipped              (clipped),
    .peak_level           (peak_level),
    .signal_active        (signal_active),
    .level_valid          (level_valid),
    .frequency_x100       (frequency_x100),
    .pitch_valid          (pitch_valid),
    .note_busy            (note_busy),
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

always #(CLK_PERIOD_NS / 2) clk = ~clk;

always @(posedge clk) begin
    if (!rst) begin
        if (processed_sample_valid)
            valid_sample_count = valid_sample_count + 1;
        if (frame_ready)
            frame_count = frame_count + 1;
        if (pitch_valid)
            pitch_count = pitch_count + 1;
        if (note_valid)
            note_count = note_count + 1;
        if (level_valid)
            level_count = level_count + 1;
        if (clipped)
            clip_count = clip_count + 1;
    end
end

initial begin
    clk                   = 1'b0;
    rst                   = 1'b1;
    use_test_tone         = 1'b1;
    external_sample       = 16'sd0;
    external_sample_valid = 1'b0;
    gain_shift            = 3'd0;
    frame_read_enable     = 1'b0;
    frame_read_address    = 10'd0;
    valid_sample_count    = 0;
    frame_count           = 0;
    pitch_count           = 0;
    note_count            = 0;
    level_count           = 0;
    clip_count            = 0;
    error_count           = 0;
    timeout_count         = 0;
    first_frame_sample    = 16'sd0;

    repeat (8) @(posedge clk);
    rst = 1'b0;

    while ((frame_count < 2) && (timeout_count < 250000)) begin
        @(posedge clk);
        timeout_count = timeout_count + 1;
    end

    if (frame_count < 2) begin
        error_count = error_count + 1;
        $display("ERROR: two audio frames were not captured");
    end

    if (valid_sample_count < 2048) begin
        error_count = error_count + 1;
        $display("ERROR: only %0d samples became valid", valid_sample_count);
    end

    if (pitch_count == 0) begin
        error_count = error_count + 1;
        $display("ERROR: no pitch result was produced");
    end

    if ((note_count == 0) || (midi_note != 7'd69) ||
        (note_class != 4'd9) || (note_octave != 4'd4)) begin
        error_count = error_count + 1;
        $display("ERROR: note decoder midi=%0d class=%0d octave=%0d",
                 midi_note, note_class, note_octave);
    end

    frequency_error = frequency_x100 - 44000;
    if (frequency_error < 0)
        frequency_error = -frequency_error;
    if (frequency_error > 1500) begin
        error_count = error_count + 1;
        $display("ERROR: detected frequency=%0d.%02d Hz",
                 frequency_x100 / 100, frequency_x100 % 100);
    end

    if ((level_count == 0) || !signal_active || (peak_level < 16000)) begin
        error_count = error_count + 1;
        $display("ERROR: level meter peak=%0d active=%0d",
                 peak_level, signal_active);
    end

    frame_read_address = 10'd0;
    frame_read_enable  = 1'b1;
    @(posedge clk);
    #1;
    frame_read_enable  = 1'b0;
    first_frame_sample = frame_read_data;

    // Increase gain by 4x. The limiter must clip instead of wrapping around.
    gain_shift = 3'd2;
    repeat (40000) @(posedge clk);

    if (clip_count == 0) begin
        error_count = error_count + 1;
        $display("ERROR: gain limiter never reported clipping");
    end

    if (error_count == 0) begin
        $display("PASS: complete audio processing core verified.");
        $display("INFO: samples=%0d frames=%0d pitch_results=%0d note_results=%0d",
                 valid_sample_count, frame_count, pitch_count, note_count);
        $display("INFO: frequency=%0d.%02d Hz peak=%0d clips=%0d bram[0]=%0d",
                 frequency_x100 / 100, frequency_x100 % 100,
                 peak_level, clip_count, first_frame_sample);
    end else begin
        $display("FAIL: %0d audio core check(s) failed.", error_count);
    end

    $finish;
end

endmodule
