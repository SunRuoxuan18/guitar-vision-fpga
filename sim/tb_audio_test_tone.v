`timescale 1ns / 1ps

module tb_audio_test_tone;

localparam integer CLK_PERIOD_NS = 10;

reg clk;
reg rst;
reg sample_tick;
wire signed [15:0] pcm_sample;
wire sample_valid;

integer sample_count;
integer positive_count;
integer negative_count;
integer zero_cross_count;
integer max_sample;
integer min_sample;
integer error_count;
reg signed [15:0] previous_sample;

audio_test_tone dut (
    .clk          (clk),
    .rst          (rst),
    .sample_tick  (sample_tick),
    .pcm_sample   (pcm_sample),
    .sample_valid (sample_valid)
);

always #(CLK_PERIOD_NS / 2) clk = ~clk;

// The testbench produces one sample request every clock so that the
// waveform can be checked quickly. On the board this input comes from
// audio_sample_tick at 48 kHz.
always @(negedge clk) begin
    if (rst)
        sample_tick = 1'b0;
    else
        sample_tick = 1'b1;
end

initial begin
    clk              = 1'b0;
    rst              = 1'b1;
    sample_tick      = 1'b0;
    sample_count     = 0;
    positive_count   = 0;
    negative_count   = 0;
    zero_cross_count = 0;
    max_sample       = -32768;
    min_sample       = 32767;
    error_count      = 0;
    previous_sample  = 16'sd0;

    repeat (5) @(posedge clk);
    rst = 1'b0;

    while (sample_count < 480) begin
        @(posedge clk);
        #1;

        if (sample_valid) begin
            sample_count = sample_count + 1;

            if ($signed(pcm_sample) > 0)
                positive_count = positive_count + 1;
            if ($signed(pcm_sample) < 0)
                negative_count = negative_count + 1;
            if ($signed(pcm_sample) > max_sample)
                max_sample = $signed(pcm_sample);
            if ($signed(pcm_sample) < min_sample)
                min_sample = $signed(pcm_sample);

            if (($signed(previous_sample) <= 0) && ($signed(pcm_sample) > 0))
                zero_cross_count = zero_cross_count + 1;

            previous_sample = pcm_sample;
        end
    end

    if (positive_count == 0 || negative_count == 0) begin
        error_count = error_count + 1;
        $display("ERROR: waveform did not contain both half cycles");
    end

    if (max_sample < 16000 || min_sample > -16000) begin
        error_count = error_count + 1;
        $display("ERROR: amplitude range is too small: min=%0d max=%0d",
                 min_sample, max_sample);
    end

    if ((zero_cross_count < 4) || (zero_cross_count > 5)) begin
        error_count = error_count + 1;
        $display("ERROR: unexpected positive zero crossings=%0d",
                 zero_cross_count);
    end

    if (error_count == 0) begin
        $display("PASS: 440 Hz PCM test tone verified.");
        $display("INFO: samples=%0d min=%0d max=%0d crossings=%0d",
                 sample_count, min_sample, max_sample, zero_cross_count);
    end else begin
        $display("FAIL: %0d check(s) failed.", error_count);
    end

    $finish;
end

endmodule
