`timescale 1ns / 1ps

module tb_pynq_audio_stream_top;

localparam integer SAMPLE_WIDTH = 24;
localparam integer SLOT_WIDTH   = 32;

reg sys_clk;
reg bclk;
reg lrclk;
reg rst;
reg tone_tick;
reg [2:0] gain_shift;

wire signed [15:0] tone_pcm;
wire tone_valid_unused;
wire signed [23:0] source_sample = {tone_pcm, 8'b0};
wire source_sdata;
wire output_sdata;

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
reg frame_read_enable;
reg [9:0] frame_read_address;
wire signed [15:0] frame_read_data;

wire signed [23:0] sink_left;
wire signed [23:0] sink_right;
wire sink_valid;

integer frame_index;
integer input_count;
integer output_count;
integer note_count;
integer nonzero_output_count;
integer sink_count;
integer zero_output_run;
integer longest_zero_output_run;
integer input_gap;
integer output_gap;
integer longest_input_gap;
integer longest_output_gap;
integer frequency_error;
integer error_count;

always #2 sys_clk = ~sys_clk;
always #5 bclk = ~bclk;

audio_test_tone tone_source (
    .clk          (bclk),
    .rst          (rst),
    .sample_tick  (tone_tick),
    .pcm_sample   (tone_pcm),
    .sample_valid (tone_valid_unused)
);

audio_i2s_tx #(
    .SAMPLE_WIDTH(SAMPLE_WIDTH),
    .SLOT_WIDTH  (SLOT_WIDTH)
) source_tx (
    .rst         (rst),
    .bclk        (bclk),
    .lrclk       (lrclk),
    .left_sample (source_sample),
    .right_sample(source_sample),
    .sdata_out   (source_sdata)
);

pynq_audio_stream_top #(
    .CLK_FREQ_HZ(125_000_000),
    .SAMPLE_RATE_HZ(48_000),
    .FRAME_SAMPLES(1024),
    .FRAME_ADDR_WIDTH(10),
    .LEVEL_WINDOW_SAMPLES(1024)
) dut (
    .sys_clk(sys_clk),
    .rst(rst),
    .use_test_tone(1'b0),
    .audio_bclk(bclk),
    .audio_lrclk(lrclk),
    .audio_sdata_in(source_sdata),
    .audio_sdata_out(output_sdata),
    .gain_shift(gain_shift),
    .pcm_input_sample(pcm_input_sample),
    .pcm_input_valid(pcm_input_valid),
    .pcm_output_sample(pcm_output_sample),
    .pcm_output_valid(pcm_output_valid),
    .peak_level(peak_level),
    .signal_active(signal_active),
    .right_signal_active(right_signal_active),
    .frequency_x100(frequency_x100),
    .pitch_valid(pitch_valid),
    .note_valid(note_valid),
    .midi_note(midi_note),
    .note_class(note_class),
    .note_octave(note_octave),
    .target_frequency_x100(target_frequency_x100),
    .frequency_error_x100(frequency_error_x100),
    .frame_ready(frame_ready),
    .frame_number(frame_number),
    .frame_read_enable(frame_read_enable),
    .frame_read_address(frame_read_address),
    .frame_read_data(frame_read_data)
);

audio_i2s_rx #(
    .SAMPLE_WIDTH(SAMPLE_WIDTH),
    .SLOT_WIDTH  (SLOT_WIDTH)
) output_rx (
    .rst(rst),
    .bclk(bclk),
    .lrclk(lrclk),
    .sdata_in(output_sdata),
    .left_sample(sink_left),
    .right_sample(sink_right),
    .sample_valid(sink_valid)
);

task run_slot;
    input slot_level;
    integer bit_index;
    begin
        // Match the codec: LRCLK changes after a falling edge. The serial
        // transmitter sees that change on the following falling edge.
        @(negedge bclk);
        #1 lrclk = slot_level;
        for (bit_index = 0; bit_index < SLOT_WIDTH; bit_index = bit_index + 1)
            @(negedge bclk);
    end
endtask

// Prepare the next PCM sample during the right-channel slot so that it is
// stable before the following left-channel slot begins.
always @(posedge lrclk) begin
    if (!rst) begin
        tone_tick = 1'b1;
        @(posedge bclk);
        #1 tone_tick = 1'b0;
    end
end

always @(posedge sys_clk) begin
    if (!rst) begin
        input_gap = input_gap + 1;
        output_gap = output_gap + 1;
        if (pcm_input_valid) begin
            input_count = input_count + 1;
            if (input_count > 2 && input_gap > longest_input_gap)
                longest_input_gap = input_gap;
            input_gap = 0;
        end
        if (pcm_output_valid) begin
            output_count = output_count + 1;
            if (output_count > 2 && output_gap > longest_output_gap)
                longest_output_gap = output_gap;
            output_gap = 0;
            if (pcm_output_sample !== pcm_input_sample) begin
                error_count = error_count + 1;
                $display("ERROR: PCM mismatch input=%0d output=%0d",
                         pcm_input_sample, pcm_output_sample);
            end
        end
        if (note_valid)
            note_count = note_count + 1;
    end
end

always @(posedge bclk) begin
    if (!rst && sink_valid) begin
        sink_count = sink_count + 1;
        if (sink_left != 24'sd0) begin
            nonzero_output_count = nonzero_output_count + 1;
            zero_output_run = 0;
        end else if (sink_count > 8) begin
            zero_output_run = zero_output_run + 1;
            if (zero_output_run > longest_zero_output_run)
                longest_zero_output_run = zero_output_run;
        end
    end
end

initial begin
    sys_clk = 1'b0;
    bclk = 1'b0;
    lrclk = 1'b1;
    rst = 1'b1;
    tone_tick = 1'b0;
    gain_shift = 3'd0;
    frame_read_enable = 1'b0;
    frame_read_address = 10'd0;
    input_count = 0;
    output_count = 0;
    note_count = 0;
    nonzero_output_count = 0;
    sink_count = 0;
    zero_output_run = 0;
    longest_zero_output_run = 0;
    input_gap = 0;
    output_gap = 0;
    longest_input_gap = 0;
    longest_output_gap = 0;
    frequency_error = 0;
    error_count = 0;

    repeat (8) @(posedge bclk);
    rst = 1'b0;

    for (frame_index = 0; frame_index < 1300; frame_index = frame_index + 1) begin
        run_slot(1'b0);
        run_slot(1'b1);
    end

    repeat (20) @(posedge sys_clk);

    frequency_error = frequency_x100 - 44000;
    if (frequency_error < 0)
        frequency_error = -frequency_error;

    if (input_count != 1300 || output_count != input_count) begin
        error_count = error_count + 1;
        $display("ERROR: input_count=%0d output_count=%0d",
                 input_count, output_count);
    end

    if (sink_count < 1290 || longest_zero_output_run > 4) begin
        error_count = error_count + 1;
        $display("ERROR: sink_count=%0d longest_zero_output_run=%0d",
                 sink_count, longest_zero_output_run);
    end
    if (longest_input_gap > 170 || longest_output_gap > 170) begin
        error_count = error_count + 1;
        $display("ERROR: PCM event gaps input=%0d output=%0d sys_clk cycles",
                 longest_input_gap, longest_output_gap);
    end

    if (nonzero_output_count == 0) begin
        error_count = error_count + 1;
        $display("ERROR: no nonzero PCM returned through I2S output");
    end

    if (note_count == 0 || midi_note != 7'd69 ||
        note_class != 4'd9 || note_octave != 4'd4 ||
        frequency_error > 1500) begin
        error_count = error_count + 1;
        $display("ERROR: frequency=%0d midi=%0d class=%0d octave=%0d",
                 frequency_x100, midi_note, note_class, note_octave);
    end

    if (!signal_active || !right_signal_active || peak_level < 16'd15000) begin
        error_count = error_count + 1;
        $display("ERROR: peak=%0d left=%0d right=%0d",
                 peak_level, signal_active, right_signal_active);
    end

    if (error_count == 0) begin
        $display("PASS: complete PYNQ audio stream path verified.");
        $display("INFO: input=%0d output=%0d frequency=%0d.%02d Hz MIDI=%0d class=%0d octave=%0d",
                 input_count, output_count,
                 frequency_x100 / 100, frequency_x100 % 100,
                 midi_note, note_class, note_octave);
    end else begin
        $display("FAIL: %0d complete-stream check(s) failed.", error_count);
    end

    $finish;
end

endmodule
