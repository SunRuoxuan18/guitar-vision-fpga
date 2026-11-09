`timescale 1ns / 1ps

module tb_audio_stereo_tone;

localparam integer SLOT_BITS = 32;
localparam integer SAMPLE_BITS = 24;
localparam integer WARMUP_FRAMES = 8;
localparam integer CHECK_FRAMES = 128;
localparam real BCLK_HALF_PERIOD_NS = 1.0e9 / (2.0 * 3_072_000.0);

reg sys_clk = 1'b0;
reg bclk = 1'b0;
reg lrclk = 1'b1;
reg rst = 1'b1;
reg serial_started = 1'b0;
integer slot_bit = 0;

wire sdata_out;
wire signed [15:0] pcm_output_sample;
wire pcm_output_valid;

reg signed [15:0] latest_pcm = 16'sd0;
reg [23:0] frame_reference = 24'd0;
reg [23:0] serial_word = 24'd0;
reg [23:0] left_word = 24'd0;
reg [23:0] right_word = 24'd0;
reg [23:0] previous_left = 24'd0;
reg [23:0] previous_right = 24'd0;
integer frames_seen = 0;
integer left_nonzero = 0;
integer right_nonzero = 0;
integer left_positive = 0;
integer right_positive = 0;
integer left_negative = 0;
integer right_negative = 0;
integer left_changes = 0;
integer right_changes = 0;

always #4 sys_clk = ~sys_clk;
always #(BCLK_HALF_PERIOD_NS) bclk = ~bclk;

// Keep the DUT's default 125 MHz / 48 kHz parameters.
pynq_audio_stream_top dut (
    .sys_clk(sys_clk),
    .rst(rst),
    .use_test_tone(1'b1),
    .audio_bclk(bclk),
    .audio_lrclk(lrclk),
    .audio_sdata_in(1'b0),
    .audio_sdata_out(sdata_out),
    .gain_shift(3'd0),
    .pcm_output_sample(pcm_output_sample),
    .pcm_output_valid(pcm_output_valid),
    .frame_read_enable(1'b0),
    .frame_read_address(10'd0)
);

always @(posedge sys_clk) begin
    if (rst)
        latest_pcm = 16'sd0;
    else if (pcm_output_valid)
        latest_pcm = pcm_output_sample;
end

initial begin
    // Reset spans both clock domains. Start each frame after the tone/CDC
    // update, so a new mono sample cannot fall between channel latches.
    repeat (8) @(negedge bclk);
    #1 rst = 1'b0;
    repeat (12) @(negedge bclk);
    #1;
    lrclk = 1'b0;
    serial_started = 1'b1;

    forever begin
        @(negedge bclk);
        #1;
        if (slot_bit == SLOT_BITS - 1) begin
            slot_bit = 0;
            lrclk = ~lrclk;
        end else begin
            slot_bit = slot_bit + 1;
        end
    end
end

// Independent sink: slot 0 is the I2S delay; slots 1..24 carry MSB first.
// No DUT receiver, shift counter, or transmit internals are used as an oracle.
always @(posedge bclk) begin
    if (!rst && serial_started) begin
        if (sdata_out !== 1'b0 && sdata_out !== 1'b1)
            $fatal(1, "Unknown serial bit: frame=%0d channel=%0d bit=%0d",
                   frames_seen, lrclk, slot_bit);

        if (slot_bit == 0) begin
            serial_word = 24'd0;
            if (!lrclk)
                frame_reference = {latest_pcm, 8'b0};
        end else if (slot_bit <= SAMPLE_BITS) begin
            serial_word = {serial_word[22:0], sdata_out};
            if (slot_bit == SAMPLE_BITS) begin
                if (lrclk)
                    right_word = serial_word;
                else
                    left_word = serial_word;
            end
        end else if (sdata_out !== 1'b0) begin
            $fatal(1, "Nonzero I2S padding: frame=%0d channel=%0d bit=%0d",
                   frames_seen, lrclk, slot_bit);
        end

        if (lrclk && slot_bit == SLOT_BITS - 1) begin
            if (frames_seen >= WARMUP_FRAMES) begin
                if (left_word !== right_word)
                    $fatal(1, "Mono mismatch: frame=%0d left=%h right=%h",
                           frames_seen, left_word, right_word);
                if (left_word !== frame_reference || right_word !== frame_reference)
                    $fatal(1, "PCM/serial mismatch: frame=%0d left=%h right=%h expected=%h",
                           frames_seen, left_word, right_word, frame_reference);

                if (left_word != 24'd0) left_nonzero = left_nonzero + 1;
                if (right_word != 24'd0) right_nonzero = right_nonzero + 1;
                if ($signed(left_word) > 0) left_positive = left_positive + 1;
                if ($signed(right_word) > 0) right_positive = right_positive + 1;
                if ($signed(left_word) < 0) left_negative = left_negative + 1;
                if ($signed(right_word) < 0) right_negative = right_negative + 1;
                if (left_word !== previous_left) left_changes = left_changes + 1;
                if (right_word !== previous_right) right_changes = right_changes + 1;
                previous_left = left_word;
                previous_right = right_word;
            end

            frames_seen = frames_seen + 1;
            if (frames_seen == WARMUP_FRAMES + CHECK_FRAMES) begin
                if (left_nonzero < CHECK_FRAMES / 2 ||
                    right_nonzero < CHECK_FRAMES / 2 ||
                    left_positive == 0 || right_positive == 0 ||
                    left_negative == 0 || right_negative == 0 ||
                    left_changes < 8 || right_changes < 8)
                    $fatal(1, "Missing stereo tone activity: L/R nonzero=%0d/%0d positive=%0d/%0d negative=%0d/%0d changes=%0d/%0d",
                           left_nonzero, right_nonzero, left_positive, right_positive,
                           left_negative, right_negative, left_changes, right_changes);
                $display("PASS: %0d mono tone frames, both channels equal PCM; L/R nonzero=%0d/%0d, 32 BCLKs/channel at 48 kHz.",
                         CHECK_FRAMES, left_nonzero, right_nonzero);
                $finish;
            end
        end
    end
end

initial begin
    #5_000_000;
    $fatal(1, "Stereo tone timeout: completed %0d of %0d frames",
           frames_seen, WARMUP_FRAMES + CHECK_FRAMES);
end

endmodule
