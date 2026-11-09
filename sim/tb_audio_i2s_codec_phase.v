`timescale 1ns / 1ps

// The codec changes LRCLK and serial data just after a falling BCLK edge.
// A transmit design must still drive each sample in the correct channel slot.
module tb_audio_i2s_codec_phase;
    reg bclk = 1'b0;
    reg rst = 1'b1;
    reg lrclk = 1'b1;
    reg sdata_in = 1'b0;
    wire signed [23:0] left_sample;
    wire signed [23:0] right_sample;
    wire sample_valid;
    wire sdata_out;
    wire signed [23:0] output_left;
    wire signed [23:0] output_right;
    wire output_valid;
    integer slot;
    integer cycle;
    integer errors = 0;

    localparam [23:0] LEFT_WORD = 24'h13579B;
    localparam [23:0] RIGHT_WORD = 24'hA468C2;

    always #5 bclk = ~bclk;

    audio_i2s_rx rx (
        .rst(rst), .bclk(bclk), .lrclk(lrclk), .sdata_in(sdata_in),
        .left_sample(left_sample), .right_sample(right_sample),
        .sample_valid(sample_valid)
    );

    audio_i2s_tx tx (
        .rst(rst), .bclk(bclk), .lrclk(lrclk),
        .left_sample(left_sample), .right_sample(right_sample),
        .sdata_out(sdata_out)
    );

    audio_i2s_rx output_rx (
        .rst(rst), .bclk(bclk), .lrclk(lrclk), .sdata_in(sdata_out),
        .left_sample(output_left), .right_sample(output_right),
        .sample_valid(output_valid)
    );

    initial begin
        repeat (4) @(posedge bclk);
        @(negedge bclk);
        rst = 1'b0;

        for (cycle = 0; cycle < 8 * 64; cycle = cycle + 1) begin
            @(negedge bclk);
            #1;
            slot = cycle % 32;
            lrclk = ((cycle % 64) >= 32);
            if (slot == 0 || slot > 24)
                sdata_in = 1'b0;
            else if (lrclk)
                sdata_in = RIGHT_WORD[24 - slot];
            else
                sdata_in = LEFT_WORD[24 - slot];

            @(posedge bclk);
            #1;
            if (cycle >= 2 * 64 && (cycle % 64) == 26) begin
                if (left_sample !== LEFT_WORD || output_left !== LEFT_WORD) begin
                    errors = errors + 1;
                    $display("LEFT cycle=%0d input=%h output=%h expected=%h",
                             cycle, left_sample, output_left, LEFT_WORD);
                end
            end
            if (cycle >= 2 * 64 && (cycle % 64) == 58) begin
                if (right_sample !== RIGHT_WORD || output_right !== RIGHT_WORD) begin
                    errors = errors + 1;
                    $display("RIGHT cycle=%0d input=%h output=%h expected=%h",
                             cycle, right_sample, output_right, RIGHT_WORD);
                end
            end
        end

        if (errors == 0)
            $display("PASS: codec-edge I2S decode and encode remain aligned.");
        else
            $display("FAIL: %0d codec-edge I2S mismatches.", errors);
        $finish;
    end
endmodule
