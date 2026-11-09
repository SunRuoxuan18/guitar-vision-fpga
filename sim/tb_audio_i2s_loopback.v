`timescale 1ns / 1ps

module tb_audio_i2s_loopback;

localparam integer SAMPLE_WIDTH = 24;
localparam integer SLOT_WIDTH   = 32;

reg rst;
reg bclk;
reg lrclk;
reg signed [SAMPLE_WIDTH-1:0] left_sample;
reg signed [SAMPLE_WIDTH-1:0] right_sample;
wire sdata;
wire signed [SAMPLE_WIDTH-1:0] rx_left;
wire signed [SAMPLE_WIDTH-1:0] rx_right;
wire sample_valid;
integer error_count;
integer valid_count;

audio_i2s_tx #(
    .SAMPLE_WIDTH(SAMPLE_WIDTH),
    .SLOT_WIDTH(SLOT_WIDTH)
) tx_inst (
    .rst(rst),
    .bclk(bclk),
    .lrclk(lrclk),
    .left_sample(left_sample),
    .right_sample(right_sample),
    .sdata_out(sdata)
);

audio_i2s_rx #(
    .SAMPLE_WIDTH(SAMPLE_WIDTH),
    .SLOT_WIDTH(SLOT_WIDTH)
) rx_inst (
    .rst(rst),
    .bclk(bclk),
    .lrclk(lrclk),
    .sdata_in(sdata),
    .left_sample(rx_left),
    .right_sample(rx_right),
    .sample_valid(sample_valid)
);

always #5 bclk = ~bclk;

task run_slot;
    input slot_level;
    begin
        // Change LRCLK just after a falling edge, as the codec does.
        @(negedge bclk);
        #1 lrclk = slot_level;
        repeat (SLOT_WIDTH) @(negedge bclk);
    end
endtask

initial begin
    rst = 1'b1;
    bclk = 1'b0;
    lrclk = 1'b1;
    left_sample = 24'sh123456;
    right_sample = 24'shA5A5A5;
    error_count = 0;
    valid_count = 0;

    repeat (3) @(posedge bclk);
    rst = 1'b0;

    run_slot(1'b0);
    run_slot(1'b1);

    @(posedge bclk);
    #1;
    if (rx_left !== left_sample) begin
        error_count = error_count + 1;
        $display("ERROR: left=%h expected=%h", rx_left, left_sample);
    end
    if (rx_right !== right_sample) begin
        error_count = error_count + 1;
        $display("ERROR: right=%h expected=%h", rx_right, right_sample);
    end

    if (error_count == 0)
        $display("PASS: I2S TX/RX loopback verified.");
    else
        $display("FAIL: %0d I2S check(s) failed.", error_count);
    $finish;
end

always @(posedge bclk) begin
    if (sample_valid)
        valid_count = valid_count + 1;
end

endmodule
