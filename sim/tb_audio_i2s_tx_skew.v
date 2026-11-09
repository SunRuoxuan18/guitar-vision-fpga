`timescale 1ns / 1ps

// Check actual wire bits, independently of the project's I2S receiver.
module tb_audio_i2s_tx_skew;
    reg bclk = 1'b0;
    reg rst = 1'b1;
    reg lrclk = 1'b1;
    reg signed [23:0] left_sample = 24'h13579b;
    reg signed [23:0] right_sample = 24'ha468c2;
    wire sdata_out;
    integer skew;
    integer cycle;
    integer slot;
    integer errors = 0;
    integer checked = 0;
    reg expected_bit;

    always #50 bclk = ~bclk;

    audio_i2s_tx dut (
        .rst(rst), .bclk(bclk), .lrclk(lrclk),
        .left_sample(left_sample), .right_sample(right_sample),
        .sdata_out(sdata_out)
    );

    initial begin
        // LRCLK may arrive slightly before or after BCLK at the FPGA pads.
        for (skew = -2; skew <= 2; skew = skew + 1) begin
            rst = 1'b1;
            lrclk = 1'b1;
            repeat (4) @(posedge bclk);
            @(negedge bclk);
            #10 rst = 1'b0;
            @(posedge bclk);
            #1;
            for (cycle = 0; cycle < 8 * 64; cycle = cycle + 1) begin
                #(49 + skew);
                slot = cycle % 32;
                lrclk = ((cycle % 64) >= 32);
                @(posedge bclk);
                #1;
                expected_bit = 1'b0;
                if (slot >= 1 && slot <= 24)
                    expected_bit = lrclk ? right_sample[24-slot] : left_sample[24-slot];
                if (cycle >= 64) begin
                    checked = checked + 1;
                    if (sdata_out !== expected_bit) begin
                        if (errors < 12)
                            $display("ERROR: skew=%0d cycle=%0d slot=%0d bit=%b expected=%b",
                                     skew, cycle, slot, sdata_out, expected_bit);
                        errors = errors + 1;
                    end
                end
            end
        end
        if (errors == 0)
            $display("PASS: %0d independent I2S wire bits, LRCLK skew -2..+2 ns.", checked);
        else
            $display("FAIL: %0d of %0d wire bits misaligned with LRCLK skew.", errors, checked);
        $finish;
    end

    initial begin
        #1000000;
        $display("FAIL: I2S skew test timed out.");
        $finish;
    end
endmodule
