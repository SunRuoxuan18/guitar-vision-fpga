`timescale 1ns / 1ps

module tb_adau1761_init;

reg clk;
reg rst;
integer transaction_count;
integer timeout_count;
integer error_count;

wire ok_scl_low;
wire ok_sda_low;
wire ok_busy;
wire ok_done;
wire ok_error;

wire fail_scl_low;
wire fail_sda_low;
wire fail_busy;
wire fail_done;
wire fail_error;

always #4 clk = ~clk;

adau1761_init #(
    .CLK_FREQ_HZ(125_000_000),
    .I2C_FREQ_HZ(5_000_000),
    .STARTUP_WAIT_CYCLES(4),
    .PLL_WAIT_CYCLES(8),
    .CORE_WAIT_CYCLES(4)
) dut_ok (
    .clk(clk),
    .rst(rst),
    .sda_in(1'b0),
    .scl_drive_low(ok_scl_low),
    .sda_drive_low(ok_sda_low),
    .busy(ok_busy),
    .done(ok_done),
    .error(ok_error)
);

adau1761_init #(
    .CLK_FREQ_HZ(125_000_000),
    .I2C_FREQ_HZ(5_000_000),
    .STARTUP_WAIT_CYCLES(4),
    .PLL_WAIT_CYCLES(8),
    .CORE_WAIT_CYCLES(4)
) dut_fail (
    .clk(clk),
    .rst(rst),
    .sda_in(1'b1),
    .scl_drive_low(fail_scl_low),
    .sda_drive_low(fail_sda_low),
    .busy(fail_busy),
    .done(fail_done),
    .error(fail_error)
);

always @(posedge clk) begin
    if (!rst && dut_ok.master_start) begin
        transaction_count = transaction_count + 1;
        if ((dut_ok.command_index == 1) &&
            ((dut_ok.command_payload !== 64'h4002_0000_0000_0000) ||
             (dut_ok.command_length !== 4'd8)))
            $fatal(1, "PLL must be disabled with a complete six-byte write");
        if ((dut_ok.command_index == 2) &&
            ((dut_ok.command_payload !== 64'h4002_0271_023C_2101) ||
             (dut_ok.command_length !== 4'd8)))
            $fatal(1, "PLL enable parameters changed unexpectedly");
        case (dut_ok.command_index)
            24: if (dut_ok.command_payload !== 64'h4017_0000_0000_0000) $fatal(1, "Converter rate missing");
            25: if (dut_ok.command_payload !== 64'h4018_0000_0000_0000) $fatal(1, "Converter pair mapping missing");
            26: if (dut_ok.command_payload !== 64'h401A_0000_0000_0000) $fatal(1, "Left ADC volume missing");
            27: if (dut_ok.command_payload !== 64'h401B_0000_0000_0000) $fatal(1, "Right ADC volume missing");
            28: if (dut_ok.command_payload !== 64'h401D_0000_0000_0000) $fatal(1, "Left analog bypass must be muted");
            29: if (dut_ok.command_payload !== 64'h401F_0000_0000_0000) $fatal(1, "Right analog bypass must be muted");
            30: if (dut_ok.command_payload !== 64'h402B_0000_0000_0000) $fatal(1, "Left DAC volume missing");
            31: if (dut_ok.command_payload !== 64'h402C_0000_0000_0000) $fatal(1, "Right DAC volume missing");
            32: if (dut_ok.command_payload !== 64'h40F4_0000_0000_0000) $fatal(1, "Serial pin mode missing");
            33: if (dut_ok.command_payload !== 64'h40F7_0000_0000_0000) $fatal(1, "Headphone volume control missing");
        endcase
    end
end

initial begin
    clk = 1'b0;
    rst = 1'b1;
    transaction_count = 0;
    timeout_count = 0;
    error_count = 0;

    repeat (5) @(posedge clk);
    rst = 1'b0;

    while ((!ok_done || !fail_done) && (timeout_count < 100000)) begin
        @(posedge clk);
        timeout_count = timeout_count + 1;
    end

    if (!ok_done || ok_error || (transaction_count != 34)) begin
        error_count = error_count + 1;
        $display("ERROR: ACK path done=%0d error=%0d transactions=%0d",
                 ok_done, ok_error, transaction_count);
    end

    if (!fail_done || !fail_error) begin
        error_count = error_count + 1;
        $display("ERROR: NACK path done=%0d error=%0d",
                 fail_done, fail_error);
    end

    // Reinitialize an already configured codec; repeat the full sequence.
    @(negedge clk);
    rst = 1'b1;
    transaction_count = 0;
    timeout_count = 0;
    repeat (5) @(negedge clk);
    rst = 1'b0;
    while ((!ok_done || !fail_done) && (timeout_count < 100000)) begin
        @(posedge clk);
        #1;
        timeout_count = timeout_count + 1;
    end
    if (!ok_done || ok_error || !fail_done || !fail_error ||
        transaction_count != 34)
        $fatal(1, "Warm initialization or NACK path failed");

    if (error_count == 0)
        $display("PASS: ADAU1761 initialization sequence verified.");
    else
        $fatal(1, "FAIL: %0d ADAU1761 initialization check(s) failed.",
                 error_count);

    $finish;
end

endmodule
