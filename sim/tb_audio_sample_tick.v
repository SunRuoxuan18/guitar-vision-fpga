`timescale 1ns / 1ps

module tb_audio_sample_tick;

localparam integer CLK_FREQ_HZ    = 125_000_000;
localparam integer SAMPLE_RATE_HZ = 48_000;
localparam integer CLK_PERIOD_NS  = 8;

reg  clk;
reg  rst;
wire sample_tick;

integer cycle_count;
integer last_tick_cycle;
integer interval_cycles;
integer tick_count;
integer error_count;

audio_sample_tick #(
    .CLK_FREQ_HZ    (CLK_FREQ_HZ),
    .SAMPLE_RATE_HZ (SAMPLE_RATE_HZ)
) dut (
    .clk         (clk),
    .rst         (rst),
    .sample_tick (sample_tick)
);

always #(CLK_PERIOD_NS / 2) clk = ~clk;

always @(posedge clk) begin
    if (rst) begin
        cycle_count <= 0;
    end else begin
        cycle_count <= cycle_count + 1;
    end
end

initial begin
    clk             = 1'b0;
    rst             = 1'b1;
    cycle_count     = 0;
    last_tick_cycle = 0;
    interval_cycles = 0;
    tick_count      = 0;
    error_count     = 0;

    repeat (5) @(posedge clk);
    rst = 1'b0;

    while (tick_count < 12) begin
        @(posedge clk);
        #1;

        if (sample_tick) begin
            tick_count = tick_count + 1;

            if (tick_count > 1) begin
                interval_cycles = cycle_count - last_tick_cycle;
                if ((interval_cycles != 2604) && (interval_cycles != 2605)) begin
                    error_count = error_count + 1;
                    $display("ERROR: interval=%0d clocks", interval_cycles);
                end
            end

            last_tick_cycle = cycle_count;
            $display("INFO: sample_tick %0d at cycle %0d", tick_count, cycle_count);

            @(posedge clk);
            #1;
            if (sample_tick !== 1'b0) begin
                error_count = error_count + 1;
                $display("ERROR: sample_tick lasted longer than one clock");
            end
        end
    end

    if (error_count == 0) begin
        $display("PASS: 48 kHz sample tick verified.");
    end else begin
        $display("FAIL: %0d check(s) failed.", error_count);
    end

    $finish;
end

endmodule
