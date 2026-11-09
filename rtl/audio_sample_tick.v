`timescale 1ns / 1ps

module audio_sample_tick #(
    parameter integer CLK_FREQ_HZ   = 125_000_000,
    parameter integer SAMPLE_RATE_HZ = 48_000
)(
    input  wire clk,
    input  wire rst,
    output reg  sample_tick
);

reg [31:0] accumulator;

always @(posedge clk) begin
    if (rst) begin
        accumulator <= 32'd0;
        sample_tick <= 1'b0;
    end else if (accumulator >= CLK_FREQ_HZ - SAMPLE_RATE_HZ) begin
        accumulator <= accumulator + SAMPLE_RATE_HZ - CLK_FREQ_HZ;
        sample_tick <= 1'b1;
    end else begin
        accumulator <= accumulator + SAMPLE_RATE_HZ;
        sample_tick <= 1'b0;
    end
end

endmodule
