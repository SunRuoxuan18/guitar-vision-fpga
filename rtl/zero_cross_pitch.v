`timescale 1ns / 1ps

module zero_cross_pitch #(
    parameter integer SAMPLE_RATE_HZ = 48_000,
    parameter integer MIN_FREQUENCY_HZ = 60,
    parameter integer MAX_FREQUENCY_HZ = 1200,
    parameter [15:0]  CROSS_THRESHOLD = 16'd500
)(
    input  wire                     clk,
    input  wire                     rst,
    input  wire                     sample_valid,
    input  wire signed [15:0]       sample_in,
    output reg  [31:0]              frequency_x100,
    output reg                      pitch_valid
);

localparam integer MIN_PERIOD = SAMPLE_RATE_HZ / MAX_FREQUENCY_HZ;
localparam integer MAX_PERIOD = SAMPLE_RATE_HZ / MIN_FREQUENCY_HZ;

reg [15:0] period_counter;
reg        negative_seen;
reg        have_reference_crossing;

always @(posedge clk) begin
    if (rst) begin
        period_counter          <= 16'd0;
        negative_seen           <= 1'b0;
        have_reference_crossing <= 1'b0;
        frequency_x100          <= 32'd0;
        pitch_valid             <= 1'b0;
    end else begin
        pitch_valid <= 1'b0;

        if (sample_valid) begin
            if (period_counter < 16'hffff)
                period_counter <= period_counter + 1'b1;

            if ($signed(sample_in) <= -$signed({1'b0, CROSS_THRESHOLD}))
                negative_seen <= 1'b1;

            if (negative_seen &&
                ($signed(sample_in) >= $signed({1'b0, CROSS_THRESHOLD}))) begin
                negative_seen <= 1'b0;

                if (have_reference_crossing &&
                    (period_counter >= MIN_PERIOD) &&
                    (period_counter <= MAX_PERIOD)) begin
                    frequency_x100 <= (SAMPLE_RATE_HZ * 100) / period_counter;
                    pitch_valid    <= 1'b1;
                end

                have_reference_crossing <= 1'b1;
                period_counter          <= 16'd0;
            end

            if (period_counter > MAX_PERIOD + 16) begin
                have_reference_crossing <= 1'b0;
                period_counter          <= 16'd0;
            end
        end
    end
end

endmodule
