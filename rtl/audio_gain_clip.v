`timescale 1ns / 1ps

module audio_gain_clip (
    input  wire                      clk,
    input  wire                      rst,
    input  wire                      sample_valid_in,
    input  wire signed [15:0]        sample_in,
    input  wire [2:0]                gain_shift,
    output reg                       sample_valid_out,
    output reg signed [15:0]         sample_out,
    output reg                       clipped
);

reg signed [23:0] amplified;

always @(posedge clk) begin
    if (rst) begin
        sample_valid_out <= 1'b0;
        sample_out       <= 16'sd0;
        clipped          <= 1'b0;
    end else begin
        sample_valid_out <= sample_valid_in;
        clipped          <= 1'b0;

        if (sample_valid_in) begin
            amplified = $signed(sample_in) <<< gain_shift;

            if (amplified > 24'sd32767) begin
                sample_out <= 16'sd32767;
                clipped    <= 1'b1;
            end else if (amplified < -24'sd32768) begin
                sample_out <= -16'sd32768;
                clipped    <= 1'b1;
            end else begin
                sample_out <= amplified[15:0];
            end
        end
    end
end

endmodule
