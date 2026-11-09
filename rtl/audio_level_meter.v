`timescale 1ns / 1ps

module audio_level_meter #(
    parameter integer WINDOW_SAMPLES = 1024,
    parameter [15:0]  ACTIVE_THRESHOLD = 16'd1000
)(
    input  wire                     clk,
    input  wire                     rst,
    input  wire                     sample_valid,
    input  wire signed [15:0]       sample_in,
    output reg  [15:0]              peak_level,
    output reg                      signal_active,
    output reg                      level_valid
);

reg [15:0] window_peak;
reg [15:0] sample_index;
reg [15:0] magnitude;

always @(*) begin
    if (sample_in[15]) begin
        if (sample_in == -16'sd32768)
            magnitude = 16'd32768;
        else
            magnitude = -sample_in;
    end else begin
        magnitude = sample_in;
    end
end

always @(posedge clk) begin
    if (rst) begin
        window_peak  <= 16'd0;
        sample_index <= 16'd0;
        peak_level   <= 16'd0;
        signal_active <= 1'b0;
        level_valid  <= 1'b0;
    end else begin
        level_valid <= 1'b0;

        if (sample_valid) begin
            if (sample_index == WINDOW_SAMPLES - 1) begin
                if (magnitude > window_peak)
                    peak_level <= magnitude;
                else
                    peak_level <= window_peak;

                signal_active <= ((magnitude > window_peak) ? magnitude : window_peak)
                                 >= ACTIVE_THRESHOLD;
                level_valid   <= 1'b1;
                sample_index  <= 16'd0;
                window_peak   <= 16'd0;
            end else begin
                sample_index <= sample_index + 1'b1;
                if (magnitude > window_peak)
                    window_peak <= magnitude;
            end
        end
    end
end

endmodule
