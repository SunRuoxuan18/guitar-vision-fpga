`timescale 1ns / 1ps

module audio_frame_buffer #(
    parameter integer FRAME_SAMPLES = 1024,
    parameter integer ADDR_WIDTH = 10
)(
    input  wire                     clk,
    input  wire                     rst,
    input  wire                     sample_valid,
    input  wire signed [15:0]       sample_in,
    output reg                      frame_ready,
    output reg  [31:0]              frame_number,
    input  wire                     read_enable,
    input  wire [ADDR_WIDTH-1:0]    read_address,
    output reg signed [15:0]        read_data
);

reg signed [15:0] sample_memory [0:FRAME_SAMPLES-1];
reg [ADDR_WIDTH-1:0] write_address;

always @(posedge clk) begin
    if (rst) begin
        write_address <= {ADDR_WIDTH{1'b0}};
        frame_ready   <= 1'b0;
        frame_number  <= 32'd0;
        read_data     <= 16'sd0;
    end else begin
        frame_ready <= 1'b0;

        if (sample_valid) begin
            sample_memory[write_address] <= sample_in;

            if (write_address == FRAME_SAMPLES - 1) begin
                write_address <= {ADDR_WIDTH{1'b0}};
                frame_ready   <= 1'b1;
                frame_number  <= frame_number + 1'b1;
            end else begin
                write_address <= write_address + 1'b1;
            end
        end

        if (read_enable)
            read_data <= sample_memory[read_address];
    end
end

endmodule
