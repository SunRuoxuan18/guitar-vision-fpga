`timescale 1ns / 1ps

// Two banks keep acquisition independent of a burst FFT reader. A full bank
// remains owned by the reader until release_bank is asserted.
module audio_fft_capture #(
    parameter integer ADDR_WIDTH = 12
)(
    input wire clk,
    input wire rst,
    input wire sample_valid,
    input wire signed [15:0] sample_in,
    output reg [1:0] bank_ready,
    input wire release_valid,
    input wire release_bank,
    input wire read_enable,
    input wire read_bank,
    input wire [ADDR_WIDTH-1:0] read_address,
    output wire signed [15:0] read_data,
    output reg [31:0] dropped_frames
);
localparam integer DEPTH = 1 << ADDR_WIDTH;
(* ram_style = "block" *) reg signed [15:0] bank0 [0:DEPTH-1];
(* ram_style = "block" *) reg signed [15:0] bank1 [0:DEPTH-1];
reg [ADDR_WIDTH-1:0] write_address;
reg write_bank;
reg discard_frame;
reg signed [15:0] bank0_read;
reg signed [15:0] bank1_read;
reg read_bank_q;
wire write_enable = !rst && sample_valid && !discard_frame && !bank_ready[write_bank];

assign read_data = read_bank_q ? bank1_read : bank0_read;

always @(posedge clk) begin
    if (write_enable && !write_bank)
        bank0[write_address] <= sample_in;
    if (read_enable)
        bank0_read <= bank0[read_address];
end

always @(posedge clk) begin
    if (write_enable && write_bank)
        bank1[write_address] <= sample_in;
    if (read_enable)
        bank1_read <= bank1[read_address];
end

always @(posedge clk) begin
    if (read_enable)
        read_bank_q <= read_bank;
end

always @(posedge clk) begin
    if (rst) begin
        bank_ready <= 2'b00;
        write_address <= 0;
        write_bank <= 0;
        discard_frame <= 0;
        dropped_frames <= 0;
    end else begin
        if (release_valid)
            bank_ready[release_bank] <= 1'b0;
        if (sample_valid) begin
            if (&write_address) begin
                write_address <= 0;
                if (!discard_frame && !bank_ready[write_bank])
                    bank_ready[write_bank] <= 1'b1;
                else
                    dropped_frames <= dropped_frames + 1'b1;
                // Prefer the other bank. Account for a release on this edge.
                if (!bank_ready[!write_bank] ||
                    (release_valid && release_bank == !write_bank)) begin
                    write_bank <= !write_bank;
                    discard_frame <= 0;
                end else if ((discard_frame && !bank_ready[write_bank]) ||
                             (release_valid && release_bank == write_bank)) begin
                    discard_frame <= 0;
                end else begin
                    discard_frame <= 1;
                end
            end else begin
                write_address <= write_address + 1'b1;
            end
        end
    end
end
endmodule
