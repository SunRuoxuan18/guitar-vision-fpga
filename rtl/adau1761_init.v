`timescale 1ns / 1ps

// Programs the PYNQ-Z2 ADAU1761 for 48 kHz, 24-bit standard I2S operation.
// The codec becomes the BCLK/LRCLK master and routes FPGA serial audio to the
// headphone output. LINE IN is also enabled for the next board-test stage.
module adau1761_init #(
    parameter integer CLK_FREQ_HZ = 125_000_000,
    parameter integer I2C_FREQ_HZ = 100_000,
    parameter integer STARTUP_WAIT_CYCLES = 125_000,
    parameter integer PLL_WAIT_CYCLES = 3_125_000,
    parameter integer CORE_WAIT_CYCLES = 625_000
)(
    input  wire clk,
    input  wire rst,
    input  wire sda_in,
    output wire scl_drive_low,
    output wire sda_drive_low,
    output wire busy,
    output reg  done,
    output reg  error,
    output wire [5:0] debug_command_index
);

localparam [2:0] ST_STARTUP = 3'd0;
localparam [2:0] ST_ISSUE   = 3'd1;
localparam [2:0] ST_WAIT    = 3'd2;
localparam [2:0] ST_DELAY   = 3'd3;
localparam [2:0] ST_DONE    = 3'd4;

localparam [5:0] LAST_COMMAND = 6'd33;

reg [2:0] state;
reg [5:0] command_index;
reg [31:0] wait_count;
reg master_start;

wire master_busy;
wire master_done;
wire master_nack;
wire [63:0] command_payload;
wire [3:0] command_length;

assign busy = (state != ST_DONE);
assign debug_command_index = command_index;
assign command_length = ((command_index == 6'd1) ||
                         (command_index == 6'd2)) ? 4'd8 : 4'd3;

function [63:0] payload_for_command;
    input [5:0] index;
    begin
        case (index)
            6'd0:  payload_for_command = {8'h40, 8'h00, 8'h0E, 40'd0};
            // Disable the complete PLL register before changing its parameters.
            6'd1:  payload_for_command = 64'h4002_0000_0000_0000;
            6'd2:  payload_for_command = {8'h40, 8'h02, 8'h02, 8'h71,
                                           8'h02, 8'h3C, 8'h21, 8'h01};
            6'd3:  payload_for_command = {8'h40, 8'h00, 8'h0F, 40'd0};
            6'd4:  payload_for_command = {8'h40, 8'h15, 8'h01, 40'd0};
            6'd5:  payload_for_command = {8'h40, 8'h16, 8'h00, 40'd0};
            6'd6:  payload_for_command = {8'h40, 8'h0A, 8'h01, 40'd0};
            6'd7:  payload_for_command = {8'h40, 8'h0B, 8'h03, 40'd0};
            6'd8:  payload_for_command = {8'h40, 8'h0C, 8'h01, 40'd0};
            6'd9:  payload_for_command = {8'h40, 8'h0D, 8'h03, 40'd0};
            6'd10: payload_for_command = {8'h40, 8'h19, 8'h33, 40'd0};
            6'd11: payload_for_command = {8'h40, 8'h1C, 8'h21, 40'd0};
            6'd12: payload_for_command = {8'h40, 8'h1E, 8'h41, 40'd0};
            // Start at a conservative headphone level (about -17 dB).
            6'd13: payload_for_command = {8'h40, 8'h23, 8'hA3, 40'd0};
            6'd14: payload_for_command = {8'h40, 8'h24, 8'hA3, 40'd0};
            6'd15: payload_for_command = {8'h40, 8'h29, 8'h03, 40'd0};
            6'd16: payload_for_command = {8'h40, 8'h2A, 8'h03, 40'd0};
            6'd17: payload_for_command = {8'h40, 8'hF2, 8'h01, 40'd0};
            6'd18: payload_for_command = {8'h40, 8'hF3, 8'h01, 40'd0};
            6'd19: payload_for_command = {8'h40, 8'hF5, 8'h01, 40'd0};
            6'd20: payload_for_command = {8'h40, 8'hF6, 8'h01, 40'd0};
            6'd21: payload_for_command = {8'h40, 8'hF8, 8'h00, 40'd0};
            6'd22: payload_for_command = {8'h40, 8'hF9, 8'h7F, 40'd0};
            6'd23: payload_for_command = {8'h40, 8'hFA, 8'h03, 40'd0};
            // Do not inherit digital attenuation, bypass or pin modes from
            // a previously running codec configuration.
            6'd24: payload_for_command = {8'h40, 8'h17, 8'h00, 40'd0};
            6'd25: payload_for_command = {8'h40, 8'h18, 8'h00, 40'd0};
            6'd26: payload_for_command = {8'h40, 8'h1A, 8'h00, 40'd0};
            6'd27: payload_for_command = {8'h40, 8'h1B, 8'h00, 40'd0};
            6'd28: payload_for_command = {8'h40, 8'h1D, 8'h00, 40'd0};
            6'd29: payload_for_command = {8'h40, 8'h1F, 8'h00, 40'd0};
            6'd30: payload_for_command = {8'h40, 8'h2B, 8'h00, 40'd0};
            6'd31: payload_for_command = {8'h40, 8'h2C, 8'h00, 40'd0};
            6'd32: payload_for_command = {8'h40, 8'hF4, 8'h00, 40'd0};
            6'd33: payload_for_command = {8'h40, 8'hF7, 8'h00, 40'd0};
            default: payload_for_command = 64'd0;
        endcase
    end
endfunction

assign command_payload = payload_for_command(command_index);

i2c_write_master #(
    .CLK_FREQ_HZ(CLK_FREQ_HZ),
    .I2C_FREQ_HZ(I2C_FREQ_HZ)
) i2c_master_inst (
    .clk            (clk),
    .rst            (rst),
    .start          (master_start),
    .device_address (7'h3B),
    .payload        (command_payload),
    .payload_length (command_length),
    .sda_in         (sda_in),
    .scl_drive_low  (scl_drive_low),
    .sda_drive_low  (sda_drive_low),
    .busy           (master_busy),
    .done           (master_done),
    .nack           (master_nack)
);

always @(posedge clk) begin
    if (rst) begin
        state         <= ST_STARTUP;
        command_index <= 6'd0;
        wait_count    <= 32'd0;
        master_start  <= 1'b0;
        done          <= 1'b0;
        error         <= 1'b0;
    end else begin
        master_start <= 1'b0;

        case (state)
            ST_STARTUP: begin
                if (wait_count >= STARTUP_WAIT_CYCLES - 1) begin
                    wait_count <= 32'd0;
                    state      <= ST_ISSUE;
                end else begin
                    wait_count <= wait_count + 1'b1;
                end
            end

            ST_ISSUE: begin
                if (!master_busy) begin
                    master_start <= 1'b1;
                    state        <= ST_WAIT;
                end
            end

            ST_WAIT: begin
                if (master_done) begin
                    if (master_nack) begin
                        error <= 1'b1;
                        done  <= 1'b1;
                        state <= ST_DONE;
                    end else if (command_index == LAST_COMMAND) begin
                        done  <= 1'b1;
                        state <= ST_DONE;
                    end else if ((command_index == 6'd2) ||
                                 (command_index == 6'd3)) begin
                        command_index <= command_index + 1'b1;
                        wait_count    <= 32'd0;
                        state         <= ST_DELAY;
                    end else begin
                        command_index <= command_index + 1'b1;
                        state         <= ST_ISSUE;
                    end
                end
            end

            ST_DELAY: begin
                if (((command_index == 6'd3) &&
                     (wait_count >= PLL_WAIT_CYCLES - 1)) ||
                    ((command_index == 6'd4) &&
                     (wait_count >= CORE_WAIT_CYCLES - 1))) begin
                    wait_count <= 32'd0;
                    state      <= ST_ISSUE;
                end else begin
                    wait_count <= wait_count + 1'b1;
                end
            end

            default: begin
                done  <= 1'b1;
                state <= ST_DONE;
            end
        endcase
    end
end

endmodule
