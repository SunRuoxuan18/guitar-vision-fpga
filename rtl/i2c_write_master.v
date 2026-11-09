`timescale 1ns / 1ps

// Small write-only I2C controller for board-startup register programming.
// scl_drive_low and sda_drive_low are open-drain controls: 1 pulls the line
// low and 0 releases it to the external pull-up.
module i2c_write_master #(
    parameter integer CLK_FREQ_HZ = 125_000_000,
    parameter integer I2C_FREQ_HZ = 100_000
)(
    input  wire        clk,
    input  wire        rst,
    input  wire        start,
    input  wire [6:0]  device_address,
    input  wire [63:0] payload,
    input  wire [3:0]  payload_length,
    input  wire        sda_in,
    output reg         scl_drive_low,
    output reg         sda_drive_low,
    output reg         busy,
    output reg         done,
    output reg         nack
);

localparam integer HALF_PERIOD_CYCLES =
    (CLK_FREQ_HZ + I2C_FREQ_HZ) / (2 * I2C_FREQ_HZ);
localparam integer DIV_WIDTH =
    (HALF_PERIOD_CYCLES <= 1) ? 1 : $clog2(HALF_PERIOD_CYCLES);

localparam [3:0] ST_IDLE       = 4'd0;
localparam [3:0] ST_START_HOLD = 4'd1;
localparam [3:0] ST_BIT_LOW    = 4'd2;
localparam [3:0] ST_BIT_HIGH   = 4'd3;
localparam [3:0] ST_ACK_LOW    = 4'd4;
localparam [3:0] ST_ACK_HIGH   = 4'd5;
localparam [3:0] ST_STOP_LOW   = 4'd6;
localparam [3:0] ST_STOP_HIGH  = 4'd7;
localparam [3:0] ST_BUS_FREE   = 4'd8;

reg [3:0] state;
reg [DIV_WIDTH-1:0] divider_count;
reg [7:0] current_byte;
reg [2:0] bit_index;
reg [63:0] payload_shift;
reg [3:0] bytes_remaining;
reg sending_address;

wire half_period_done = (divider_count == HALF_PERIOD_CYCLES - 1);

always @(posedge clk) begin
    if (rst) begin
        state           <= ST_IDLE;
        divider_count   <= {DIV_WIDTH{1'b0}};
        current_byte    <= 8'd0;
        bit_index       <= 3'd7;
        payload_shift   <= 64'd0;
        bytes_remaining <= 4'd0;
        sending_address <= 1'b0;
        scl_drive_low   <= 1'b0;
        sda_drive_low   <= 1'b0;
        busy            <= 1'b0;
        done            <= 1'b0;
        nack            <= 1'b0;
    end else begin
        done <= 1'b0;

        if (!busy) begin
            divider_count <= {DIV_WIDTH{1'b0}};

            if (start && (payload_length != 0)) begin
                current_byte    <= {device_address, 1'b0};
                bit_index       <= 3'd7;
                payload_shift   <= payload;
                bytes_remaining <= payload_length;
                sending_address <= 1'b1;
                scl_drive_low   <= 1'b0;
                sda_drive_low   <= 1'b1;
                busy            <= 1'b1;
                nack            <= 1'b0;
                state           <= ST_START_HOLD;
            end
        end else if (half_period_done) begin
            divider_count <= {DIV_WIDTH{1'b0}};

            case (state)
                ST_START_HOLD: begin
                    scl_drive_low <= 1'b1;
                    sda_drive_low <= ~current_byte[7];
                    state         <= ST_BIT_LOW;
                end

                ST_BIT_LOW: begin
                    scl_drive_low <= 1'b0;
                    state         <= ST_BIT_HIGH;
                end

                ST_BIT_HIGH: begin
                    scl_drive_low <= 1'b1;

                    if (bit_index == 0) begin
                        sda_drive_low <= 1'b0;
                        state         <= ST_ACK_LOW;
                    end else begin
                        bit_index     <= bit_index - 1'b1;
                        sda_drive_low <= ~current_byte[bit_index - 1'b1];
                        state         <= ST_BIT_LOW;
                    end
                end

                ST_ACK_LOW: begin
                    scl_drive_low <= 1'b0;
                    state         <= ST_ACK_HIGH;
                end

                ST_ACK_HIGH: begin
                    scl_drive_low <= 1'b1;

                    if (sda_in) begin
                        nack          <= 1'b1;
                        sda_drive_low <= 1'b1;
                        state         <= ST_STOP_LOW;
                    end else if (sending_address) begin
                        sending_address <= 1'b0;
                        current_byte    <= payload_shift[63:56];
                        payload_shift   <= {payload_shift[55:0], 8'd0};
                        bytes_remaining <= bytes_remaining - 1'b1;
                        bit_index       <= 3'd7;
                        sda_drive_low   <= ~payload_shift[63];
                        state           <= ST_BIT_LOW;
                    end else if (bytes_remaining != 0) begin
                        current_byte    <= payload_shift[63:56];
                        payload_shift   <= {payload_shift[55:0], 8'd0};
                        bytes_remaining <= bytes_remaining - 1'b1;
                        bit_index       <= 3'd7;
                        sda_drive_low   <= ~payload_shift[63];
                        state           <= ST_BIT_LOW;
                    end else begin
                        sda_drive_low <= 1'b1;
                        state         <= ST_STOP_LOW;
                    end
                end

                ST_STOP_LOW: begin
                    scl_drive_low <= 1'b0;
                    sda_drive_low <= 1'b1;
                    state         <= ST_STOP_HIGH;
                end

                ST_STOP_HIGH: begin
                    sda_drive_low <= 1'b0;
                    state         <= ST_BUS_FREE;
                end

                // Standard-mode I2C requires at least 4.7 us between a STOP
                // and the following START. One 100 kHz half-period is 5 us.
                ST_BUS_FREE: begin
                    busy  <= 1'b0;
                    done  <= 1'b1;
                    state <= ST_IDLE;
                end

                default: begin
                    scl_drive_low <= 1'b0;
                    sda_drive_low <= 1'b0;
                    busy          <= 1'b0;
                    nack          <= 1'b1;
                    done          <= 1'b1;
                    state         <= ST_IDLE;
                end
            endcase
        end else begin
            divider_count <= divider_count + 1'b1;
        end
    end
end

endmodule
