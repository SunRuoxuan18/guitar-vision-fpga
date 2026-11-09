`timescale 1ns / 1ps

// Generic I2S receiver. Data is sampled on the rising edge of BCLK and is
// expected to change on the falling edge. LRCLK=0 is left, LRCLK=1 is right.
// The one-bit I2S delay is handled by ignoring the edge on which LRCLK changes.
module audio_i2s_rx #(
    parameter integer SAMPLE_WIDTH = 24,
    parameter integer SLOT_WIDTH   = 32
)(
    input  wire                         rst,
    input  wire                         bclk,
    input  wire                         lrclk,
    input  wire                         sdata_in,
    output reg signed [SAMPLE_WIDTH-1:0] left_sample,
    output reg signed [SAMPLE_WIDTH-1:0] right_sample,
    output reg                          sample_valid
);

localparam integer COUNT_WIDTH = (SLOT_WIDTH < 2) ? 1 : $clog2(SLOT_WIDTH + 1);

reg                         lrclk_d;
reg                         channel;
reg [COUNT_WIDTH-1:0]       slot_count;
reg [SAMPLE_WIDTH-1:0]      shift_register;

always @(posedge bclk) begin
    if (rst) begin
        lrclk_d       <= lrclk;
        channel       <= 1'b0;
        slot_count    <= {COUNT_WIDTH{1'b0}};
        shift_register<= {SAMPLE_WIDTH{1'b0}};
        left_sample   <= {SAMPLE_WIDTH{1'b0}};
        right_sample  <= {SAMPLE_WIDTH{1'b0}};
        sample_valid  <= 1'b0;
    end else begin
        sample_valid <= 1'b0;

        if (lrclk != lrclk_d) begin
            // This edge is the I2S one-bit delay edge. The first data bit is
            // sampled on the following rising BCLK edge.
            lrclk_d        <= lrclk;
            channel        <= lrclk;
            slot_count     <= {COUNT_WIDTH{1'b0}};
            shift_register <= {SAMPLE_WIDTH{1'b0}};
        end else begin
            if (slot_count < SAMPLE_WIDTH) begin
                if (slot_count == SAMPLE_WIDTH - 1) begin
                    if (channel == 1'b0)
                        left_sample <= {shift_register[SAMPLE_WIDTH-2:0], sdata_in};
                    else begin
                        right_sample <= {shift_register[SAMPLE_WIDTH-2:0], sdata_in};
                        sample_valid <= 1'b1;
                    end
                end
                shift_register <= {shift_register[SAMPLE_WIDTH-2:0], sdata_in};
            end

            if (slot_count == SLOT_WIDTH - 1)
                slot_count <= {COUNT_WIDTH{1'b0}};
            else
                slot_count <= slot_count + 1'b1;
        end
    end
end

endmodule
