`timescale 1ns / 1ps

// Generic I2S transmitter. Data changes on falling BCLK. LRCLK and BCLK are
// supplied by the codec or by an external I2S master.
module audio_i2s_tx #(
    parameter integer SAMPLE_WIDTH = 24,
    parameter integer SLOT_WIDTH   = 32
)(
    input  wire                         rst,
    input  wire                         bclk,
    input  wire                         lrclk,
    input  wire signed [SAMPLE_WIDTH-1:0] left_sample,
    input  wire signed [SAMPLE_WIDTH-1:0] right_sample,
    output reg                          sdata_out
);

localparam integer COUNT_WIDTH = (SLOT_WIDTH < 2) ? 1 : $clog2(SLOT_WIDTH + 1);

reg                         lrclk_d;
reg                         lrclk_sampled;
reg                         channel;
reg [COUNT_WIDTH-1:0]       slot_count;
reg [SAMPLE_WIDTH-1:0]      shift_register;

// Sample LRCLK away from its falling-edge transition, then use the stable
// value half a BCLK later to preserve the I2S one-bit delay at the pads.
always @(posedge bclk) begin
    lrclk_sampled <= lrclk;
end

always @(negedge bclk) begin
    if (rst) begin
        lrclk_d        <= lrclk_sampled;
        channel        <= 1'b0;
        slot_count     <= {COUNT_WIDTH{1'b0}};
        shift_register <= {SAMPLE_WIDTH{1'b0}};
        sdata_out      <= 1'b0;
    end else begin
        if (lrclk_sampled != lrclk_d) begin
            // The rising edge recognized the new slot's one-bit delay.
            lrclk_d        <= lrclk_sampled;
            channel        <= lrclk_sampled;
            shift_register <= (lrclk_sampled == 1'b0) ?
                              {left_sample[SAMPLE_WIDTH-2:0], 1'b0} :
                              {right_sample[SAMPLE_WIDTH-2:0], 1'b0};
            sdata_out      <= (lrclk_sampled == 1'b0) ?
                              left_sample[SAMPLE_WIDTH-1] :
                              right_sample[SAMPLE_WIDTH-1];
            slot_count     <= {{(COUNT_WIDTH-1){1'b0}}, 1'b1};
        end else begin
            if (slot_count < SAMPLE_WIDTH) begin
                sdata_out      <= shift_register[SAMPLE_WIDTH-1];
                shift_register <= {shift_register[SAMPLE_WIDTH-2:0], 1'b0};
            end else begin
                sdata_out <= 1'b0;
            end

            if (slot_count == SLOT_WIDTH - 1)
                slot_count <= {COUNT_WIDTH{1'b0}};
            else
                slot_count <= slot_count + 1'b1;
        end
    end
end

endmodule
