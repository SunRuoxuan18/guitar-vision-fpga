`timescale 1ns / 1ps

// Transfers a low-rate PCM sample event from the I2S BCLK domain into the
// system-clock domain. The source data is held stable while a toggle crosses
// a two-register synchronizer.
module audio_sample_cdc #(
    parameter integer SAMPLE_WIDTH = 16
)(
    input  wire                         src_clk,
    input  wire                         src_rst,
    input  wire                         src_valid,
    input  wire signed [SAMPLE_WIDTH-1:0] src_sample,
    input  wire                         dst_clk,
    input  wire                         dst_rst,
    output reg                          dst_valid,
    output reg signed [SAMPLE_WIDTH-1:0] dst_sample
);

reg signed [SAMPLE_WIDTH-1:0] src_sample_hold;
reg src_toggle;
reg dst_toggle_meta;
reg dst_toggle_sync;
reg dst_toggle_seen;

always @(posedge src_clk) begin
    if (src_rst) begin
        src_sample_hold <= {SAMPLE_WIDTH{1'b0}};
        src_toggle      <= 1'b0;
    end else if (src_valid) begin
        src_sample_hold <= src_sample;
        src_toggle      <= ~src_toggle;
    end
end

always @(posedge dst_clk) begin
    if (dst_rst) begin
        dst_toggle_meta <= 1'b0;
        dst_toggle_sync <= 1'b0;
        dst_toggle_seen <= 1'b0;
        dst_valid       <= 1'b0;
        dst_sample      <= {SAMPLE_WIDTH{1'b0}};
    end else begin
        dst_toggle_meta <= src_toggle;
        dst_toggle_sync <= dst_toggle_meta;
        dst_valid       <= 1'b0;

        if (dst_toggle_sync != dst_toggle_seen) begin
            dst_toggle_seen <= dst_toggle_sync;
            dst_sample      <= src_sample_hold;
            dst_valid       <= 1'b1;
        end
    end
end

endmodule
