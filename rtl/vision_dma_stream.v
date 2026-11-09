`timescale 1ns / 1ps

// Fixed raster DMA contract: FRAME_WIDTH*FRAME_HEIGHT accepted 32-bit beats
// per frame, TKEEP=4'hf throughout, and input TLAST only on the final beat.
// Each input word is {8'h00, R, G, B} (little-endian bytes B,G,R,0).
// Input TLAST/TKEEP do not resynchronize or qualify pixels: reset restarts
// the raster, and the DMA producer must supply complete frames in order.
// Output bytes are gray, clipped Sobel L1, edge (255 or 0), and zero.
// Sobel uses the trailing 3x3 window; its first two rows/columns are zero.
// Hold threshold constant for a stream (tie to 8'd64 in the block design).
module vision_dma_stream #(
    parameter integer FRAME_WIDTH = 640,
    parameter integer FRAME_HEIGHT = 480
) (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 aclk CLK",
       X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF S_AXIS:M_AXIS, ASSOCIATED_RESET aresetn, FREQ_HZ 100000000" *)
    input wire aclk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 aresetn RST",
       X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input wire aresetn,
    input wire [7:0] threshold,

    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TDATA",
       X_INTERFACE_PARAMETER = "TDATA_NUM_BYTES 4, HAS_TKEEP 1, HAS_TLAST 1, HAS_TREADY 1" *)
    input wire [31:0] s_axis_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TKEEP" *)
    input wire [3:0] s_axis_tkeep,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TLAST" *)
    input wire s_axis_tlast,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TVALID" *)
    input wire s_axis_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TREADY" *)
    output wire s_axis_tready,

    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TDATA",
       X_INTERFACE_PARAMETER = "TDATA_NUM_BYTES 4, HAS_TKEEP 1, HAS_TLAST 1, HAS_TREADY 1" *)
    output wire [31:0] m_axis_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TKEEP" *)
    output wire [3:0] m_axis_tkeep,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TLAST" *)
    output wire m_axis_tlast,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TVALID" *)
    output wire m_axis_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TREADY" *)
    input wire m_axis_tready
);

localparam integer X_BITS = (FRAME_WIDTH > 1) ? $clog2(FRAME_WIDTH) : 1;
localparam integer Y_BITS = (FRAME_HEIGHT > 1) ? $clog2(FRAME_HEIGHT) : 1;
localparam integer FRAME_PIXELS = FRAME_WIDTH * FRAME_HEIGHT;
localparam integer PIXEL_BITS = (FRAME_PIXELS > 1) ? $clog2(FRAME_PIXELS) : 1;
localparam [X_BITS-1:0] LAST_COLUMN = FRAME_WIDTH - 1;
localparam [Y_BITS-1:0] LAST_ROW = FRAME_HEIGHT - 1;
localparam [PIXEL_BITS-1:0] LAST_PIXEL = FRAME_PIXELS - 1;

wire rst = !aresetn;
reg [X_BITS-1:0] input_x;
reg [Y_BITS-1:0] input_y;
reg [PIXEL_BITS-1:0] output_pixel;
wire input_sof = (input_x == 0) && (input_y == 0);
wire input_eol = (input_x == LAST_COLUMN);
wire [7:0] gray;
wire gray_valid, gray_ready, gray_sof, gray_eol;
wire [7:0] sobel_gray, magnitude;
wire sobel_edge, sobel_valid;

vision_rgb_gray gray_stage (
    .clk(aclk), .rst(rst), .s_rgb(s_axis_tdata[23:0]),
    .s_valid(s_axis_tvalid), .s_ready(s_axis_tready),
    .s_sof(input_sof), .s_eol(input_eol),
    .m_gray(gray), .m_valid(gray_valid), .m_ready(gray_ready),
    .m_sof(gray_sof), .m_eol(gray_eol)
);

vision_sobel3x3 #(.FRAME_WIDTH(FRAME_WIDTH)) sobel_stage (
    .clk(aclk), .rst(rst), .s_gray(gray), .s_valid(gray_valid),
    .s_ready(gray_ready), .s_sof(gray_sof), .s_eol(gray_eol),
    .threshold(threshold), .m_gx(), .m_gy(),
    .m_magnitude(magnitude), .m_edge(sobel_edge),
    .m_valid(sobel_valid), .m_ready(m_axis_tready),
    .m_sof(), .m_eol(), .m_gray(sobel_gray)
);

assign m_axis_tdata = {8'd0, {8{sobel_edge}}, magnitude, sobel_gray};
assign m_axis_tkeep = 4'hf;
assign m_axis_tvalid = aresetn && sobel_valid;
assign m_axis_tlast = m_axis_tvalid && (output_pixel == LAST_PIXEL);

always @(posedge aclk) begin
    if (rst) begin
        input_x <= 0;
        input_y <= 0;
        output_pixel <= 0;
    end else begin
        if (s_axis_tvalid && s_axis_tready) begin
            if (input_eol) begin
                input_x <= 0;
                input_y <= (input_y == LAST_ROW) ? 0 : input_y + 1'b1;
            end else begin
                input_x <= input_x + 1'b1;
            end
        end
        if (m_axis_tvalid && m_axis_tready)
            output_pixel <= (output_pixel == LAST_PIXEL) ? 0 : output_pixel + 1'b1;
    end
end

// synthesis translate_off
initial begin
    if ((FRAME_WIDTH < 3) || (FRAME_HEIGHT < 1))
        $fatal(1, "vision_dma_stream requires FRAME_WIDTH >= 3 and FRAME_HEIGHT >= 1");
end
// synthesis translate_on

endmodule
