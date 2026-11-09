`timescale 1ns / 1ps

// Raster input: FRAME_WIDTH >= 3; EOL accompanies column FRAME_WIDTH-1.
// SOF restarts at (0,0), including mid-line, without discarding queued outputs.
// The output for (x,y) uses the window ending at (x,y), not centered at (x,y).
// First two rows/columns have zero gradients, magnitude and edge flag.
// Threshold is sampled on the input handshake; interior edge = clipped L1 >= it.
// Elastic gradient registers separate the RAM/window path from magnitude math.
module vision_sobel3x3 #(
    parameter integer FRAME_WIDTH = 640
) (
    input  wire               clk,
    input  wire               rst,
    input  wire [7:0]         s_gray,
    input  wire               s_valid,
    output wire               s_ready,
    input  wire               s_sof,
    input  wire               s_eol,
    input  wire [7:0]         threshold,
    output reg signed [11:0]  m_gx,
    output reg signed [11:0]  m_gy,
    output reg  [7:0]         m_magnitude,
    output reg                m_edge,
    output reg                m_valid,
    input  wire               m_ready,
    output reg                m_sof,
    output reg                m_eol,
    output reg  [7:0]         m_gray
);

localparam integer X_BITS = (FRAME_WIDTH > 1) ? $clog2(FRAME_WIDTH) : 1;
localparam [X_BITS-1:0] LAST_COLUMN = FRAME_WIDTH - 1;

// Alternating read-first banks retain the preceding two rows. There is no
// RAM reset; rows_seen masks unwritten/stale data after reset or accepted SOF.
(* ram_style = "block" *) reg [7:0] line0 [0:FRAME_WIDTH-1];
(* ram_style = "block" *) reg [7:0] line1 [0:FRAME_WIDTH-1];
reg [7:0] line0_q, line1_q;
reg [X_BITS-1:0] x_count;
reg [1:0] rows_seen;
reg row_bank;

reg pending_valid;
reg [7:0] pending_gray, pending_threshold;
reg [X_BITS-1:0] pending_x;
reg [1:0] pending_rows;
reg pending_bank, pending_sof, pending_eol;
reg [7:0] top_left, top_center;
reg [7:0] middle_left, middle_center;
reg [7:0] bottom_left, bottom_center;
reg gradient_valid;
reg signed [10:0] gradient_gx, gradient_gy;
reg [7:0] gradient_gray, gradient_threshold;
reg gradient_interior, gradient_sof, gradient_eol;

wire output_ready = !m_valid || m_ready;
wire gradient_ready = !gradient_valid || output_ready;
assign s_ready = !rst && (!pending_valid || gradient_ready);
wire accept_pixel = s_valid && s_ready;
wire [X_BITS-1:0] input_x = s_sof ? {X_BITS{1'b0}} : x_count;
wire [1:0] input_rows = s_sof ? 2'd0 : rows_seen;
wire input_bank = s_sof ? 1'b0 : row_bank;

always @(posedge clk) begin
    if (accept_pixel) begin
        line0_q <= line0[input_x];
        if (!input_bank)
            line0[input_x] <= s_gray;
    end
end

always @(posedge clk) begin
    if (accept_pixel) begin
        line1_q <= line1[input_x];
        if (input_bank)
            line1[input_x] <= s_gray;
    end
end

wire [7:0] top_pixel = (pending_rows < 2) ? 8'd0 :
                       (pending_bank ? line1_q : line0_q);
wire [7:0] middle_pixel = (pending_rows == 0) ? 8'd0 :
                          (pending_bank ? line0_q : line1_q);

// Pair opposite taps first. Differences fit in 9 signed bits, row sums in
// 10 unsigned bits, and each complete derivative in 11 signed bits (+/-1020).
wire signed [8:0] dx_top = $signed({1'b0, top_pixel}) - $signed({1'b0, top_left});
wire signed [8:0] dx_middle = $signed({1'b0, middle_pixel}) - $signed({1'b0, middle_left});
wire signed [8:0] dx_bottom = $signed({1'b0, pending_gray}) - $signed({1'b0, bottom_left});
wire signed [9:0] dx_outer = $signed({dx_top[8], dx_top}) +
                             $signed({dx_bottom[8], dx_bottom});
wire signed [9:0] dx_weighted = {dx_middle, 1'b0};
wire signed [10:0] gx = $signed({dx_outer[9], dx_outer}) +
                        $signed({dx_weighted[9], dx_weighted});
wire [8:0] top_outer = {1'b0, top_left} + {1'b0, top_pixel};
wire [8:0] bottom_outer = {1'b0, bottom_left} + {1'b0, pending_gray};
wire [9:0] top_sum = {1'b0, top_outer} + {1'b0, top_center, 1'b0};
wire [9:0] bottom_sum = {1'b0, bottom_outer} + {1'b0, bottom_center, 1'b0};
wire signed [10:0] gy = $signed({1'b0, bottom_sum}) - $signed({1'b0, top_sum});
wire [10:0] abs_gx = gradient_gx[10] ? -gradient_gx : gradient_gx;
wire [10:0] abs_gy = gradient_gy[10] ? -gradient_gy : gradient_gy;
wire [11:0] l1 = {1'b0, abs_gx} + {1'b0, abs_gy};
wire [7:0] clipped_l1 = (|l1[11:8]) ? 8'd255 : l1[7:0];

always @(posedge clk) begin
    if (rst) begin
        x_count        <= 0;
        rows_seen      <= 0;
        row_bank       <= 0;
        pending_valid  <= 0;
        pending_gray   <= 0;
        pending_threshold <= 0;
        pending_x      <= 0;
        pending_rows   <= 0;
        pending_bank   <= 0;
        pending_sof    <= 0;
        pending_eol    <= 0;
        top_left       <= 0;
        top_center     <= 0;
        middle_left    <= 0;
        middle_center  <= 0;
        bottom_left    <= 0;
        bottom_center  <= 0;
        gradient_valid <= 0;
        gradient_gx    <= 0;
        gradient_gy    <= 0;
        gradient_gray  <= 0;
        gradient_threshold <= 0;
        gradient_interior <= 0;
        gradient_sof   <= 0;
        gradient_eol   <= 0;
        m_gx           <= 0;
        m_gy           <= 0;
        m_magnitude    <= 0;
        m_edge         <= 0;
        m_valid        <= 0;
        m_sof          <= 0;
        m_eol          <= 0;
        m_gray         <= 0;
    end else begin
        if (s_ready) begin
            pending_valid <= s_valid;
            if (s_valid) begin
                pending_gray      <= s_gray;
                pending_threshold <= threshold;
                pending_x         <= input_x;
                pending_rows      <= input_rows;
                pending_bank      <= input_bank;
                pending_sof       <= s_sof;
                pending_eol       <= s_eol;
                if (input_x == LAST_COLUMN) begin
                    x_count   <= 0;
                    row_bank  <= !input_bank;
                    rows_seen <= (input_rows < 2) ? input_rows + 1'b1 : 2'd2;
                end else begin
                    x_count   <= input_x + 1'b1;
                    row_bank  <= input_bank;
                    rows_seen <= input_rows;
                end
            end
        end

        if (gradient_ready) begin
            gradient_valid <= pending_valid;
            if (pending_valid) begin
                gradient_gx <= gx;
                gradient_gy <= gy;
                gradient_gray <= pending_gray;
                gradient_threshold <= pending_threshold;
                gradient_interior <= (pending_rows >= 2) && (pending_x >= 2);
                gradient_sof <= pending_sof;
                gradient_eol <= pending_eol;

                // Clear horizontal history only when this row's first pixel
                // reaches the window stage, never when a future SOF is queued.
                top_left      <= (pending_x == 0) ? 8'd0 : top_center;
                middle_left   <= (pending_x == 0) ? 8'd0 : middle_center;
                bottom_left   <= (pending_x == 0) ? 8'd0 : bottom_center;
                top_center    <= top_pixel;
                middle_center <= middle_pixel;
                bottom_center <= pending_gray;
            end
        end

        if (output_ready) begin
            m_valid <= gradient_valid;
            if (gradient_valid) begin
                m_gray <= gradient_gray;
                m_sof <= gradient_sof;
                m_eol <= gradient_eol;
                m_gx <= gradient_interior ? $signed(gradient_gx) : 12'sd0;
                m_gy <= gradient_interior ? $signed(gradient_gy) : 12'sd0;
                m_magnitude <= gradient_interior ? clipped_l1 : 8'd0;
                m_edge <= gradient_interior && (clipped_l1 >= gradient_threshold);
            end
        end
    end
end

// synthesis translate_off
initial begin
    if (FRAME_WIDTH < 3)
        $fatal(1, "vision_sobel3x3 requires FRAME_WIDTH >= 3");
end
// synthesis translate_on

endmodule
