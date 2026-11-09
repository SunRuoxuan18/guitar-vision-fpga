`timescale 1ns / 1ps

// RGB888 is {R, G, B}. Two elastic stages sustain one pixel per clock.
module vision_rgb_gray (
    input  wire        clk,
    input  wire        rst,
    input  wire [23:0] s_rgb,
    input  wire        s_valid,
    output wire        s_ready,
    input  wire        s_sof,
    input  wire        s_eol,
    output reg  [7:0]  m_gray,
    output reg         m_valid,
    input  wire        m_ready,
    output reg         m_sof,
    output reg         m_eol
);

reg [14:0] red_term;
reg [15:0] green_term;
reg [12:0] blue_term;
reg terms_valid, terms_sof, terms_eol;
wire [15:0] red_blue = {1'b0, red_term} + {3'b0, blue_term};
wire [15:0] green_rounded = green_term + 16'd128;
wire [15:0] rounded_sum = red_blue + green_rounded;

wire output_ready = !m_valid || m_ready;
assign s_ready = !rst && (!terms_valid || output_ready);

always @(posedge clk) begin
    if (rst) begin
        red_term <= 0;
        green_term <= 0;
        blue_term <= 0;
        terms_valid <= 0;
        terms_sof <= 0;
        terms_eol <= 0;
        m_gray  <= 8'd0;
        m_valid <= 1'b0;
        m_sof   <= 1'b0;
        m_eol   <= 1'b0;
    end else begin
        // Register exact constant products before the balanced rounded sum.
        if (s_ready) begin
            terms_valid <= s_valid;
            if (s_valid) begin
                red_term <= s_rgb[23:16] * 15'd77;
                green_term <= s_rgb[15:8] * 16'd150;
                blue_term <= s_rgb[7:0] * 13'd29;
                terms_sof <= s_sof;
                terms_eol <= s_eol;
            end
        end
        if (output_ready) begin
            m_valid <= terms_valid;
            if (terms_valid) begin
                m_gray <= rounded_sum[15:8];
                m_sof <= terms_sof;
                m_eol <= terms_eol;
            end
        end
    end
end

endmodule
