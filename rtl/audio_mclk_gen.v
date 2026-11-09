`timescale 1ns / 1ps

// Generates the ADAU1761 10 MHz master clock from a stable 125 MHz input.
// The original board top uses the Ethernet PHY; the diagnostic uses PS FCLK0.
module audio_mclk_gen (
    input  wire clk_125mhz,
    input  wire rst,
    output wire codec_mclk,
    output wire locked
);

wire clkfb_raw;
wire clkfb_buf;
wire clk_10mhz_raw;
wire clk_10mhz;
wire codec_mclk_oddr;

MMCME2_BASE #(
    .BANDWIDTH          ("OPTIMIZED"),
    .CLKIN1_PERIOD      (8.000),
    .DIVCLK_DIVIDE      (5),
    .CLKFBOUT_MULT_F    (40.000),
    .CLKOUT0_DIVIDE_F   (100.000),
    .CLKOUT0_DUTY_CYCLE (0.500),
    .STARTUP_WAIT       ("FALSE")
) mmcm_inst (
    .CLKIN1   (clk_125mhz),
    .RST      (rst),
    .PWRDWN   (1'b0),
    .CLKFBIN  (clkfb_buf),
    .CLKFBOUT (clkfb_raw),
    .CLKOUT0  (clk_10mhz_raw),
    .LOCKED   (locked)
);

BUFG feedback_buf (
    .I(clkfb_raw),
    .O(clkfb_buf)
);

BUFG mclk_buf (
    .I(clk_10mhz_raw),
    .O(clk_10mhz)
);

ODDR #(
    .DDR_CLK_EDGE("SAME_EDGE")
) mclk_output_ddr (
    .Q  (codec_mclk_oddr),
    .C  (clk_10mhz),
    .CE (1'b1),
    .D1 (1'b1),
    .D2 (1'b0),
    .R  (1'b0),
    .S  (1'b0)
);

OBUF mclk_output_buf (
    .I(codec_mclk_oddr),
    .O(codec_mclk)
);

endmodule
