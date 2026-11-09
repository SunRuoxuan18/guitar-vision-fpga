`timescale 1ns / 1ps
module audio_music_pl (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 clk CLK",
       X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF S_AXI, ASSOCIATED_RESET rstn, FREQ_HZ 125000000" *)
    input wire clk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 rstn RST",
       X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input wire rstn,
    input wire rst_btn, source_sw, gain_sw,
    output wire codec_mclk,
    input wire audio_bclk, audio_lrclk, audio_sdata_in,
    output wire audio_sdata_out,
    inout wire audio_i2c_scl, audio_i2c_sda,
    output wire [1:0] codec_addr,
    output wire [3:0] led,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWADDR",
       X_INTERFACE_PARAMETER = "PROTOCOL AXI4LITE, ADDR_WIDTH 14, DATA_WIDTH 32, HAS_BURST 0" *)
    input wire [13:0] s_axi_awaddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWVALID" *) input wire s_axi_awvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWREADY" *) output wire s_axi_awready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WDATA" *) input wire [31:0] s_axi_wdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WSTRB" *) input wire [3:0] s_axi_wstrb,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WVALID" *) input wire s_axi_wvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WREADY" *) output wire s_axi_wready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BRESP" *) output wire [1:0] s_axi_bresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BVALID" *) output wire s_axi_bvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BREADY" *) input wire s_axi_bready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARADDR" *) input wire [13:0] s_axi_araddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARVALID" *) input wire s_axi_arvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARREADY" *) output wire s_axi_arready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RDATA" *) output wire [31:0] s_axi_rdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RRESP" *) output wire [1:0] s_axi_rresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RVALID" *) output wire s_axi_rvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RREADY" *) input wire s_axi_rready
);
wire signed [15:0] pcm;
wire pcm_valid, audio_reset;
wire freeze, peaks_busy, fft_busy, frame_start, frame_done, fft_error;
wire bin_valid;
wire [11:0] bin_index;
wire [31:0] bin_power, dropped_frames;
wire analysis_rst = !rstn || audio_reset;
wire [1:0] analysis_test_mode;
reg [1:0] previous_test_mode;
always @(posedge clk) begin
    if (analysis_rst) previous_test_mode <= 0;
    else previous_test_mode <= analysis_test_mode;
end
wire source_changed = previous_test_mode != analysis_test_mode;
reg [3:0] mode_reset_pipe = 4'hf;
always @(posedge clk) begin
    if (analysis_rst || source_changed) mode_reset_pipe <= 4'hf;
    else mode_reset_pipe <= {mode_reset_pipe[2:0],1'b0};
end
wire signed [15:0] selected_pcm;
wire selected_valid;
audio_analysis_test_source diagnostic_inst (
    .clk(clk), .rst(analysis_rst), .mode(analysis_test_mode),
    .live_sample(pcm), .live_valid(pcm_valid),
    .sample(selected_pcm), .valid(selected_valid)
);

pynq_audio_board_top board_inst (
    .sys_clk(clk), .rst_btn(rst_btn), .source_sw(source_sw), .gain_sw(gain_sw),
    .codec_mclk(codec_mclk), .audio_bclk(audio_bclk), .audio_lrclk(audio_lrclk),
    .audio_sdata_in(audio_sdata_in), .audio_sdata_out(audio_sdata_out),
    .audio_i2c_scl(audio_i2c_scl), .audio_i2c_sda(audio_i2c_sda),
    .codec_addr(codec_addr), .led(led), .analysis_pcm_sample(pcm),
    .analysis_pcm_valid(pcm_valid), .analysis_reset(audio_reset)
);
audio_fft_engine engine_inst (
    .clk(clk), .rst(analysis_rst || mode_reset_pipe[3]), .pcm_sample(selected_pcm), .pcm_valid(selected_valid),
    .freeze(freeze), .peaks_busy(peaks_busy), .busy(fft_busy),
    .frame_start(frame_start), .frame_done(frame_done),
    .bin_valid(bin_valid), .bin_index(bin_index), .bin_power(bin_power),
    .dropped_frames(dropped_frames), .fft_error(fft_error)
);
audio_spectrum_axi spectrum_inst (
    .clk(clk), .rst(analysis_rst),
    .s_axi_awaddr(s_axi_awaddr), .s_axi_awvalid(s_axi_awvalid), .s_axi_awready(s_axi_awready),
    .s_axi_wdata(s_axi_wdata), .s_axi_wstrb(s_axi_wstrb), .s_axi_wvalid(s_axi_wvalid),
    .s_axi_wready(s_axi_wready), .s_axi_bresp(s_axi_bresp), .s_axi_bvalid(s_axi_bvalid),
    .s_axi_bready(s_axi_bready), .s_axi_araddr(s_axi_araddr),
    .s_axi_arvalid(s_axi_arvalid), .s_axi_arready(s_axi_arready),
    .s_axi_rdata(s_axi_rdata), .s_axi_rresp(s_axi_rresp),
    .s_axi_rvalid(s_axi_rvalid), .s_axi_rready(s_axi_rready),
    .bin_valid(bin_valid), .bin_index(bin_index), .bin_power(bin_power),
    .fft_busy(fft_busy), .fft_error(fft_error), .frame_start(frame_start), .frame_done(frame_done),
    .dropped_frames(dropped_frames), .freeze(freeze), .peaks_busy(peaks_busy), .frame_hold(),
    .analysis_test_mode(analysis_test_mode)
);
endmodule
