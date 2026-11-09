`timescale 1ns / 1ps

// Dedicated diagnostic voices go only to the analyser, never the headphones.
module audio_analysis_test_source (
    input wire clk,
    input wire rst,
    input wire [1:0] mode,
    input wire signed [15:0] live_sample,
    input wire live_valid,
    output wire signed [15:0] sample,
    output wire valid
);
wire tick;
wire signed [15:0] a3, c4, e4, g4, a4;
wire a3_valid, c4_valid, e4_valid, g4_valid, a4_valid;
wire signed [17:0] c_major = {{2{c4[15]}},c4} + {{2{e4[15]}},e4} + {{2{g4[15]}},g4};
wire signed [17:0] a_minor = {{2{a3[15]}},a3} + {{2{c4[15]}},c4} + {{2{e4[15]}},e4};
audio_sample_tick tick_inst(.clk(clk),.rst(rst),.sample_tick(tick));
audio_test_tone #(.PHASE_STEP(32'd19685267)) a3_inst(.clk(clk),.rst(rst),.sample_tick(tick),.pcm_sample(a3),.sample_valid(a3_valid));
audio_test_tone #(.PHASE_STEP(32'd23409859)) c4_inst(.clk(clk),.rst(rst),.sample_tick(tick),.pcm_sample(c4),.sample_valid(c4_valid));
audio_test_tone #(.PHASE_STEP(32'd29494575)) e4_inst(.clk(clk),.rst(rst),.sample_tick(tick),.pcm_sample(e4),.sample_valid(e4_valid));
audio_test_tone #(.PHASE_STEP(32'd35075158)) g4_inst(.clk(clk),.rst(rst),.sample_tick(tick),.pcm_sample(g4),.sample_valid(g4_valid));
audio_test_tone #(.PHASE_STEP(32'd39370534)) a4_inst(.clk(clk),.rst(rst),.sample_tick(tick),.pcm_sample(a4),.sample_valid(a4_valid));
assign sample = mode == 0 ? live_sample : mode == 1 ? a4 :
                mode == 2 ? (c_major >>> 2) : (a_minor >>> 2);
assign valid = mode == 0 ? live_valid : a4_valid;
endmodule
