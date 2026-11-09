`timescale 1ns / 1ps

module audio_test_tone #(
    parameter [31:0] PHASE_STEP = 32'd39370534
)(
    input  wire               clk,
    input  wire               rst,
    input  wire               sample_tick,
    output reg  signed [15:0] pcm_sample,
    output reg                sample_valid
);

reg [31:0] phase_accumulator;
wire [5:0] lut_address = phase_accumulator[31:26];

function signed [15:0] sine_lut;
    input [5:0] address;
    begin
        case (address)
            6'd0:  sine_lut = 16'sd0;
            6'd1:  sine_lut = 16'sd1606;
            6'd2:  sine_lut = 16'sd3196;
            6'd3:  sine_lut = 16'sd4756;
            6'd4:  sine_lut = 16'sd6270;
            6'd5:  sine_lut = 16'sd7723;
            6'd6:  sine_lut = 16'sd9102;
            6'd7:  sine_lut = 16'sd10394;
            6'd8:  sine_lut = 16'sd11585;
            6'd9:  sine_lut = 16'sd12665;
            6'd10: sine_lut = 16'sd13623;
            6'd11: sine_lut = 16'sd14449;
            6'd12: sine_lut = 16'sd15137;
            6'd13: sine_lut = 16'sd15679;
            6'd14: sine_lut = 16'sd16069;
            6'd15: sine_lut = 16'sd16305;
            6'd16: sine_lut = 16'sd16384;
            6'd17: sine_lut = 16'sd16305;
            6'd18: sine_lut = 16'sd16069;
            6'd19: sine_lut = 16'sd15679;
            6'd20: sine_lut = 16'sd15137;
            6'd21: sine_lut = 16'sd14449;
            6'd22: sine_lut = 16'sd13623;
            6'd23: sine_lut = 16'sd12665;
            6'd24: sine_lut = 16'sd11585;
            6'd25: sine_lut = 16'sd10394;
            6'd26: sine_lut = 16'sd9102;
            6'd27: sine_lut = 16'sd7723;
            6'd28: sine_lut = 16'sd6270;
            6'd29: sine_lut = 16'sd4756;
            6'd30: sine_lut = 16'sd3196;
            6'd31: sine_lut = 16'sd1606;
            6'd32: sine_lut = 16'sd0;
            6'd33: sine_lut = -16'sd1606;
            6'd34: sine_lut = -16'sd3196;
            6'd35: sine_lut = -16'sd4756;
            6'd36: sine_lut = -16'sd6270;
            6'd37: sine_lut = -16'sd7723;
            6'd38: sine_lut = -16'sd9102;
            6'd39: sine_lut = -16'sd10394;
            6'd40: sine_lut = -16'sd11585;
            6'd41: sine_lut = -16'sd12665;
            6'd42: sine_lut = -16'sd13623;
            6'd43: sine_lut = -16'sd14449;
            6'd44: sine_lut = -16'sd15137;
            6'd45: sine_lut = -16'sd15679;
            6'd46: sine_lut = -16'sd16069;
            6'd47: sine_lut = -16'sd16305;
            6'd48: sine_lut = -16'sd16384;
            6'd49: sine_lut = -16'sd16305;
            6'd50: sine_lut = -16'sd16069;
            6'd51: sine_lut = -16'sd15679;
            6'd52: sine_lut = -16'sd15137;
            6'd53: sine_lut = -16'sd14449;
            6'd54: sine_lut = -16'sd13623;
            6'd55: sine_lut = -16'sd12665;
            6'd56: sine_lut = -16'sd11585;
            6'd57: sine_lut = -16'sd10394;
            6'd58: sine_lut = -16'sd9102;
            6'd59: sine_lut = -16'sd7723;
            6'd60: sine_lut = -16'sd6270;
            6'd61: sine_lut = -16'sd4756;
            6'd62: sine_lut = -16'sd3196;
            default: sine_lut = -16'sd1606;
        endcase
    end
endfunction

always @(posedge clk) begin
    if (rst) begin
        phase_accumulator <= 32'd0;
        pcm_sample        <= 16'sd0;
        sample_valid      <= 1'b0;
    end else begin
        sample_valid <= 1'b0;

        if (sample_tick) begin
            pcm_sample        <= sine_lut(lut_address);
            phase_accumulator <= phase_accumulator + PHASE_STEP;
            sample_valid      <= 1'b1;
        end
    end
end

endmodule
