`timescale 1ns / 1ps

// Converts a measured frequency into the nearest equal-tempered MIDI note.
// The supported range is MIDI 40 (E2) through MIDI 88 (E6), covering the
// normal electric-guitar range with room for bends and higher frets.
module pitch_note_decoder (
    input  wire               clk,
    input  wire               rst,
    input  wire               pitch_valid,
    input  wire [31:0]        frequency_x100,
    output reg                busy,
    output reg                note_valid,
    output reg  [6:0]         midi_note,
    output wire [3:0]         note_class,
    output wire [3:0]         octave,
    output reg  [31:0]        target_frequency_x100,
    output reg  signed [31:0] frequency_error_x100
);

reg [31:0] measured_frequency;
reg [6:0]  scan_note;
reg [6:0]  best_note;
reg [31:0] best_frequency;
reg [31:0] best_difference;

wire [31:0] scan_frequency = midi_frequency_x100(scan_note);
wire [31:0] scan_difference =
    (measured_frequency >= scan_frequency) ?
    (measured_frequency - scan_frequency) :
    (scan_frequency - measured_frequency);
wire scan_is_better = scan_difference < best_difference;

assign note_class = midi_note % 12;
assign octave     = (midi_note / 12) - 1;

function [31:0] midi_frequency_x100;
    input [6:0] note_number;
    begin
        case (note_number)
            7'd40: midi_frequency_x100 = 32'd8241;
            7'd41: midi_frequency_x100 = 32'd8731;
            7'd42: midi_frequency_x100 = 32'd9250;
            7'd43: midi_frequency_x100 = 32'd9800;
            7'd44: midi_frequency_x100 = 32'd10383;
            7'd45: midi_frequency_x100 = 32'd11000;
            7'd46: midi_frequency_x100 = 32'd11654;
            7'd47: midi_frequency_x100 = 32'd12347;
            7'd48: midi_frequency_x100 = 32'd13081;
            7'd49: midi_frequency_x100 = 32'd13859;
            7'd50: midi_frequency_x100 = 32'd14683;
            7'd51: midi_frequency_x100 = 32'd15556;
            7'd52: midi_frequency_x100 = 32'd16481;
            7'd53: midi_frequency_x100 = 32'd17461;
            7'd54: midi_frequency_x100 = 32'd18500;
            7'd55: midi_frequency_x100 = 32'd19600;
            7'd56: midi_frequency_x100 = 32'd20765;
            7'd57: midi_frequency_x100 = 32'd22000;
            7'd58: midi_frequency_x100 = 32'd23308;
            7'd59: midi_frequency_x100 = 32'd24694;
            7'd60: midi_frequency_x100 = 32'd26163;
            7'd61: midi_frequency_x100 = 32'd27718;
            7'd62: midi_frequency_x100 = 32'd29366;
            7'd63: midi_frequency_x100 = 32'd31113;
            7'd64: midi_frequency_x100 = 32'd32963;
            7'd65: midi_frequency_x100 = 32'd34923;
            7'd66: midi_frequency_x100 = 32'd36999;
            7'd67: midi_frequency_x100 = 32'd39200;
            7'd68: midi_frequency_x100 = 32'd41530;
            7'd69: midi_frequency_x100 = 32'd44000;
            7'd70: midi_frequency_x100 = 32'd46616;
            7'd71: midi_frequency_x100 = 32'd49388;
            7'd72: midi_frequency_x100 = 32'd52325;
            7'd73: midi_frequency_x100 = 32'd55437;
            7'd74: midi_frequency_x100 = 32'd58733;
            7'd75: midi_frequency_x100 = 32'd62225;
            7'd76: midi_frequency_x100 = 32'd65926;
            7'd77: midi_frequency_x100 = 32'd69846;
            7'd78: midi_frequency_x100 = 32'd73999;
            7'd79: midi_frequency_x100 = 32'd78399;
            7'd80: midi_frequency_x100 = 32'd83061;
            7'd81: midi_frequency_x100 = 32'd88000;
            7'd82: midi_frequency_x100 = 32'd93233;
            7'd83: midi_frequency_x100 = 32'd98777;
            7'd84: midi_frequency_x100 = 32'd104650;
            7'd85: midi_frequency_x100 = 32'd110873;
            7'd86: midi_frequency_x100 = 32'd117466;
            7'd87: midi_frequency_x100 = 32'd124451;
            7'd88: midi_frequency_x100 = 32'd131851;
            default: midi_frequency_x100 = 32'd0;
        endcase
    end
endfunction

always @(posedge clk) begin
    if (rst) begin
        measured_frequency   <= 32'd0;
        scan_note            <= 7'd40;
        best_note            <= 7'd40;
        best_frequency       <= 32'd8241;
        best_difference      <= 32'hffffffff;
        busy                 <= 1'b0;
        note_valid           <= 1'b0;
        midi_note            <= 7'd0;
        target_frequency_x100<= 32'd0;
        frequency_error_x100 <= 32'sd0;
    end else begin
        note_valid <= 1'b0;

        if (pitch_valid && !busy) begin
            measured_frequency <= frequency_x100;
            scan_note           <= 7'd40;
            best_note           <= 7'd40;
            best_frequency      <= 32'd8241;
            best_difference     <= 32'hffffffff;
            busy                <= 1'b1;
        end else if (busy) begin
            if (scan_is_better) begin
                best_note       <= scan_note;
                best_frequency  <= scan_frequency;
                best_difference <= scan_difference;
            end

            if (scan_note == 7'd88) begin
                if (scan_is_better) begin
                    midi_note             <= scan_note;
                    target_frequency_x100 <= scan_frequency;
                    frequency_error_x100  <= $signed(measured_frequency) -
                                              $signed(scan_frequency);
                end else begin
                    midi_note             <= best_note;
                    target_frequency_x100 <= best_frequency;
                    frequency_error_x100  <= $signed(measured_frequency) -
                                              $signed(best_frequency);
                end
                busy       <= 1'b0;
                note_valid <= 1'b1;
            end else begin
                scan_note <= scan_note + 1'b1;
            end
        end
    end
end

endmodule
