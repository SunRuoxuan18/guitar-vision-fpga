`timescale 1ns / 1ps

module tb_pitch_note_decoder;

reg clk;
reg rst;
reg pitch_valid;
reg [31:0] frequency_x100;
wire busy;
wire note_valid;
wire [6:0] midi_note;
wire [3:0] note_class;
wire [3:0] octave;
wire [31:0] target_frequency_x100;
wire signed [31:0] frequency_error_x100;
integer error_count;

pitch_note_decoder dut (
    .clk(clk),
    .rst(rst),
    .pitch_valid(pitch_valid),
    .frequency_x100(frequency_x100),
    .busy(busy),
    .note_valid(note_valid),
    .midi_note(midi_note),
    .note_class(note_class),
    .octave(octave),
    .target_frequency_x100(target_frequency_x100),
    .frequency_error_x100(frequency_error_x100)
);

always #5 clk = ~clk;

task check_note;
    input [31:0] measured;
    input [6:0] expected_midi;
    input [3:0] expected_class;
    input [3:0] expected_octave;
    input [31:0] expected_target;
    begin
        @(negedge clk);
        frequency_x100 = measured;
        pitch_valid = 1'b1;
        @(negedge clk);
        pitch_valid = 1'b0;
        wait (note_valid);
        #1;
        if ((midi_note !== expected_midi) ||
            (note_class !== expected_class) ||
            (octave !== expected_octave) ||
            (target_frequency_x100 !== expected_target)) begin
            error_count = error_count + 1;
            $display("ERROR: measured=%0d midi=%0d class=%0d octave=%0d target=%0d",
                     measured, midi_note, note_class, octave,
                     target_frequency_x100);
        end
    end
endtask

initial begin
    clk = 1'b0;
    rst = 1'b1;
    pitch_valid = 1'b0;
    frequency_x100 = 32'd0;
    error_count = 0;

    repeat (4) @(posedge clk);
    rst = 1'b0;

    check_note(32'd8241,  7'd40, 4'd4, 4'd2, 32'd8241);  // E2
    check_note(32'd11030, 7'd45, 4'd9, 4'd2, 32'd11000); // A2, slightly sharp
    check_note(32'd44444, 7'd69, 4'd9, 4'd4, 32'd44000); // A4
    check_note(32'd32920, 7'd64, 4'd4, 4'd4, 32'd32963); // E4

    if (error_count == 0)
        $display("PASS: pitch-to-note decoder verified.");
    else
        $display("FAIL: %0d note decoder check(s) failed.", error_count);
    $finish;
end

endmodule
