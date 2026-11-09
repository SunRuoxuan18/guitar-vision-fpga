`timescale 1ns / 1ps

module tb_audio_status_axi_lite;

reg clk;
reg resetn;

reg  [5:0]  awaddr;
reg         awvalid;
wire        awready;
reg  [31:0] wdata;
reg  [3:0]  wstrb;
reg         wvalid;
wire        wready;
wire [1:0]  bresp;
wire        bvalid;
reg         bready;

reg  [5:0]  araddr;
reg         arvalid;
wire        arready;
wire [31:0] rdata;
wire [1:0]  rresp;
wire        rvalid;
reg         rready;

reg [15:0] peak_level;
reg signal_active;
reg [31:0] frequency_x100;
reg pitch_valid;
reg note_valid;
reg [6:0] midi_note;
reg [3:0] note_class;
reg [3:0] note_octave;
reg [31:0] target_frequency_x100;
reg signed [31:0] frequency_error_x100;
reg frame_ready;
reg [31:0] frame_number;

wire use_test_tone;
wire [2:0] gain_shift;

integer error_count;
reg [31:0] read_value;

always #5 clk = ~clk;

audio_status_axi_lite dut (
    .s_axi_aclk(clk),
    .s_axi_aresetn(resetn),
    .s_axi_awaddr(awaddr),
    .s_axi_awvalid(awvalid),
    .s_axi_awready(awready),
    .s_axi_wdata(wdata),
    .s_axi_wstrb(wstrb),
    .s_axi_wvalid(wvalid),
    .s_axi_wready(wready),
    .s_axi_bresp(bresp),
    .s_axi_bvalid(bvalid),
    .s_axi_bready(bready),
    .s_axi_araddr(araddr),
    .s_axi_arvalid(arvalid),
    .s_axi_arready(arready),
    .s_axi_rdata(rdata),
    .s_axi_rresp(rresp),
    .s_axi_rvalid(rvalid),
    .s_axi_rready(rready),
    .peak_level(peak_level),
    .signal_active(signal_active),
    .frequency_x100(frequency_x100),
    .pitch_valid(pitch_valid),
    .note_valid(note_valid),
    .midi_note(midi_note),
    .note_class(note_class),
    .note_octave(note_octave),
    .target_frequency_x100(target_frequency_x100),
    .frequency_error_x100(frequency_error_x100),
    .frame_ready(frame_ready),
    .frame_number(frame_number),
    .use_test_tone(use_test_tone),
    .gain_shift(gain_shift)
);

task axi_write;
    input [5:0] address;
    input [31:0] value;
    begin
        @(negedge clk);
        awaddr  = address;
        awvalid = 1'b1;
        wdata   = value;
        wstrb   = 4'hf;
        wvalid  = 1'b1;

        while (!(awready && wready))
            @(negedge clk);

        @(negedge clk);
        awvalid = 1'b0;
        wvalid  = 1'b0;

        while (!bvalid)
            @(negedge clk);

        if (bresp != 2'b00) begin
            error_count = error_count + 1;
            $display("ERROR: AXI write response was %b", bresp);
        end
    end
endtask

task axi_read;
    input [5:0] address;
    output [31:0] value;
    begin
        @(negedge clk);
        araddr  = address;
        arvalid = 1'b1;

        while (!arready)
            @(negedge clk);

        @(negedge clk);
        arvalid = 1'b0;

        while (!rvalid)
            @(negedge clk);

        value = rdata;
        if (rresp != 2'b00) begin
            error_count = error_count + 1;
            $display("ERROR: AXI read response was %b", rresp);
        end
    end
endtask

task check_equal;
    input [31:0] actual;
    input [31:0] expected;
    input [8*32-1:0] label_text;
    begin
        if (actual !== expected) begin
            error_count = error_count + 1;
            $display("ERROR: %0s actual=0x%08x expected=0x%08x",
                     label_text, actual, expected);
        end
    end
endtask

initial begin
    clk = 1'b0;
    resetn = 1'b0;
    awaddr = 6'd0;
    awvalid = 1'b0;
    wdata = 32'd0;
    wstrb = 4'd0;
    wvalid = 1'b0;
    bready = 1'b1;
    araddr = 6'd0;
    arvalid = 1'b0;
    rready = 1'b1;

    peak_level = 16'd0;
    signal_active = 1'b0;
    frequency_x100 = 32'd0;
    pitch_valid = 1'b0;
    note_valid = 1'b0;
    midi_note = 7'd0;
    note_class = 4'd0;
    note_octave = 4'd0;
    target_frequency_x100 = 32'd0;
    frequency_error_x100 = 32'sd0;
    frame_ready = 1'b0;
    frame_number = 32'd0;
    error_count = 0;

    repeat (5) @(posedge clk);
    resetn = 1'b1;
    repeat (2) @(posedge clk);

    axi_read(6'h00, read_value);
    check_equal(read_value, 32'h0000_0001, "reset control");

    // use_test_tone=1, gain_shift=2.
    axi_write(6'h00, 32'h0000_0005);
    if (!use_test_tone || gain_shift != 3'd2) begin
        error_count = error_count + 1;
        $display("ERROR: control outputs tone=%b gain=%0d",
                 use_test_tone, gain_shift);
    end

    @(negedge clk);
    signal_active = 1'b1;
    frequency_x100 = 32'd44000;
    pitch_valid = 1'b1;
    @(negedge clk);
    pitch_valid = 1'b0;

    peak_level = 16'd24576;
    midi_note = 7'd69;
    note_class = 4'd9;
    note_octave = 4'd4;
    target_frequency_x100 = 32'd44000;
    frequency_error_x100 = -32'sd23;
    note_valid = 1'b1;
    @(negedge clk);
    note_valid = 1'b0;

    frame_number = 32'd7;
    frame_ready = 1'b1;
    @(negedge clk);
    frame_ready = 1'b0;

    axi_read(6'h04, read_value);
    check_equal(read_value, 32'h0000_0507, "status flags");
    axi_read(6'h08, read_value);
    check_equal(read_value, 32'd44000, "frequency");
    axi_read(6'h0c, read_value);
    check_equal(read_value, 32'h0000_4945, "note information");
    axi_read(6'h10, read_value);
    check_equal(read_value, 32'd44000, "target frequency");
    axi_read(6'h14, read_value);
    check_equal(read_value, -32'sd23, "frequency error");
    axi_read(6'h18, read_value);
    check_equal(read_value, 32'd24576, "peak level");
    axi_read(6'h1c, read_value);
    check_equal(read_value, 32'd7, "frame number");
    axi_read(6'h20, read_value);
    check_equal(read_value, 32'd1, "result sequence");
    axi_read(6'h24, read_value);
    check_equal(read_value, 32'd1, "frame sequence");
    axi_read(6'h28, read_value);
    check_equal(read_value, 32'h4155_4430, "core ID");

    // Writing ones to STATUS[2:1] clears the two pending flags.
    axi_write(6'h04, 32'h0000_0006);
    axi_read(6'h04, read_value);
    check_equal(read_value, 32'h0000_0501, "cleared status flags");

    if (error_count == 0)
        $display("PASS: AXI-Lite audio status bridge verified.");
    else
        $display("FAIL: %0d AXI-Lite bridge check(s) failed.", error_count);

    $finish;
end

endmodule
