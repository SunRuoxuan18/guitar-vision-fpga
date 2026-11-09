`timescale 1ns / 1ps
module tb_audio_fft_capture;
reg clk=0, rst=1, valid=0, release_valid=0, release_bank=0;
reg read_enable=0, read_bank=0;
reg signed [15:0] sample_in=0;
reg [3:0] read_address=0;
wire [1:0] ready;
wire signed [15:0] read_data;
wire [31:0] drops;
integer i;
always #4 clk=~clk;
audio_fft_capture #(.ADDR_WIDTH(4)) dut (
    .clk(clk), .rst(rst), .sample_valid(valid), .sample_in(sample_in),
    .bank_ready(ready), .release_valid(release_valid), .release_bank(release_bank),
    .read_enable(read_enable), .read_bank(read_bank), .read_address(read_address),
    .read_data(read_data), .dropped_frames(drops)
);
task samples;
input integer base;
begin
    for (i=0;i<16;i=i+1) begin
        @(negedge clk); valid=1; sample_in=base+i;
    end
    @(negedge clk); valid=0;
end
endtask
task check_bank;
input integer bank;
input integer base;
begin
    for(i=0;i<16;i=i+1) begin
        @(negedge clk); read_enable=1; read_bank=bank; read_address=i;
        @(posedge clk); #1;
        if(read_data !== base+i) $fatal(1,"Stored frame corrupted bank=%0d addr=%0d value=%0d",bank,i,read_data);
    end
    @(negedge clk); read_enable=0;
end
endtask
initial begin
    repeat(4) @(negedge clk); rst=0;
    samples(100); samples(200);
    if(ready !== 2'b11) $fatal(1,"Both frames must be ready");
    samples(300);
    if(drops !== 1) $fatal(1,"Overrun must count one complete discarded frame");
    check_bank(0,100); check_bank(1,200);
    @(negedge clk); release_valid=1; release_bank=0;
    @(negedge clk); release_valid=0;
    // Discard a frame already begun during overrun; next boundary resumes.
    samples(400); samples(500);
    check_bank(0,500);
    if(drops !== 2) $fatal(1,"Drops must be counted per whole frame");
    $display("PASS: ping-pong capture, synchronous read and overrun recovery");
    $finish;
end
initial begin #100000; $fatal(1,"Capture test timeout"); end
endmodule
