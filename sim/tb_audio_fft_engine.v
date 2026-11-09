`timescale 1ns / 1ps
module tb_audio_fft_engine;
reg clk=0, rst=1, pcm_valid=0, freeze=0;
reg signed [15:0] pcm_sample=0;
wire busy, frame_start, frame_done, bin_valid, fft_error;
wire [11:0] bin_index;
wire [31:0] bin_power, drops;
reg [15:0] samples[0:4095];
reg [63:0] reference[0:4095];
reg [31:0] measured[0:2047];
integer i, count, frame_number, failures=0;
real re, im, expected, difference, tolerance;
reg [255:0] case_name;
always #4 clk=~clk;
audio_fft_engine dut (
    .clk(clk), .rst(rst), .pcm_sample(pcm_sample), .pcm_valid(pcm_valid),
    .freeze(freeze), .peaks_busy(1'b0), .busy(busy),
    .frame_start(frame_start), .frame_done(frame_done),
    .bin_valid(bin_valid), .bin_index(bin_index), .bin_power(bin_power),
    .dropped_frames(drops), .fft_error(fft_error)
);
always @(posedge clk) begin
    if(rst) count=0;
    else if(bin_valid) begin
        if(bin_index !== count) $fatal(1,"Bin index discontinuity %0d != %0d",bin_index,count);
        measured[bin_index]=bin_power;
        count=count+1;
    end
end
task run_case;
input [255:0] name;
begin
    rst=1; freeze=0; pcm_valid=0;
    repeat(8) @(negedge clk);
    $readmemh($sformatf("D:/FPGA/ZynqAudioLab/sim/vectors/%0s_pcm.mem",name),samples);
    $readmemh($sformatf("D:/FPGA/ZynqAudioLab/sim/vectors/%0s_fft.mem",name),reference);
    rst=0;
    for(i=0;i<4096;i=i+1) begin
        @(negedge clk); pcm_sample=samples[i]; pcm_valid=1;
        @(negedge clk); pcm_valid=0;
    end
    wait(frame_start); @(negedge clk); freeze=1;
    wait(frame_done); @(negedge clk);
    if(count != 2048 || fft_error || drops) $fatal(1,"FFT framing error count=%0d error=%b drops=%0d",count,fft_error,drops);
    for(i=0;i<2048;i=i+1) begin
        re=$signed(reference[i][31:0]); im=$signed(reference[i][63:32]);
        expected=(re*re+im*im)/1048576.0;
        if ((^measured[i]) === 1'bx || (^reference[i]) === 1'bx)
            $fatal(1,"Unknown numerical data: vector load or FFT failure");
        difference=measured[i]-expected;
        if(difference<0) difference=-difference;
        tolerance=expected*0.015+16.0;
        if(difference>tolerance) begin
            failures=failures+1;
            if(failures<8) $display("MISMATCH %0s bin%0d power=%0d ref=%f",name,i,measured[i],expected);
        end
    end
    $display("CASE: %0s all 2048 bins checked against independent NumPy reference",name);
end
endtask
initial begin
    run_case("silence"); run_case("a4"); run_case("c_major"); run_case("a_minor");
    if(failures) $fatal(1,"FFT numerical checks failed: %0d",failures);
    $display("PASS: real AMD4096 FFT, Hann window, silence/mono/polyphonic numerical checks");
    $finish;
end
initial begin #20000000; $fatal(1,"FFT test timeout"); end
endmodule
