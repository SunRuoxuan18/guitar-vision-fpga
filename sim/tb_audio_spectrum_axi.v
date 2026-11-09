`timescale 1ns / 1ps

module tb_audio_spectrum_axi;

reg clk = 1'b0;
always #5 clk = ~clk;
reg rst = 1'b1;
reg [13:0] awaddr = 14'd0, araddr = 14'd0;
reg awvalid = 1'b0, wvalid = 1'b0, arvalid = 1'b0;
reg [31:0] wdata = 32'd0;
reg [3:0] wstrb = 4'd0;
reg bready = 1'b0, rready = 1'b0;
wire awready, wready, bvalid, arready, rvalid;
wire [1:0] bresp, rresp;
wire [31:0] rdata;
reg bin_valid = 1'b0;
reg [11:0] bin_index = 12'd0;
reg [31:0] bin_power = 32'd0;
reg fft_busy = 1'b0, frame_start = 1'b0, frame_done = 1'b0;
reg fft_error = 1'b0;
reg [31:0] dropped_frames = 32'd0;
wire freeze, peaks_busy, frame_hold;
wire [1:0] analysis_test_mode;
integer checks = 0;
integer index;
reg [31:0] value;

audio_spectrum_axi #(.PEAK_THRESHOLD(32'd20)) dut (
    .clk(clk), .rst(rst),
    .s_axi_awaddr(awaddr), .s_axi_awvalid(awvalid), .s_axi_awready(awready),
    .s_axi_wdata(wdata), .s_axi_wstrb(wstrb),
    .s_axi_wvalid(wvalid), .s_axi_wready(wready),
    .s_axi_bresp(bresp), .s_axi_bvalid(bvalid), .s_axi_bready(bready),
    .s_axi_araddr(araddr), .s_axi_arvalid(arvalid), .s_axi_arready(arready),
    .s_axi_rdata(rdata), .s_axi_rresp(rresp),
    .s_axi_rvalid(rvalid), .s_axi_rready(rready),
    .bin_valid(bin_valid), .bin_index(bin_index), .bin_power(bin_power),
    .fft_busy(fft_busy), .fft_error(fft_error),
    .frame_start(frame_start), .frame_done(frame_done),
    .dropped_frames(dropped_frames), .freeze(freeze),
    .peaks_busy(peaks_busy), .frame_hold(frame_hold),
    .analysis_test_mode(analysis_test_mode)
);

task check;
    input condition;
    input [8*96-1:0] label_text;
    begin
        checks = checks + 1;
        if (condition !== 1'b1) $fatal(1, "FAIL: %0s", label_text);
    end
endtask

task equal;
    input [31:0] actual, expected;
    input [8*96-1:0] label_text;
    begin
        checks = checks + 1;
        if (actual !== expected)
            $fatal(1, "FAIL: %0s actual=%08x expected=%08x",
                   label_text, actual, expected);
    end
endtask

// Capture AW or W independently, then corrupt the live payload to prove
// that the slave uses its accepted address/data rather than the live bus.
task send_aw;
    input [13:0] address;
    begin
        @(negedge clk);
        awaddr = address;
        awvalid = 1'b1;
        @(posedge clk);
        while (!awready) @(posedge clk);
        @(negedge clk);
        awvalid = 1'b0;
        awaddr = 14'h3ffc;
    end
endtask

task send_w;
    input [31:0] data;
    input [3:0] strobes;
    begin
        @(negedge clk);
        wdata = data;
        wstrb = strobes;
        wvalid = 1'b1;
        @(posedge clk);
        while (!wready) @(posedge clk);
        @(negedge clk);
        wvalid = 1'b0;
        wdata = ~data;
        wstrb = ~strobes;
    end
endtask

task finish_write;
    input [1:0] response;
    input integer stall_cycles;
    integer cycle;
    begin
        while (!bvalid) @(negedge clk);
        equal({30'd0, bresp}, {30'd0, response}, "write response");
        // A second request cannot overwrite an unconsumed B response.
        awvalid = 1'b1;
        wvalid = 1'b1;
        for (cycle = 0; cycle < stall_cycles; cycle = cycle + 1) begin
            @(negedge clk);
            check(bvalid && !awready && !wready, "write response held under backpressure");
            equal({30'd0, bresp}, {30'd0, response}, "stalled BRESP stable");
        end
        awvalid = 1'b0;
        wvalid = 1'b0;
        bready = 1'b1;
        @(negedge clk);
        check(!bvalid, "BVALID clears after handshake");
        bready = 1'b0;
    end
endtask

task axi_write;
    input [13:0] address;
    input [31:0] data;
    input [3:0] strobes;
    input integer order; // 0=simultaneous, 1=AW first, 2=W first
    input [1:0] response;
    begin
        if (order == 1) begin
            send_aw(address);
            repeat (5) begin
                @(negedge clk);
                check(!bvalid && !awready && wready, "AW waiting for independent W");
            end
            send_w(data, strobes);
        end else if (order == 2) begin
            send_w(data, strobes);
            repeat (5) begin
                @(negedge clk);
                check(!bvalid && awready && !wready, "W waiting for independent AW");
            end
            send_aw(address);
        end else begin
            fork
                send_aw(address);
                send_w(data, strobes);
            join
        end
        finish_write(response, 4);
    end
endtask

task axi_read;
    input [13:0] address;
    input [1:0] response;
    input integer stall_cycles;
    output [31:0] data;
    integer cycle;
    begin
        @(negedge clk);
        araddr = address;
        arvalid = 1'b1;
        @(posedge clk);
        while (!arready) @(posedge clk);
        @(negedge clk);
        check(!rvalid && !arready, "AR starts explicit synchronous read latency");
        arvalid = 1'b0;
        araddr = 14'h3ffc;
        @(negedge clk);
        check(rvalid, "RVALID arrives one clock after AR acceptance");
        equal({30'd0, rresp}, {30'd0, response}, "read response");
        data = rdata;
        arvalid = 1'b1;
        for (cycle = 0; cycle < stall_cycles; cycle = cycle + 1) begin
            @(negedge clk);
            check(rvalid && !arready, "read response held under backpressure");
            equal(rdata, data, "stalled RDATA stable");
            equal({30'd0, rresp}, {30'd0, response}, "stalled RRESP stable");
        end
        arvalid = 1'b0;
        rready = 1'b1;
        @(negedge clk);
        check(!rvalid, "RVALID clears after handshake");
        rready = 1'b0;
    end
endtask

task expect_register;
    input [13:0] address;
    input [31:0] expected;
    reg [31:0] actual;
    begin
        axi_read(address, 2'b00, 2, actual);
        equal(actual, expected, "register or RAM read");
    end
endtask

function [31:0] spectral_power;
    input integer bin_number, pattern;
    begin
        spectral_power = 32'd10;
        if (pattern == 0) begin
            case (bin_number)
                0: spectral_power = 32'h1234_5678;
                6: spectral_power = 32'd150;
                9: spectral_power = 32'd900;
                10: spectral_power = 32'd1000;
                11: spectral_power = 32'd800;
                14: spectral_power = 32'd20;
                19: spectral_power = 32'd1900;
                20, 30: spectral_power = 32'd2000;
                21: spectral_power = 32'd1800;
                40: spectral_power = 32'h8000_0005;
                50, 51: spectral_power = 32'd3000;
                599, 601: spectral_power = 32'd499;
                600: spectral_power = 32'd500;
                602: spectral_power = 32'd10000;
                2047: spectral_power = 32'hfedc_ba98;
                default: begin end
            endcase
        end else if (pattern == 1) begin
            // Thirty ascending isolated peaks exercise replacement of a
            // full top-24 list, rather than merely sorting <=24 candidates.
            if (bin_number >= 6 && bin_number <= 122 &&
                (bin_number - 6) % 4 == 0)
                spectral_power = 100 + ((bin_number - 6) / 4) * 10;
            if (bin_number == 0) spectral_power = 32'hffff_ffff;
            if (bin_number == 2047) spectral_power = 32'h8765_4321;
        end else if (pattern == 2) begin
            // All thirty ties must retain the first (smallest-bin) 24.
            if (bin_number >= 6 && bin_number <= 122 &&
                (bin_number - 6) % 4 == 0) spectral_power = 32'd100;
        end else begin
            spectral_power = 32'd0;
        end
    end
endfunction

task stream_frame;
    input integer pattern, gaps, delayed_done, already_running;
    integer bin_number;
    begin
        if (already_running)
            check(fft_busy && freeze && !peaks_busy, "active transform finishes despite setup-phase freeze");
        else
            check(!frame_hold && !freeze && !peaks_busy, "engine may start next transform");
        for (bin_number = 0; bin_number < 2048; bin_number = bin_number + 1) begin
            @(negedge clk);
            fft_busy = 1'b1;
            frame_start = bin_number == 0;
            frame_done = bin_number == 2047 && !delayed_done;
            bin_valid = 1'b1;
            bin_index = bin_number;
            bin_power = spectral_power(bin_number, pattern);
            if (gaps && bin_number % 17 == 16 && bin_number != 2047) begin
                @(negedge clk);
                bin_valid = 1'b0;
                repeat (2) @(negedge clk);
            end
        end
        if (delayed_done) begin
            @(negedge clk);
            bin_valid = 1'b0;
            repeat (7) @(negedge clk);
            check(!peaks_busy && fft_busy, "capture waits for separately timed frame_done");
            frame_done = 1'b1;
        end
        @(negedge clk);
        fft_busy = 1'b0;
        frame_start = 1'b0;
        frame_done = 1'b0;
        bin_valid = 1'b0;
        check(peaks_busy && frame_hold, "frame_done begins post-FFT extraction hold");
    end
endtask

task wait_publish;
    input [31:0] expected_sequence;
    integer cycles;
    begin
        cycles = 0;
        while (peaks_busy && cycles < 10000) begin
            @(negedge clk);
            cycles = cycles + 1;
        end
        check(!peaks_busy, "peak extraction completes within conservative bound");
        expect_register(14'h0008, expected_sequence);
        axi_read(14'h0004, 2'b00, 0, value);
        equal(value & 32'h3, 32'h2, "idle ready snapshot after publish");
    end
endtask

task expect_peak;
    input integer rank;
    input [31:0] bin_number, power;
    begin
        expect_register(14'h0100 + rank * 8, bin_number);
        expect_register(14'h0104 + rank * 8, power);
    end
endtask

// Independent AXI channel monitors catch unstable VALID payloads in every
// scenario, including when capture and software transactions overlap.
reg previous_rstall = 1'b0, previous_bstall = 1'b0;
reg [31:0] previous_rdata;
reg [1:0] previous_rresp, previous_bresp;
always @(posedge clk) begin
    if (!rst) begin
        if (previous_rstall && (!rvalid || rdata !== previous_rdata ||
                                rresp !== previous_rresp))
            $fatal(1, "FAIL: read response changed while stalled");
        if (previous_bstall && (!bvalid || bresp !== previous_bresp))
            $fatal(1, "FAIL: write response changed while stalled");
        previous_rstall <= rvalid && !rready;
        previous_bstall <= bvalid && !bready;
        previous_rdata <= rdata;
        previous_rresp <= rresp;
        previous_bresp <= bresp;
    end else begin
        previous_rstall <= 1'b0;
        previous_bstall <= 1'b0;
    end
end

initial begin
    #2000000;
    $fatal(1, "FAIL: testbench watchdog timed out");
end

initial begin
    repeat (5) @(negedge clk);
    check(freeze && frame_hold && !awready && !arready, "reset holds pipeline and AXI");
    rst = 1'b0;
    expect_register(14'h0000, 32'h4155_4434);
    expect_register(14'h0004, 32'd0);
    expect_register(14'h0008, 32'd0);
    expect_register(14'h000c, 32'd0);
    equal({30'd0, analysis_test_mode}, 32'd0, "reset selects live analysis audio");
    expect_register(14'h0010, 32'd4096);
    expect_register(14'h0014, 32'd48000);
    expect_register(14'h0020, 32'd1048576);
    expect_register(14'h0024, 32'd4095);
    dropped_frames = 32'd7;
    expect_register(14'h0018, 32'd7);
    axi_read(14'h1000, 2'b10, 3, value);
    equal(value, 32'd0, "uninitialized RAM cannot be published");
    axi_read(14'h0100, 2'b10, 0, value);

    // All mode/freeze combinations round-trip; reserved bits remain zero.
    for (index = 0; index < 8; index = index + 1) begin
        axi_write(14'h000c, 32'hffff_fff8 | index, 4'h1, index % 3, 2'b00);
        expect_register(14'h000c, index);
        equal({30'd0, analysis_test_mode}, index >> 1, "analysis mode output follows control bits");
        check(freeze === ((index & 1) != 0), "mode selection preserves independent freeze bit");
    end
    axi_write(14'h000c, 32'd0, 4'h1, 0, 2'b00);
    axi_write(14'h000c, 32'd7, 4'h0, 1, 2'b00);
    check(!freeze, "zero WSTRB leaves freeze clear");
    equal({30'd0, analysis_test_mode}, 32'd0, "zero WSTRB cannot alter analysis mode");
    expect_register(14'h000c, 32'd0);
    axi_write(14'h000c, 32'd7, 4'he, 2, 2'b00);
    check(!freeze, "upper byte strobes cannot alter freeze");
    equal({30'd0, analysis_test_mode}, 32'd0, "upper byte strobes cannot alter analysis mode");
    expect_register(14'h000c, 32'd0);
    axi_write(14'h000c, 32'hffff_ffff, 4'h1, 0, 2'b00);
    check(freeze && frame_hold && !peaks_busy, "freeze asserted while idle");
    expect_register(14'h000c, 32'd7);
    equal({30'd0, analysis_test_mode}, 32'd3, "low byte write selects A minor");
    expect_register(14'h0004, 32'd4);
    axi_write(14'h000c, 32'd0, 4'he, 0, 2'b00);
    check(freeze, "upper byte write cannot release freeze");
    equal({30'd0, analysis_test_mode}, 32'd3, "upper byte write preserves selected mode");
    expect_register(14'h000c, 32'd7);
    axi_write(14'h000c, 32'd0, 4'h1, 2, 2'b00);
    check(!frame_hold, "freeze release enables engine");
    equal({30'd0, analysis_test_mode}, 32'd0, "low byte zero returns analysis to audio");

    fft_busy = 1'b1;
    check(!peaks_busy, "FFT activity precedes spectral capture");
    expect_register(14'h0004, 32'd1);
    axi_write(14'h000c, 32'd1, 4'h1, 0, 2'b00);
    check(freeze && !peaks_busy, "freeze request reaches engine before capture");
    expect_register(14'h0004, 32'd1);
    fft_busy = 1'b0;
    expect_register(14'h0004, 32'd4);
    axi_write(14'h000c, 32'd0, 4'h1, 0, 2'b00);

    // Freeze mid-capture, while the engine continues all remaining bins.
    fork
        stream_frame(0, 1, 0, 0);
        begin
            wait (bin_valid && bin_index == 12'd100);
            axi_write(14'h000c, 32'd1, 4'h1, 1, 2'b00);
            check(freeze && !peaks_busy && fft_busy, "freeze does not abort active FFT");
            expect_register(14'h0004, 32'd1);
            expect_register(14'h0008, 32'd0);
            axi_read(14'h1000, 2'b10, 4, value);
            equal(value, 32'd0, "busy capture rejects incoherent RAM reads");
        end
    join
    expect_register(14'h0004, 32'd1);
    expect_register(14'h0008, 32'd0);
    axi_read(14'h0100, 2'b10, 0, value);
    wait_publish(32'd1);
    expect_register(14'h0004, 32'd6);
    expect_register(14'h001c, 32'd6);
    expect_peak(0, 32'd40, 32'h8000_0005);
    expect_peak(1, 32'd20, 32'd2000);
    expect_peak(2, 32'd30, 32'd2000);
    expect_peak(3, 32'd10, 32'd1000);
    expect_peak(4, 32'd600, 32'd500);
    expect_peak(5, 32'd6, 32'd150);
    expect_peak(6, 32'd0, 32'd0);
    expect_peak(23, 32'd0, 32'd0);

    // Check all 2048 addresses, including both endpoints and high unsigned
    // values, against the input stimulus rather than DUT internals.
    for (index = 0; index < 2048; index = index + 1)
        expect_register(14'h1000 + index * 4, spectral_power(index, 0));
    dropped_frames = 32'd9;
    expect_register(14'h0018, 32'd9);
    repeat (20) @(negedge clk);
    check(freeze && frame_hold, "frozen snapshot remains held");
    expect_register(14'h0008, 32'd1);
    axi_read(14'h3000, 2'b10, 3, value);
    axi_read(14'h2fff, 2'b10, 0, value);
    axi_read(14'h01c0, 2'b10, 0, value);
    axi_write(14'h1000, 32'd0, 4'hf, 0, 2'b10);
    axi_write(14'h000d, 32'd0, 4'hf, 0, 2'b10);
    expect_register(14'h1000, spectral_power(0, 0));

    // Even a misbehaving engine cannot change RAM while frozen.
    @(negedge clk);
    frame_start = 1'b1;
    bin_valid = 1'b1;
    bin_index = 12'd0;
    bin_power = 32'd0;
    frame_done = 1'b1;
    @(negedge clk);
    frame_start = 1'b0;
    bin_valid = 1'b0;
    frame_done = 1'b0;
    expect_register(14'h0004, 32'd14);
    expect_register(14'h0008, 32'd1);
    expect_register(14'h1000, spectral_power(0, 0));
    axi_write(14'h0004, 32'd8, 4'he, 1, 2'b00);
    expect_register(14'h0004, 32'd14);
    axi_write(14'h0004, 32'd8, 4'h1, 2, 2'b00);
    expect_register(14'h0004, 32'd6);
    axi_write(14'h000c, 32'd0, 4'h1, 0, 2'b00);

    stream_frame(1, 0, 1, 0);
    axi_write(14'h000c, 32'd1, 4'h1, 0, 2'b00);
    wait_publish(32'd2);
    expect_register(14'h001c, 32'd24);
    for (index = 0; index < 24; index = index + 1)
        expect_peak(index, 122 - index * 4, 390 - index * 10);
    expect_register(14'h1000, 32'hffff_ffff);
    expect_register(14'h2ffc, 32'h8765_4321);

    axi_write(14'h000c, 32'd0, 4'h1, 0, 2'b00);
    stream_frame(2, 0, 0, 0);
    axi_write(14'h000c, 32'd1, 4'h1, 0, 2'b00);
    wait_publish(32'd3);
    expect_register(14'h001c, 32'd24);
    for (index = 0; index < 24; index = index + 1)
        expect_peak(index, 6 + index * 4, 32'd100);

    axi_write(14'h000c, 32'd0, 4'h1, 0, 2'b00);
    stream_frame(3, 0, 0, 0);
    axi_write(14'h000c, 32'd1, 4'h1, 0, 2'b00);
    wait_publish(32'd4);
    expect_register(14'h001c, 32'd0);
    expect_peak(0, 32'd0, 32'd0);
    expect_register(14'h1000, 32'd0);
    expect_register(14'h2ffc, 32'd0);

    // Incomplete/out-of-order frames cannot increment the sequence or
    // expose a partially overwritten snapshot.
    axi_write(14'h000c, 32'd0, 4'h1, 0, 2'b00);
    @(negedge clk);
    fft_busy = 1'b1;
    frame_start = 1'b1;
    bin_valid = 1'b1;
    bin_index = 12'd0;
    bin_power = 32'd77;
    @(negedge clk);
    frame_start = 1'b0;
    bin_index = 12'd2;
    frame_done = 1'b1;
    @(negedge clk);
    fft_busy = 1'b0;
    frame_done = 1'b0;
    bin_valid = 1'b0;
    expect_register(14'h0004, 32'd8);
    expect_register(14'h0008, 32'd4);
    axi_read(14'h1000, 2'b10, 2, value);
    equal(value, 32'd0, "invalid frame cannot expose partially overwritten RAM");

    // Reset abandons an accepted AW awaiting W and an R response awaiting
    // RREADY; old RAM becomes inaccessible until a new frame is published.
    axi_write(14'h000c, 32'd6, 4'h1, 0, 2'b00);
    equal({30'd0, analysis_test_mode}, 32'd3, "nonzero analysis mode before reset");
    send_aw(14'h000c);
    @(negedge clk);
    araddr = 14'h0000;
    arvalid = 1'b1;
    @(negedge clk);
    arvalid = 1'b0;
    @(negedge clk);
    check(rvalid && !awready, "pending AW and stalled R before reset");
    rst = 1'b1;
    repeat (3) @(negedge clk);
    check(!rvalid && !bvalid && freeze, "reset flushes pending responses");
    equal({30'd0, analysis_test_mode}, 32'd0, "reset clears previously selected test mode");
    rst = 1'b0;
    expect_register(14'h0004, 32'd0);
    expect_register(14'h0008, 32'd0);
    expect_register(14'h000c, 32'd0);
    axi_read(14'h1000, 2'b10, 0, value);
    axi_write(14'h000c, 32'd1, 4'h1, 2, 2'b00);
    expect_register(14'h0004, 32'd4);

    @(negedge clk);
    fft_error = 1'b1;
    expect_register(14'h0004, 32'd12);
    axi_write(14'h0004, 32'd8, 4'h1, 0, 2'b00);
    expect_register(14'h0004, 32'd12);
    @(negedge clk);
    fft_error = 1'b0;
    expect_register(14'h0004, 32'd12);
    axi_write(14'h0004, 32'd8, 4'h1, 0, 2'b00);
    expect_register(14'h0004, 32'd4);

    axi_write(14'h000c, 32'd0, 4'h1, 0, 2'b00);
    @(negedge clk);
    fft_busy = 1'b1;
    axi_write(14'h000c, 32'd1, 4'h1, 0, 2'b00);
    expect_register(14'h0004, 32'd1);
    stream_frame(0, 0, 1, 1);
    wait_publish(32'd1);
    expect_register(14'h0004, 32'd6);
    expect_register(14'h001c, 32'd6);
    expect_register(14'h1000, spectral_power(0, 0));
    expect_register(14'h2ffc, spectral_power(2047, 0));

    $display("PASS: audio_spectrum_axi %0d checks; coherent RAM, local top24, AXI stalls, freeze, FFT error and invalid frames verified.", checks);
    $finish;
end

endmodule
