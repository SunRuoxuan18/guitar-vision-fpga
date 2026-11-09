`timescale 1ns / 1ps

module tb_vision_gray_exact;

localparam integer RGB_COUNT = 16777216;
localparam integer QUEUE_DEPTH = 8;
localparam integer STRESS_COUNT = 4096;
reg clk = 0;
always #5 clk = ~clk;
reg rst = 1;
reg [23:0] s_rgb = 0;
reg s_valid = 0, s_sof = 0, s_eol = 0;
wire s_ready;
wire [7:0] m_gray;
wire m_valid, m_sof, m_eol;
reg m_ready = 0;
reg [1:0] ready_mode = 0;
integer sink_cycle = 0;

vision_rgb_gray dut (
    .clk(clk), .rst(rst), .s_rgb(s_rgb), .s_valid(s_valid),
    .s_ready(s_ready), .s_sof(s_sof), .s_eol(s_eol),
    .m_gray(m_gray), .m_valid(m_valid), .m_ready(m_ready),
    .m_sof(m_sof), .m_eol(m_eol)
);

// Only accepted inputs occupy this ring; no DUT internals or frame-sized arrays.
reg [23:0] expected_rgb [0:QUEUE_DEPTH-1];
reg [1:0] expected_flags [0:QUEUE_DEPTH-1];
integer accepted_cycle [0:QUEUE_DEPTH-1];
integer read_index = 0, write_index = 0, queued = 0;
integer cycle = 0, sent = 0, received = 0, discarded = 0;
integer peak_queued = 0, product_resets = 0, holding_resets = 0;
integer input_stalls = 0, output_stalls = 0, bubbles = 0;
integer stalled_sofs = 0, stalled_eols = 0;
integer sweep_inputs = 0, sweep_outputs = 0;
integer last_sweep_input = 0, last_sweep_output = 0;
reg [3:0] flags_seen = 0;
reg in_reset = 1, source_stalled = 0, output_stalled = 0;
reg [25:0] held_source = 0;
reg [9:0] held_output = 0;
reg sweep = 0, strict_latency = 0;
reg [7:0] reference_gray;

function [7:0] rgb_reference;
    input [23:0] rgb;
    integer red, green, blue, value;
    begin
        red = rgb[23:16];
        green = rgb[15:8];
        blue = rgb[7:0];
        value = (77 * red + 150 * green + 29 * blue + 128) / 256;
        rgb_reference = value;
    end
endfunction

always @(negedge clk) begin
    case (ready_mode)
        0: m_ready = 0;
        1: m_ready = 1;
        default: begin
            sink_cycle = sink_cycle + 1;
            // Bounded bursts of backpressure plus isolated ready bubbles.
            m_ready = (sink_cycle % 29 >= 9) && (sink_cycle % 7 != 0);
        end
    endcase
end

always @(posedge clk) begin
    cycle = cycle + 1;
    if (rst) begin
        if (!in_reset && queued != 0) begin
            discarded = discarded + queued;
            if (m_valid === 1'b0) product_resets = product_resets + 1;
            if (m_valid === 1'b1 && m_ready === 1'b0 && queued == 2)
                holding_resets = holding_resets + 1;
        end
        in_reset = 1;
        read_index = 0;
        write_index = 0;
        queued = 0;
        source_stalled = 0;
        output_stalled = 0;
        if (s_ready !== 1'b0) $fatal(1, "Input ready asserted during reset");
        // Reset is synchronous: inspect cleared outputs after the active edge.
        #1;
        if ({m_valid, m_gray, m_sof, m_eol} !== 11'd0 || s_ready !== 1'b0)
            $fatal(1, "Reset did not clear output/metadata and inhibit input");
    end else begin
        in_reset = 0;
        if ((s_ready !== 1'b0 && s_ready !== 1'b1) ||
            (m_valid !== 1'b0 && m_valid !== 1'b1))
            $fatal(1, "Unknown ready/valid at cycle %0d", cycle);
        if (source_stalled &&
            (s_valid !== 1'b1 || {s_rgb, s_sof, s_eol} !== held_source))
            $fatal(1, "TB source changed while backpressured");
        if (output_stalled &&
            (m_valid !== 1'b1 || {m_gray, m_sof, m_eol} !== held_output))
            $fatal(1, "Output data/metadata changed while backpressured");
        source_stalled = s_valid && !s_ready;
        output_stalled = m_valid && !m_ready;
        held_source = {s_rgb, s_sof, s_eol};
        held_output = {m_gray, m_sof, m_eol};
        if (source_stalled) input_stalls = input_stalls + 1;
        if (output_stalled) begin
            output_stalls = output_stalls + 1;
            if (m_sof) stalled_sofs = stalled_sofs + 1;
            if (m_eol) stalled_eols = stalled_eols + 1;
        end
        if (!s_valid) bubbles = bubbles + 1;

        if (m_valid && queued == 0)
            $fatal(1, "Unexpected output with empty reference queue");
        // Compare only transferred outputs, before recycling their ring slots.
        if (m_valid && m_ready) begin
            reference_gray = rgb_reference(expected_rgb[read_index]);
            if ({m_gray, m_sof, m_eol} !==
                {reference_gray, expected_flags[read_index]})
                $fatal(1, "Output %0d RGB=%06h: gray/flags=%02h/%b%b expected=%02h/%b",
                       received, expected_rgb[read_index], m_gray, m_sof, m_eol,
                       reference_gray, expected_flags[read_index]);
            if (strict_latency && cycle - accepted_cycle[read_index] != 2)
                $fatal(1, "Expected two-cycle latency, got %0d cycles",
                       cycle - accepted_cycle[read_index]);
            flags_seen[{m_sof, m_eol}] = 1'b1;
            read_index = (read_index + 1) % QUEUE_DEPTH;
            queued = queued - 1;
            received = received + 1;
            if (sweep) begin
                if (sweep_outputs != 0 && cycle != last_sweep_output + 1)
                    $fatal(1, "Output bubble during exhaustive full-rate sweep");
                last_sweep_output = cycle;
                sweep_outputs = sweep_outputs + 1;
            end
        end
        if (sweep && s_ready !== 1'b1)
            $fatal(1, "Input backpressure during exhaustive full-rate sweep");
        if (s_valid && s_ready) begin
            if (queued >= QUEUE_DEPTH) $fatal(1, "Reference ring overflow");
            expected_rgb[write_index] = s_rgb;
            expected_flags[write_index] = {s_sof, s_eol};
            accepted_cycle[write_index] = cycle;
            write_index = (write_index + 1) % QUEUE_DEPTH;
            queued = queued + 1;
            sent = sent + 1;
            if (queued > peak_queued) peak_queued = queued;
            if (queued > 2) $fatal(1, "DUT accepted more than two pending pixels");
            if (sweep) begin
                if (sweep_inputs >= RGB_COUNT || s_rgb !== sweep_inputs[23:0])
                    $fatal(1, "Exhaustive sweep skipped/repeated RGB %0d", sweep_inputs);
                if (sweep_inputs != 0 && cycle != last_sweep_input + 1)
                    $fatal(1, "Input bubble during exhaustive full-rate sweep");
                last_sweep_input = cycle;
                sweep_inputs = sweep_inputs + 1;
            end
        end
    end
end

task reset_dut;
    begin
        @(negedge clk);
        rst = 1;
        s_valid = 0;
        s_sof = 0;
        s_eol = 0;
        repeat (3) begin @(posedge clk); #1; end
        @(negedge clk);
        rst = 0;
    end
endtask

task send_pixel;
    input [23:0] rgb;
    input sof, eol;
    integer timeout;
    reg accepted;
    begin
        @(negedge clk);
        s_rgb = rgb;
        s_sof = sof;
        s_eol = eol;
        s_valid = 1;
        timeout = 0;
        accepted = 0;
        while (!accepted) begin
            @(posedge clk);
            accepted = (s_ready === 1'b1);
            #1;
            timeout = timeout + 1;
            if (timeout > 64) $fatal(1, "Input handshake timed out");
        end
    end
endtask

task idle_cycles;
    input integer count;
    integer n;
    begin
        for (n = 0; n < count; n = n + 1) begin
            @(negedge clk);
            s_valid = 0;
            // Invalid-cycle garbage must never enter the pipeline.
            s_rgb = ~s_rgb;
            s_sof = ~s_sof;
            s_eol = ~s_eol;
            @(posedge clk); #1;
        end
    end
endtask

task drain;
    integer timeout;
    begin
        ready_mode = 1;
        @(negedge clk);
        s_valid = 0;
        s_sof = 0;
        s_eol = 0;
        timeout = 0;
        while (queued != 0 || m_valid !== 1'b0) begin
            @(posedge clk); #1;
            timeout = timeout + 1;
            if (timeout > 16) $fatal(1, "Drain timed out: queued=%0d", queued);
        end
        // Catch stale or duplicated outputs, including immediately after reset.
        idle_cycles(4);
    end
endtask

integer pixel;
reg [31:0] noise = 32'h98b137d5;
initial begin
    reset_dut;
    ready_mode = 1;
    drain;

    // Reset with just the new product stage occupied, before output is valid.
    ready_mode = 0;
    send_pixel(24'hff0102, 1, 0);
    if (queued != 1 || m_valid !== 1'b0)
        $fatal(1, "First pixel must occupy product stage without early output");
    reset_dut;
    drain;

    // Fill both stages and reset while output and a third source pixel hold.
    ready_mode = 0;
    send_pixel(24'hfe0180, 1, 1);
    send_pixel(24'h02ff7f, 0, 1);
    if (queued != 2 || m_valid !== 1'b1)
        $fatal(1, "Two-stage elastic pipeline did not fill");
    @(negedge clk);
    s_rgb = 24'h1700ff;
    s_sof = 1;
    s_eol = 0;
    s_valid = 1;
    repeat (8) begin
        @(posedge clk);
        if (s_ready !== 1'b0) $fatal(1, "Full pipeline failed to backpressure");
        #1;
    end
    reset_dut;
    drain;

    // Release a held SOF+EOL and prove both stored pixels survive backpressure.
    ready_mode = 0;
    send_pixel(24'h010002, 1, 1);
    send_pixel(24'hffffff, 0, 0);
    idle_cycles(6);
    if (queued != 2 || s_ready !== 1'b0 || m_valid !== 1'b1)
        $fatal(1, "Elastic contents were lost while output held");
    drain;

    strict_latency = 1;
    send_pixel(24'h000000, 0, 0);
    drain;
    strict_latency = 0;

    // Reproducible stalls/bubbles; each input is held until its handshake.
    ready_mode = 2;
    for (pixel = 0; pixel < STRESS_COUNT; pixel = pixel + 1) begin
        noise = {noise[30:0], noise[31] ^ noise[21] ^ noise[1] ^ noise[0]};
        idle_cycles(noise[2:1]);
        send_pixel(noise[31:8], pixel[0], pixel[1]);
    end
    drain;

    if (sent != STRESS_COUNT + 6 || received != STRESS_COUNT + 3 ||
        discarded != 3 || sent != received + discarded || queued != 0 ||
        m_valid !== 1'b0 || peak_queued != 2 || product_resets != 1 ||
        holding_resets != 1 || input_stalls == 0 || output_stalls == 0 ||
        bubbles == 0 || stalled_sofs == 0 || stalled_eols == 0 ||
        flags_seen !== 4'b1111)
        $fatal(1, "Directed count/coverage failure: sent/received/discarded=%0d/%0d/%0d queued=%0d resets=%0d/%0d stalls=%0d/%0d flags=%b",
               sent, received, discarded, queued, product_resets, holding_resets,
               input_stalls, output_stalls, flags_seen);
    $display("VISION_GRAY_EXACT_DIRECTED_PASS inputs=%0d outputs=%0d discarded=%0d resets_product/holding=%0d/%0d stalls_input/output=%0d/%0d",
             sent, received, discarded, product_resets, holding_resets,
             input_stalls, output_stalls);
    $display("VISION_GRAY_EXACT_SWEEP_BEGIN rgb888=16777216 clock_period_ns=10");
    strict_latency = 1;
    sweep = 1;
    for (pixel = 0; pixel < RGB_COUNT; pixel = pixel + 1)
        send_pixel(pixel[23:0], pixel[0], pixel[1]);
    drain;
    sweep = 0;
    strict_latency = 0;

    if (sweep_inputs != RGB_COUNT || sweep_outputs != RGB_COUNT ||
        sent != received + discarded || queued != 0 || peak_queued != 2 ||
        product_resets != 1 || holding_resets != 1 || discarded != 3 ||
        input_stalls == 0 || output_stalls == 0 || bubbles == 0 ||
        stalled_sofs == 0 || stalled_eols == 0 || flags_seen !== 4'b1111)
        $fatal(1, "Coverage/accounting failure: sweep=%0d/%0d sent/received/discarded=%0d/%0d/%0d peak=%0d resets=%0d/%0d stalls=%0d/%0d bubbles=%0d flags=%b",
               sweep_inputs, sweep_outputs, sent, received, discarded, peak_queued,
               product_resets, holding_resets, input_stalls, output_stalls,
               bubbles, flags_seen);
    $display("VISION_GRAY_EXACT_COVERAGE sent=%0d received=%0d discarded=%0d peak_queued=%0d resets_product/holding=%0d/%0d stalls_input/output=%0d/%0d bubbles=%0d stalled_sof/eol=%0d/%0d flags=%b",
             sent, received, discarded, peak_queued, product_resets, holding_resets,
             input_stalls, output_stalls, bubbles, stalled_sofs, stalled_eols, flags_seen);
    $display("VISION_GRAY_EXACT_PASS rgb888=16777216 latency=2 throughput=1_pixel_per_clock");
    $finish;
end

initial begin
    // The exhaustive sweep alone needs 167.77216 ms at 100 MHz.
    #250000000;
    $fatal(1, "Grayscale watchdog expired after 250 ms: sweep=%0d/%0d queued=%0d",
           sweep_inputs, sweep_outputs, queued);
end

endmodule
