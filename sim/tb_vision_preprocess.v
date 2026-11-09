`timescale 1ns / 1ps

module tb_vision_preprocess;

parameter integer FRAME_WIDTH = 5;
localparam integer MAX_PIXELS = 2048;
reg clk = 0;
reg rst = 1;
always #5 clk = ~clk;

reg [23:0] s_rgb = 0;
reg s_valid = 0, s_sof = 0, s_eol = 0;
reg [7:0] source_expected_gray = 0;
wire s_ready;
wire [7:0] gray;
wire gray_valid, gray_ready, gray_sof, gray_eol;
wire signed [11:0] gx, gy;
wire [7:0] magnitude;
wire edge_pixel, m_valid, m_sof, m_eol;
reg m_ready = 0;
reg [7:0] threshold = 0;
reg [1:0] ready_mode = 0;
integer cycle = 0;
reg [31:0] noise = 32'h17ace531;

vision_rgb_gray rgb_dut (
    .clk(clk), .rst(rst), .s_rgb(s_rgb), .s_valid(s_valid),
    .s_ready(s_ready), .s_sof(s_sof), .s_eol(s_eol),
    .m_gray(gray), .m_valid(gray_valid), .m_ready(gray_ready),
    .m_sof(gray_sof), .m_eol(gray_eol)
);

vision_sobel3x3 #(.FRAME_WIDTH(FRAME_WIDTH)) sobel_dut (
    .clk(clk), .rst(rst), .s_gray(gray), .s_valid(gray_valid),
    .s_ready(gray_ready), .s_sof(gray_sof), .s_eol(gray_eol),
    .threshold(threshold), .m_gx(gx), .m_gy(gy),
    .m_magnitude(magnitude), .m_edge(edge_pixel),
    .m_valid(m_valid), .m_ready(m_ready), .m_sof(m_sof), .m_eol(m_eol)
);

reg [7:0] expected_gray [0:MAX_PIXELS-1];
reg expected_gray_sof [0:MAX_PIXELS-1];
reg expected_gray_eol [0:MAX_PIXELS-1];
integer expected_gx [0:MAX_PIXELS-1];
integer expected_gy [0:MAX_PIXELS-1];
integer expected_magnitude [0:MAX_PIXELS-1];
reg expected_edge [0:MAX_PIXELS-1];
reg expected_sof [0:MAX_PIXELS-1];
reg expected_eol [0:MAX_PIXELS-1];
integer image [0:MAX_PIXELS-1];
integer sent = 0, gray_received = 0, sobel_sent = 0, received = 0;
integer model_x = 0, model_y = 0;
integer a, b, c, d, f, g, h, i;
integer reference_x, reference_y, reference_l1;
integer gray_stalls = 0, output_stalls = 0, input_stalls = 0;
integer stalled_sofs = 0, stalled_eols = 0;
integer positive_x = 0, negative_x = 0, positive_y = 0, negative_y = 0;
integer clipped_count = 0, equality_count = 0, below_count = 0;
integer zero_threshold_count = 0, maximum_threshold_count = 0;
integer restart_count = 0, bubble_count = 0, consecutive_count = 0;
integer stalled_restart_count = 0, mixed_rgb_count = 0;
integer full_rate_inputs = 0, full_rate_outputs = 0;
reg check_full_rate = 0;
time last_input_time, last_output_time;
reg previous_output = 0;
reg gray_was_stalled = 0, output_was_stalled = 0, source_was_stalled = 0;
reg [9:0] held_gray;
reg [34:0] held_output;
reg [25:0] held_source;

function [7:0] rgb_reference;
    input [23:0] rgb;
    integer value;
    begin
        value = 77 * rgb[23:16] + 150 * rgb[15:8] + 29 * rgb[7:0] + 128;
        rgb_reference = value / 256;
    end
endfunction

// Change threshold even during stalls, proving it is sampled per input pixel.
always @(negedge clk) begin
    cycle = cycle + 1;
    noise = {noise[30:0], noise[31] ^ noise[21] ^ noise[1] ^ noise[0]};
    case (ready_mode)
        0: m_ready = 0;
        1: m_ready = 1;
        default: m_ready = ((cycle % 23) >= 9) && noise[0];
    endcase
    case (cycle % 7)
        0: threshold = 0;
        1: threshold = 127;
        2: threshold = 128;
        3: threshold = 129;
        4: threshold = 255;
        default: threshold = noise[15:8];
    endcase
end

always @(posedge clk) begin
    if (!rst) begin
        if (source_was_stalled &&
            (!s_valid || {s_rgb, s_sof, s_eol} !== held_source))
            $fatal(1, "TB source changed during input stall");
        if (gray_was_stalled &&
            (!gray_valid || {gray, gray_sof, gray_eol} !== held_gray))
            $fatal(1, "Grayscale output/metadata changed during stall");
        if (output_was_stalled &&
            (!m_valid || {gx, gy, magnitude, edge_pixel, m_sof, m_eol} !== held_output))
            $fatal(1, "Sobel output/metadata changed during stall");

        source_was_stalled = s_valid && !s_ready;
        gray_was_stalled = gray_valid && !gray_ready;
        output_was_stalled = m_valid && !m_ready;
        held_source = {s_rgb, s_sof, s_eol};
        held_gray = {gray, gray_sof, gray_eol};
        held_output = {gx, gy, magnitude, edge_pixel, m_sof, m_eol};
        if (source_was_stalled) begin
            input_stalls = input_stalls + 1;
            if (s_sof) stalled_restart_count = stalled_restart_count + 1;
        end
        if (gray_was_stalled) gray_stalls = gray_stalls + 1;
        if (output_was_stalled) begin
            output_stalls = output_stalls + 1;
            if (m_sof) stalled_sofs = stalled_sofs + 1;
            if (m_eol) stalled_eols = stalled_eols + 1;
        end
        if (!s_valid) bubble_count = bubble_count + 1;

        if (s_valid && s_ready) begin
            if (check_full_rate) begin
                if (full_rate_inputs && ($time - last_input_time != 10))
                    $fatal(1, "Unexpected input bubble during full-rate stream");
                last_input_time = $time;
                full_rate_inputs = full_rate_inputs + 1;
            end
            if ((s_rgb[23:16] != s_rgb[15:8]) || (s_rgb[15:8] != s_rgb[7:0]))
                mixed_rgb_count = mixed_rgb_count + 1;
            if (sent >= MAX_PIXELS) $fatal(1, "Reference queue overflow");
            expected_gray[sent] = source_expected_gray;
            expected_gray_sof[sent] = s_sof;
            expected_gray_eol[sent] = s_eol;
            sent = sent + 1;
        end

        if (gray_valid && gray_ready) begin
            if (gray_received >= sent) $fatal(1, "Unexpected grayscale output");
            if ({gray, gray_sof, gray_eol} !==
                {expected_gray[gray_received], expected_gray_sof[gray_received],
                 expected_gray_eol[gray_received]})
                $fatal(1, "Gray mismatch pixel %0d: got %0d expected %0d",
                       gray_received, gray, expected_gray[gray_received]);
            gray_received = gray_received + 1;

            if (gray_sof) begin
                if (model_x != 0) restart_count = restart_count + 1;
                model_x = 0;
                model_y = 0;
            end
            if (gray_eol !== (model_x == FRAME_WIDTH-1))
                $fatal(1, "TB violated raster EOL contract at (%0d,%0d)", model_x, model_y);
            if (model_y * FRAME_WIDTH + model_x >= MAX_PIXELS)
                $fatal(1, "Reference image overflow");
            image[model_y * FRAME_WIDTH + model_x] = gray;
            reference_x = 0;
            reference_y = 0;
            reference_l1 = 0;
            if ((model_x >= 2) && (model_y >= 2)) begin
                a = image[(model_y-2)*FRAME_WIDTH + model_x-2];
                b = image[(model_y-2)*FRAME_WIDTH + model_x-1];
                c = image[(model_y-2)*FRAME_WIDTH + model_x];
                d = image[(model_y-1)*FRAME_WIDTH + model_x-2];
                f = image[(model_y-1)*FRAME_WIDTH + model_x];
                g = image[model_y*FRAME_WIDTH + model_x-2];
                h = image[model_y*FRAME_WIDTH + model_x-1];
                i = image[model_y*FRAME_WIDTH + model_x];
                reference_x = c + 2*f + i - a - 2*d - g;
                reference_y = g + 2*h + i - a - 2*b - c;
                reference_l1 = ((reference_x < 0) ? -reference_x : reference_x) +
                               ((reference_y < 0) ? -reference_y : reference_y);
                if (reference_x > 0) positive_x = positive_x + 1;
                if (reference_x < 0) negative_x = negative_x + 1;
                if (reference_y > 0) positive_y = positive_y + 1;
                if (reference_y < 0) negative_y = negative_y + 1;
                if (reference_l1 > 255) begin
                    reference_l1 = 255;
                    clipped_count = clipped_count + 1;
                end
                if (reference_l1 == threshold) equality_count = equality_count + 1;
                if (reference_l1 < threshold) below_count = below_count + 1;
                if (threshold == 0) zero_threshold_count = zero_threshold_count + 1;
                if (threshold == 255) maximum_threshold_count = maximum_threshold_count + 1;
            end
            expected_gx[sobel_sent] = reference_x;
            expected_gy[sobel_sent] = reference_y;
            expected_magnitude[sobel_sent] = reference_l1;
            expected_edge[sobel_sent] = (model_x >= 2) && (model_y >= 2) &&
                                         (reference_l1 >= threshold);
            expected_sof[sobel_sent] = gray_sof;
            expected_eol[sobel_sent] = gray_eol;
            sobel_sent = sobel_sent + 1;
            if (model_x == FRAME_WIDTH-1) begin
                model_x = 0;
                model_y = model_y + 1;
            end else model_x = model_x + 1;
        end

        if (m_valid && m_ready) begin
            if (check_full_rate) begin
                if (full_rate_outputs && ($time - last_output_time != 10))
                    $fatal(1, "Unexpected output bubble during full-rate stream");
                last_output_time = $time;
                full_rate_outputs = full_rate_outputs + 1;
            end
            if (received >= sobel_sent) $fatal(1, "Unexpected extra Sobel output");
            if (($signed(gx) !== expected_gx[received]) ||
                ($signed(gy) !== expected_gy[received]) ||
                (magnitude !== expected_magnitude[received]) ||
                (edge_pixel !== expected_edge[received]) ||
                (m_sof !== expected_sof[received]) || (m_eol !== expected_eol[received]))
                $fatal(1, "Sobel mismatch pixel %0d: gx=%0d/%0d gy=%0d/%0d L1=%0d/%0d edge=%b/%b sof=%b/%b eol=%b/%b",
                       received, $signed(gx), expected_gx[received],
                       $signed(gy), expected_gy[received], magnitude,
                       expected_magnitude[received], edge_pixel, expected_edge[received],
                       m_sof, expected_sof[received], m_eol, expected_eol[received]);
            received = received + 1;
            if (previous_output) consecutive_count = consecutive_count + 1;
        end
        previous_output = m_valid && m_ready;
    end else begin
        if (s_ready || gray_ready) $fatal(1, "Ready asserted during reset");
        source_was_stalled = 0;
        gray_was_stalled = 0;
        output_was_stalled = 0;
        previous_output = 0;
    end
end

task send_pixel;
    input [23:0] rgb;
    input sof;
    input eol;
    input integer known_gray;
    begin
        @(negedge clk);
        s_rgb = rgb;
        s_sof = sof;
        s_eol = eol;
        s_valid = 1;
        source_expected_gray = (known_gray < 0) ? rgb_reference(rgb) : known_gray;
        @(posedge clk);
        while (!s_ready) @(posedge clk);
    end
endtask

task input_bubble;
    begin
        @(negedge clk);
        s_valid = 0;
        repeat (2) @(posedge clk);
    end
endtask

task send_frame;
    input integer pattern;
    input integer pixels;
    input integer bubbles;
    integer n, x, y, value;
    reg [23:0] rgb;
    begin
        for (n = 0; n < pixels; n = n + 1) begin
            x = n % FRAME_WIDTH;
            y = n / FRAME_WIDTH;
            case (pattern)
                0: value = (x < 2) ? 0 : 255;
                1: value = (FRAME_WIDTH-1-x)*9 + (4-y)*7;
                2: value = y*40;
                3: value = 73;
                default: value = (x*71 + y*37 + x*y*19) % 256;
            endcase
            rgb = {value[7:0], value[7:0], value[7:0]};
            if (pattern == 5) begin
                rgb[23:16] = (x*53 + y*17 + 5) % 256;
                rgb[15:8] = (x*11 + y*79 + 3) % 256;
                rgb[7:0] = (x*97 + y*31 + 7) % 256;
                send_pixel(rgb, n == 0, x == FRAME_WIDTH-1, -1);
            end else
                send_pixel(rgb, n == 0, x == FRAME_WIDTH-1, value);
            if (bubbles && ((n % 4) == 2)) input_bubble;
        end
    end
endtask

task send_rgb_sweep;
    integer n, pixels;
    reg [23:0] rgb;
    begin
        pixels = ((768 + FRAME_WIDTH-1) / FRAME_WIDTH) * FRAME_WIDTH;
        for (n = 0; n < pixels; n = n + 1) begin
            if (n < 256) rgb = {n[7:0], 8'd0, 8'd0};
            else if (n < 512) rgb = {8'd0, n[7:0], 8'd0};
            else if (n < 768) rgb = {8'd0, 8'd0, n[7:0]};
            else rgb = 24'hffffff;
            send_pixel(rgb, n == 0, n%FRAME_WIDTH == FRAME_WIDTH-1, -1);
            if (n%17 == 8) input_bubble;
        end
    end
endtask

task drain;
    integer watchdog;
    begin
        @(negedge clk);
        s_valid = 0;
        ready_mode = 1;
        watchdog = 0;
        while (received != sent) begin
            @(negedge clk);
            watchdog = watchdog + 1;
            if (watchdog > 1000) $fatal(1, "Drain timed out: in=%0d out=%0d", sent, received);
        end
        repeat (6) @(negedge clk);
        if (m_valid || gray_valid || (gray_received != sent) || (sobel_sent != sent))
            $fatal(1, "Pipeline did not drain exactly one output per input");
    end
endtask

integer n;
integer full_rate_start, full_rate_cycles, full_rate_pixels;
initial begin
    repeat (4) @(negedge clk);
    rst = 0;
    // Keep the first SOF blocked long enough to fill both elastic stages.
    fork
        begin
            repeat (18) @(negedge clk);
            ready_mode = 2;
        end
        begin
            for (n = 0; n < 2*FRAME_WIDTH; n = n + 1) begin
                case (n)
                    0: send_pixel(24'h000000, 1, n%FRAME_WIDTH == FRAME_WIDTH-1, 0);
                    1: send_pixel(24'hff0000, 0, n%FRAME_WIDTH == FRAME_WIDTH-1, 77);
                    2: send_pixel(24'h00ff00, 0, n%FRAME_WIDTH == FRAME_WIDTH-1, 149);
                    3: send_pixel(24'h0000ff, 0, n%FRAME_WIDTH == FRAME_WIDTH-1, 29);
                    4: send_pixel(24'hffffff, 0, n%FRAME_WIDTH == FRAME_WIDTH-1, 255);
                    default: send_pixel(24'h020000, 0, n%FRAME_WIDTH == FRAME_WIDTH-1, 1);
                endcase
            end
        end
    join
    // Back-to-back 5-row frames exercise both banks, both gradient signs,
    // non-clipped magnitudes, threshold equality, and frame-history masking.
    send_frame(0, 5*FRAME_WIDTH, 0);
    send_frame(1, 5*FRAME_WIDTH, 1);
    send_frame(2, 5*FRAME_WIDTH, 0);
    send_frame(3, 5*FRAME_WIDTH, 1);
    send_frame(4, 5*FRAME_WIDTH, 1);
    // Restart after row 2, column 1, before queued pixels have drained.
    send_frame(4, 2*FRAME_WIDTH+2, 0);
    send_frame(1, 5*FRAME_WIDTH, 0);
    drain;

    ready_mode = 2;
    send_frame(5, 5*FRAME_WIDTH, 1);
    send_rgb_sweep;
    drain;

    // Interrupt a populated frame with all three pipeline slots occupied.
    // The restart SOF stays unaccepted while old output and RAM reads hold.
    send_frame(4, 2*FRAME_WIDTH+2, 0);
    ready_mode = 0;
    fork
        begin
            repeat (12) @(posedge clk);
            ready_mode = 1;
        end
        begin
            send_frame(1, 5*FRAME_WIDTH, 0);
        end
    join
    drain;

    // Unstalled operation must accept and emit one pixel each clock.
    full_rate_pixels = 5*FRAME_WIDTH;
    check_full_rate = 1;
    full_rate_start = $time;
    send_frame(4, full_rate_pixels, 0);
    full_rate_cycles = ($time - full_rate_start) / 10;
    if (full_rate_cycles != full_rate_pixels)
        $fatal(1, "Throughput mismatch: %0d pixels in %0d clocks", full_rate_pixels, full_rate_cycles);
    drain;
    check_full_rate = 0;
    if ((full_rate_inputs != full_rate_pixels) || (full_rate_outputs != full_rate_pixels))
        $fatal(1, "Full-rate stream lost or duplicated pixels");

    // Check reset after populated RAM, while leaving contents physically intact.
    @(negedge clk);
    rst = 1;
    repeat (3) @(negedge clk);
    rst = 0;
    send_frame(3, 5*FRAME_WIDTH, 0);
    drain;

    if (!input_stalls || !gray_stalls || !output_stalls || !stalled_sofs || !stalled_eols ||
        !positive_x || !negative_x || !positive_y || !negative_y || !clipped_count ||
        !equality_count || !below_count || !zero_threshold_count || !maximum_threshold_count ||
        !restart_count || !stalled_restart_count || !mixed_rgb_count ||
        !bubble_count || !consecutive_count)
        $fatal(1, "Coverage missing: input/gray/out stalls=%0d/%0d/%0d sof/eol=%0d/%0d signs=%0d/%0d/%0d/%0d clip/eq/below=%0d/%0d/%0d thr0/255=%0d/%0d restart=%0d",
               input_stalls, gray_stalls, output_stalls, stalled_sofs, stalled_eols,
               positive_x, negative_x, positive_y, negative_y, clipped_count,
               equality_count, below_count, zero_threshold_count, maximum_threshold_count, restart_count);
    $display("VISION_PREPROCESS_PASS width=%0d inputs=%0d gray=%0d sobel=%0d stalls(input/gray/out)=%0d/%0d/%0d equality=%0d restart=%0d stalled_sof=%0d mixed_rgb=%0d throughput=1 pixel/clock",
             FRAME_WIDTH, sent, gray_received, received, input_stalls, gray_stalls,
             output_stalls, equality_count, restart_count, stalled_restart_count, mixed_rgb_count);
    $finish;
end

initial begin
    #1000000;
    $fatal(1, "TB watchdog expired");
end

endmodule

module tb_vision_preprocess_min3;
    tb_vision_preprocess #(.FRAME_WIDTH(3)) test ();
endmodule
