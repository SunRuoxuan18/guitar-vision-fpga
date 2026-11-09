`timescale 1ns / 1ps

module tb_vision_dma_stream;

parameter integer FRAME_WIDTH = 7;
parameter integer FRAME_HEIGHT = 5;
localparam integer FRAME_PIXELS = FRAME_WIDTH * FRAME_HEIGHT;
localparam integer QUEUE_DEPTH = 16;
localparam [7:0] THRESHOLD = 8'd64;
reg clk = 0;
always #5 clk = ~clk;
reg rstn = 0;
reg [31:0] s_axis_tdata = 0;
reg [3:0] s_axis_tkeep = 4'hf;
reg s_axis_tlast = 0, s_axis_tvalid = 0;
wire s_axis_tready;
wire [31:0] m_axis_tdata;
wire [3:0] m_axis_tkeep;
wire m_axis_tlast, m_axis_tvalid;
reg m_axis_tready = 0;
reg [1:0] ready_mode = 0;
reg [31:0] sink_noise = 32'h17ace531;
reg [31:0] source_noise = 32'h98b137d5;
integer cycle = 0;

vision_dma_stream #(
    .FRAME_WIDTH(FRAME_WIDTH), .FRAME_HEIGHT(FRAME_HEIGHT)
) dut (
    .aclk(clk), .aresetn(rstn), .threshold(THRESHOLD),
    .s_axis_tdata(s_axis_tdata), .s_axis_tkeep(s_axis_tkeep),
    .s_axis_tlast(s_axis_tlast), .s_axis_tvalid(s_axis_tvalid),
    .s_axis_tready(s_axis_tready), .m_axis_tdata(m_axis_tdata),
    .m_axis_tkeep(m_axis_tkeep), .m_axis_tlast(m_axis_tlast),
    .m_axis_tvalid(m_axis_tvalid), .m_axis_tready(m_axis_tready)
);

reg [7:0] image [0:FRAME_PIXELS-1];
reg [31:0] expected_data [0:QUEUE_DEPTH-1];
reg expected_last [0:QUEUE_DEPTH-1];
integer sent = 0, received = 0, gray_received = 0;
integer input_lasts = 0, output_lasts = 0;
integer total_sent = 0, total_received = 0, discarded = 0;
integer total_input_lasts = 0, total_output_lasts = 0;
integer input_stalls = 0, gray_stalls = 0, output_stalls = 0;
integer stalled_lasts = 0, bubbles = 0, reset_queued = 0;
integer line_ends = 0, clipped = 0, equal_threshold = 0, below_threshold = 0;
integer positive_gx = 0, negative_gx = 0, positive_gy = 0, negative_gy = 0;
integer peak_queued = 0;
integer pixel, x, y, gray_value, gx, gy, magnitude;
reg edge_value;
reg in_reset = 1;
reg source_stalled = 0, gray_stalled = 0, output_stalled = 0;
reg [36:0] held_source, held_output;
reg [9:0] held_gray;
reg check_full_rate = 0;
integer full_rate_inputs = 0, full_rate_outputs = 0;
time last_input_time, last_output_time;

function [31:0] pixel_word;
    input integer pattern;
    input integer index;
    integer px, py, value;
    reg [7:0] red, green, blue;
    begin
        px = index % FRAME_WIDTH;
        py = index / FRAME_WIDTH;
        case (pattern)
            0: begin
                case (index % 8)
                    0: pixel_word = 32'h00000000;
                    1: pixel_word = 32'h00ff0000;
                    2: pixel_word = 32'h0000ff00;
                    3: pixel_word = 32'h000000ff;
                    4: pixel_word = 32'h00ffffff;
                    5: pixel_word = 32'h00020000;
                    6: pixel_word = 32'h00000100;
                    default: pixel_word = 32'h00000002;
                endcase
            end
            1, 2, 3, 4: begin
                case (pattern)
                    1: value = (px*3 + py*5) % 256;
                    2: value = 255 - ((px*3 + py*5) % 256);
                    3: value = 73;
                    default: value = ((px/2 + py/3) % 2) ? 255 : 0;
                endcase
                pixel_word = {8'd0, value[7:0], value[7:0], value[7:0]};
            end
            default: begin
                red = px*53 + py*17 + 5;
                green = px*11 + py*79 + 3;
                blue = px*97 + py*31 + 7;
                pixel_word = {8'd0, red, green, blue};
            end
        endcase
    end
endfunction

always @(negedge clk) begin
    cycle = cycle + 1;
    sink_noise = {sink_noise[30:0],
                  sink_noise[31] ^ sink_noise[21] ^ sink_noise[1] ^ sink_noise[0]};
    case (ready_mode)
        0: m_axis_tready = 0;
        1: m_axis_tready = 1;
        default: m_axis_tready = (cycle % 37 >= 11) && (sink_noise[0] || sink_noise[3]);
    endcase
end

// The reference is computed from accepted RGB pixels in a full image, using
// integer convolution independently of the DUT's two line banks and pipeline.
always @(posedge clk) begin
    if (!rstn) begin
        if (!in_reset && (sent != received)) begin
            discarded = discarded + sent - received;
            reset_queued = reset_queued + 1;
        end
        in_reset = 1;
        sent = 0;
        received = 0;
        gray_received = 0;
        input_lasts = 0;
        output_lasts = 0;
        source_stalled = 0;
        gray_stalled = 0;
        output_stalled = 0;
        if (s_axis_tready || m_axis_tvalid || m_axis_tlast)
            $fatal(1, "DMA handshake active during reset");
        #1;
        if (m_axis_tdata !== 32'd0 || dut.sobel_stage.m_gray !== 8'd0)
            $fatal(1, "Reset did not clear packed output/grayscale register");
    end else begin
        in_reset = 0;
        if (source_stalled && (!s_axis_tvalid ||
            {s_axis_tdata, s_axis_tkeep, s_axis_tlast} !== held_source))
            $fatal(1, "TB source violated AXI stability");
        if (gray_stalled && (!dut.gray_valid ||
            {dut.gray, dut.gray_sof, dut.gray_eol} !== held_gray))
            $fatal(1, "Grayscale stage changed while stalled");
        if (output_stalled && (!m_axis_tvalid ||
            {m_axis_tdata, m_axis_tkeep, m_axis_tlast} !== held_output))
            $fatal(1, "DMA packed output/TLAST/TKEEP changed while stalled");
        if (m_axis_tkeep !== 4'hf)
            $fatal(1, "Output TKEEP must always be 0xf");

        source_stalled = s_axis_tvalid && !s_axis_tready;
        gray_stalled = dut.gray_valid && !dut.gray_ready;
        output_stalled = m_axis_tvalid && !m_axis_tready;
        held_source = {s_axis_tdata, s_axis_tkeep, s_axis_tlast};
        held_gray = {dut.gray, dut.gray_sof, dut.gray_eol};
        held_output = {m_axis_tdata, m_axis_tkeep, m_axis_tlast};
        if (source_stalled) input_stalls = input_stalls + 1;
        if (gray_stalled) gray_stalls = gray_stalls + 1;
        if (output_stalled) begin
            output_stalls = output_stalls + 1;
            if (m_axis_tlast) stalled_lasts = stalled_lasts + 1;
        end
        if (!s_axis_tvalid) bubbles = bubbles + 1;

        if (s_axis_tvalid && s_axis_tready) begin
            if (sent - received >= QUEUE_DEPTH)
                $fatal(1, "Reference queue overflow");
            pixel = sent % FRAME_PIXELS;
            x = pixel % FRAME_WIDTH;
            y = pixel / FRAME_WIDTH;
            if (s_axis_tkeep !== 4'hf || s_axis_tlast !== (pixel == FRAME_PIXELS-1))
                $fatal(1, "TB input violated fixed-frame TLAST/TKEEP contract");
            gray_value = (77*s_axis_tdata[23:16] + 150*s_axis_tdata[15:8] +
                          29*s_axis_tdata[7:0] + 128) / 256;
            image[pixel] = gray_value;
            gx = 0;
            gy = 0;
            magnitude = 0;
            edge_value = 0;
            if ((x >= 2) && (y >= 2)) begin
                gx = image[(y-2)*FRAME_WIDTH+x] + 2*image[(y-1)*FRAME_WIDTH+x] +
                     image[y*FRAME_WIDTH+x] - image[(y-2)*FRAME_WIDTH+x-2] -
                     2*image[(y-1)*FRAME_WIDTH+x-2] - image[y*FRAME_WIDTH+x-2];
                gy = image[y*FRAME_WIDTH+x-2] + 2*image[y*FRAME_WIDTH+x-1] +
                     image[y*FRAME_WIDTH+x] - image[(y-2)*FRAME_WIDTH+x-2] -
                     2*image[(y-2)*FRAME_WIDTH+x-1] - image[(y-2)*FRAME_WIDTH+x];
                magnitude = ((gx < 0) ? -gx : gx) + ((gy < 0) ? -gy : gy);
                if (magnitude > 255) begin
                    magnitude = 255;
                    clipped = clipped + 1;
                end
                edge_value = (magnitude >= THRESHOLD);
                if (magnitude == THRESHOLD) equal_threshold = equal_threshold + 1;
                if (magnitude < THRESHOLD) below_threshold = below_threshold + 1;
                if (gx > 0) positive_gx = positive_gx + 1;
                if (gx < 0) negative_gx = negative_gx + 1;
                if (gy > 0) positive_gy = positive_gy + 1;
                if (gy < 0) negative_gy = negative_gy + 1;
            end
            expected_data[sent % QUEUE_DEPTH] =
                {8'd0, (edge_value ? 8'hff : 8'd0), magnitude[7:0], gray_value[7:0]};
            expected_last[sent % QUEUE_DEPTH] = (pixel == FRAME_PIXELS-1);
            sent = sent + 1;
            total_sent = total_sent + 1;
            if (s_axis_tlast) begin
                input_lasts = input_lasts + 1;
                total_input_lasts = total_input_lasts + 1;
            end
            if (check_full_rate) begin
                if (full_rate_inputs && ($time - last_input_time != 10))
                    $fatal(1, "Input throughput below one pixel/clock");
                last_input_time = $time;
                full_rate_inputs = full_rate_inputs + 1;
            end
        end

        if (dut.gray_valid && dut.gray_ready) begin
            if (dut.gray_sof !== (gray_received % FRAME_PIXELS == 0) ||
                dut.gray_eol !== (gray_received % FRAME_WIDTH == FRAME_WIDTH-1))
                $fatal(1, "SOF/EOL not aligned to accepted fixed-size raster");
            gray_received = gray_received + 1;
        end

        if (m_axis_tvalid && m_axis_tready) begin
            if (received >= sent) $fatal(1, "Unexpected or duplicated output");
            if (m_axis_tdata !== expected_data[received % QUEUE_DEPTH] ||
                m_axis_tlast !== expected_last[received % QUEUE_DEPTH])
                $fatal(1, "DMA mismatch width=%0d height=%0d pixel=%0d got=%08x last=%b expected=%08x last=%b",
                       FRAME_WIDTH, FRAME_HEIGHT, received, m_axis_tdata, m_axis_tlast,
                       expected_data[received % QUEUE_DEPTH], expected_last[received % QUEUE_DEPTH]);
            if (received % FRAME_WIDTH == FRAME_WIDTH-1 && !m_axis_tlast)
                line_ends = line_ends + 1;
            received = received + 1;
            total_received = total_received + 1;
            if (m_axis_tlast) begin
                output_lasts = output_lasts + 1;
                total_output_lasts = total_output_lasts + 1;
            end
            if (output_lasts != received / FRAME_PIXELS)
                $fatal(1, "Output TLAST count differs from complete frames");
            if (check_full_rate) begin
                if (full_rate_outputs && ($time - last_output_time != 10))
                    $fatal(1, "Output throughput below one pixel/clock");
                last_output_time = $time;
                full_rate_outputs = full_rate_outputs + 1;
            end
        end
        if (sent - received > peak_queued) peak_queued = sent - received;
    end
end

task send_pixels;
    input integer pattern;
    input integer count;
    input integer random_bubbles;
    integer n, delay_cycles;
    begin
        for (n = 0; n < count; n = n + 1) begin
            source_noise = {source_noise[30:0],
                            source_noise[31] ^ source_noise[21] ^ source_noise[1] ^ source_noise[0]};
            if (random_bubbles && (source_noise[2:0] < 3)) begin
                @(negedge clk);
                s_axis_tvalid = 0;
                delay_cycles = 1 + source_noise[5:4];
                repeat (delay_cycles) @(posedge clk);
            end
            @(negedge clk);
            s_axis_tdata = pixel_word(pattern, n);
            s_axis_tkeep = 4'hf;
            s_axis_tlast = (n == FRAME_PIXELS-1);
            s_axis_tvalid = 1;
            @(posedge clk);
            while (!s_axis_tready) @(posedge clk);
        end
    end
endtask

task stop_source;
    begin
        @(negedge clk);
        s_axis_tvalid = 0;
    end
endtask

task drain;
    integer watchdog;
    begin
        stop_source;
        ready_mode = 1;
        watchdog = 0;
        while (received != sent) begin
            @(negedge clk);
            watchdog = watchdog + 1;
            if (watchdog > 100) $fatal(1, "Drain timed out: in=%0d out=%0d", sent, received);
        end
        repeat (6) @(negedge clk);
        if (m_axis_tvalid || dut.gray_valid || gray_received != sent ||
            input_lasts != output_lasts || sent % FRAME_PIXELS != 0)
            $fatal(1, "Pipeline did not drain complete frames exactly");
    end
endtask

task pulse_reset;
    begin
        @(negedge clk);
        rstn = 0;
        s_axis_tvalid = 1;
        repeat (4) @(negedge clk);
        s_axis_tvalid = 0;
        rstn = 1;
        ready_mode = 2;
    end
endtask

task hold_final_output;
    begin
        wait (m_axis_tvalid && m_axis_tlast);
        @(negedge clk);
        ready_mode = 0;
        #1;
        m_axis_tready = 0;
        repeat (12) @(negedge clk);
    end
endtask

integer pattern;
time full_rate_start;
initial begin
    repeat (4) @(negedge clk);
    rstn = 1;
    // Fill every elastic slot while the first output is blocked.
    fork
        begin
            repeat (24) @(negedge clk);
            ready_mode = 2;
        end
        begin
            send_pixels(0, FRAME_PIXELS, 1);
            stop_source;
        end
    join
    for (pattern = 1; pattern <= 5; pattern = pattern + 1)
        send_pixels(pattern, FRAME_PIXELS, 1);
    drain;

    fork
        begin
            send_pixels(4, FRAME_PIXELS, 0);
            stop_source;
        end
        begin
            hold_final_output;
            ready_mode = 1;
        end
    join
    drain;

    // Reset with output, gradients, pending RAM data and both grayscale slots
    // queued. The next frame ignores the retained line RAM.
    send_pixels(5, FRAME_PIXELS/2 + 1, 0);
    stop_source;
    ready_mode = 0;
    repeat (6) @(negedge clk);
    pulse_reset;
    send_pixels(3, FRAME_PIXELS, 1);
    send_pixels(5, FRAME_PIXELS, 1);
    drain;

    // No input or output gaps are allowed, including the frame boundary.
    check_full_rate = 1;
    full_rate_start = $time;
    send_pixels(1, FRAME_PIXELS, 0);
    send_pixels(2, FRAME_PIXELS, 0);
    if (($time - full_rate_start)/10 != 2*FRAME_PIXELS)
        $fatal(1, "Sustained input stream took more than two frame periods");
    drain;
    check_full_rate = 0;
    if (full_rate_inputs != 2*FRAME_PIXELS || full_rate_outputs != 2*FRAME_PIXELS)
        $fatal(1, "Full-rate stream lost or duplicated pixels");

    // Reset while TLAST is blocked: discard the old last beat and restart
    // both frame counters so no stale TLAST leaks into the next frame.
    fork
        begin
            send_pixels(5, FRAME_PIXELS, 0);
            stop_source;
        end
        begin
            hold_final_output;
            pulse_reset;
        end
    join
    send_pixels(3, FRAME_PIXELS, 1);
    send_pixels(1, FRAME_PIXELS, 1);
    drain;

    if (!input_stalls || !gray_stalls || !output_stalls || !stalled_lasts ||
        !bubbles || reset_queued != 2 || peak_queued != 5 ||
        total_sent != total_received + discarded ||
        (FRAME_HEIGHT > 1 && !line_ends))
        $fatal(1, "Missing DMA coverage: stalls=%0d/%0d/%0d last=%0d bubbles=%0d reset=%0d peak=%0d dropped=%0d",
               input_stalls, gray_stalls, output_stalls, stalled_lasts,
               bubbles, reset_queued, peak_queued, discarded);
    if (FRAME_HEIGHT >= 3 && (!clipped || !equal_threshold || !below_threshold ||
        !positive_gx || !negative_gx || !positive_gy || !negative_gy))
        $fatal(1, "Missing Sobel arithmetic/threshold coverage");
    $display("VISION_DMA_PASS width=%0d height=%0d in=%0d out=%0d discarded=%0d input_tlast=%0d output_tlast=%0d keep_checked=%0d stalls=%0d/%0d/%0d stalled_tlast=%0d resets_queued=%0d full_rate=%0d throughput=1 pixel/clock at 100MHz",
             FRAME_WIDTH, FRAME_HEIGHT, total_sent, total_received, discarded,
             total_input_lasts, total_output_lasts, total_received, input_stalls,
             gray_stalls, output_stalls, stalled_lasts, reset_queued, full_rate_outputs);
    $finish;
end

initial begin
    #2000000000;
    $fatal(1, "DMA testbench watchdog expired");
end

endmodule

module tb_vision_dma_stream_min3;
    tb_vision_dma_stream #(.FRAME_WIDTH(3), .FRAME_HEIGHT(3)) test ();
endmodule

module tb_vision_dma_stream_row1;
    tb_vision_dma_stream #(.FRAME_WIDTH(5), .FRAME_HEIGHT(1)) test ();
endmodule

module tb_vision_dma_stream_vga;
    tb_vision_dma_stream #(.FRAME_WIDTH(640), .FRAME_HEIGHT(480)) test ();
endmodule
