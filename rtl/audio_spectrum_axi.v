`timescale 1ns / 1ps

// One snapshot, with freeze-and-idle ownership transferred to the PS.
module audio_spectrum_axi #(
    parameter [31:0] PEAK_THRESHOLD = 32'd0
)(
    input  wire        clk,
    input  wire        rst,
    input  wire [13:0] s_axi_awaddr,
    input  wire        s_axi_awvalid,
    output wire        s_axi_awready,
    input  wire [31:0] s_axi_wdata,
    input  wire [3:0]  s_axi_wstrb,
    input  wire        s_axi_wvalid,
    output wire        s_axi_wready,
    output reg  [1:0]  s_axi_bresp,
    output reg         s_axi_bvalid,
    input  wire        s_axi_bready,
    input  wire [13:0] s_axi_araddr,
    input  wire        s_axi_arvalid,
    output wire        s_axi_arready,
    output reg  [31:0] s_axi_rdata,
    output reg  [1:0]  s_axi_rresp,
    output reg         s_axi_rvalid,
    input  wire        s_axi_rready,
    input  wire        bin_valid,
    input  wire [11:0] bin_index,
    input  wire [31:0] bin_power,
    input  wire        fft_busy,
    input  wire        fft_error,
    input  wire        frame_start,
    input  wire        frame_done,
    input  wire [31:0] dropped_frames,
    output wire [1:0]  analysis_test_mode,
    output wire        freeze,
    output wire        peaks_busy,
    output wire        frame_hold
);

localparam [3:0] IDLE = 4'd0, CAPTURE = 4'd1,
                 SCAN_ISSUE = 4'd2, SCAN_WAIT = 4'd3,
                 SCAN_EVAL = 4'd4, INSERT_FIND = 4'd5,
                 INSERT_SHIFT = 4'd6, SCAN_ADVANCE = 4'd7,
                 PUBLISH = 4'd8;

reg [3:0] state;
reg freeze_requested;
reg [1:0] analysis_test_mode_reg;
reg snapshot_ready;
reg error_sticky;
reg [31:0] snapshot_sequence;
reg [11:0] captured_bins;
reg frame_bad;
(* ram_style = "block" *) reg [31:0] spectrum_ram [0:2047];
reg [31:0] ram_read_data;
reg [10:0] scan_index;
reg [31:0] left_power, center_power;
reg [10:0] peak_bin [0:23];
reg [31:0] peak_power [0:23];
reg [4:0] peak_count;
reg [10:0] candidate_bin;
reg [31:0] candidate_power;
reg [4:0] insert_index, shift_index;

reg [13:0] awaddr_hold;
reg [31:0] wdata_hold;
reg [3:0] wstrb_hold;
reg aw_pending, w_pending;
reg read_pending, read_is_spectrum;
reg [31:0] register_read_hold;
reg [1:0] read_response_hold;

wire busy = fft_busy || peaks_busy;
// fft_busy can precede frame_start while the engine configures the FFT.
// A freeze written during that interval must let the active frame finish.
wire accept_start = frame_start && state == IDLE &&
                    (!freeze_requested || fft_busy) && !rst;
wire capture_active = state == CAPTURE || accept_start;
wire [11:0] expected_bin = accept_start ? 12'd0 : captured_bins;
wire ordered_bin = bin_index == expected_bin && bin_index < 12'd2048;
wire capture_error = (bin_valid && !ordered_bin) ||
                     (frame_start && !accept_start);
wire complete_frame = !accept_start &&
                      ((captured_bins == 12'd2048 && !bin_valid) ||
                       (captured_bins == 12'd2047 && bin_valid && ordered_bin));

wire aw_fire = s_axi_awvalid && s_axi_awready;
wire w_fire = s_axi_wvalid && s_axi_wready;
wire [13:0] write_address = aw_pending ? awaddr_hold : s_axi_awaddr;
wire [31:0] write_data = w_pending ? wdata_hold : s_axi_wdata;
wire [3:0] write_strobe = w_pending ? wstrb_hold : s_axi_wstrb;
wire write_commit = !s_axi_bvalid && (aw_pending || aw_fire) &&
                    (w_pending || w_fire);
wire ar_fire = s_axi_arvalid && s_axi_arready;
wire spectrum_address = s_axi_araddr >= 14'h1000 &&
                        s_axi_araddr < 14'h3000;
wire peak_address = s_axi_araddr >= 14'h0100 &&
                    s_axi_araddr < 14'h01c0;
wire [13:0] spectrum_offset = s_axi_araddr - 14'h1000;
wire [4:0] peak_read_index = s_axi_araddr[7:3];
wire read_spectrum = ar_fire && spectrum_address &&
                     s_axi_araddr[1:0] == 2'b00 && snapshot_ready && !busy;
wire ram_read_enable = !rst && (state == SCAN_ISSUE || read_spectrum);
wire [10:0] ram_read_address = state == SCAN_ISSUE ? scan_index :
                              spectrum_offset[12:2];

assign analysis_test_mode = analysis_test_mode_reg;
assign freeze = freeze_requested || rst;
assign peaks_busy = state != IDLE && state != CAPTURE;
assign frame_hold = freeze || fft_busy || state != IDLE;
assign s_axi_awready = !rst && !aw_pending && !s_axi_bvalid;
assign s_axi_wready = !rst && !w_pending && !s_axi_bvalid;
assign s_axi_arready = !rst && !read_pending && !s_axi_rvalid;

// No RAM reset: only a complete frame makes its contents readable. Port B
// serves the peak scanner or AXI reads; those owners never overlap when idle.
always @(posedge clk) begin
    if (!rst && capture_active && bin_valid && ordered_bin)
        spectrum_ram[bin_index[10:0]] <= bin_power;
    if (ram_read_enable)
        ram_read_data <= spectrum_ram[ram_read_address];
end

always @(posedge clk) begin
    if (rst) begin
        state <= IDLE;
        freeze_requested <= 1'b0;
        analysis_test_mode_reg <= 2'b00;
        snapshot_ready <= 1'b0;
        error_sticky <= 1'b0;
        snapshot_sequence <= 32'd0;
        captured_bins <= 12'd0;
        frame_bad <= 1'b0;
        scan_index <= 11'd5;
        left_power <= 32'd0;
        center_power <= 32'd0;
        peak_count <= 5'd0;
        candidate_bin <= 11'd0;
        candidate_power <= 32'd0;
        insert_index <= 5'd0;
        shift_index <= 5'd0;
        awaddr_hold <= 14'd0;
        wdata_hold <= 32'd0;
        wstrb_hold <= 4'd0;
        aw_pending <= 1'b0;
        w_pending <= 1'b0;
        s_axi_bresp <= 2'b00;
        s_axi_bvalid <= 1'b0;
        read_pending <= 1'b0;
        read_is_spectrum <= 1'b0;
        register_read_hold <= 32'd0;
        read_response_hold <= 2'b00;
        s_axi_rdata <= 32'd0;
        s_axi_rresp <= 2'b00;
        s_axi_rvalid <= 1'b0;
    end else begin
        if (aw_fire) begin
            awaddr_hold <= s_axi_awaddr;
            aw_pending <= 1'b1;
        end
        if (w_fire) begin
            wdata_hold <= s_axi_wdata;
            wstrb_hold <= s_axi_wstrb;
            w_pending <= 1'b1;
        end
        if (write_commit) begin
            aw_pending <= 1'b0;
            w_pending <= 1'b0;
            s_axi_bvalid <= 1'b1;
            s_axi_bresp <= 2'b00;
            case (write_address)
                14'h000c: if (write_strobe[0]) begin
                    freeze_requested <= write_data[0];
                    analysis_test_mode_reg <= write_data[2:1];
                end
                14'h0004: if (write_strobe[0] && write_data[3]) error_sticky <= 1'b0;
                default: s_axi_bresp <= 2'b10;
            endcase
        end else if (s_axi_bvalid && s_axi_bready) begin
            s_axi_bvalid <= 1'b0;
        end

        // AR launches the synchronous RAM access. The following clock
        // presents RVALID, and RDATA/RRESP remain held until RREADY.
        if (ar_fire) begin
            read_pending <= 1'b1;
            read_is_spectrum <= read_spectrum;
            register_read_hold <= 32'd0;
            read_response_hold <= 2'b00;
            if (s_axi_araddr[1:0] != 2'b00) begin
                read_response_hold <= 2'b10;
            end else if (spectrum_address) begin
                if (!read_spectrum) read_response_hold <= 2'b10;
            end else if (peak_address) begin
                if (!snapshot_ready || busy) begin
                    read_response_hold <= 2'b10;
                end else if (peak_read_index < peak_count) begin
                    if (s_axi_araddr[2])
                        register_read_hold <= peak_power[peak_read_index];
                    else
                        register_read_hold <= {21'd0, peak_bin[peak_read_index]};
                end
            end else begin
                case (s_axi_araddr)
                    14'h0000: register_read_hold <= 32'h4155_4434;
                    14'h0004: register_read_hold <= {28'd0, error_sticky,
                                                  freeze_requested && !busy,
                                                  snapshot_ready, busy};
                    14'h0008: register_read_hold <= snapshot_sequence;
                    14'h000c: register_read_hold <= {29'd0, analysis_test_mode_reg,
                                                   freeze_requested};
                    14'h0010: register_read_hold <= 32'd4096;
                    14'h0014: register_read_hold <= 32'd48000;
                    14'h0018: register_read_hold <= dropped_frames;
                    14'h001c: register_read_hold <= {27'd0, peak_count};
                    14'h0020: register_read_hold <= 32'd1048576;
                    14'h0024: register_read_hold <= 32'd4095;
                    default: read_response_hold <= 2'b10;
                endcase
            end
        end else if (read_pending) begin
            s_axi_rdata <= read_is_spectrum ? ram_read_data : register_read_hold;
            s_axi_rresp <= read_response_hold;
            s_axi_rvalid <= 1'b1;
            read_pending <= 1'b0;
        end else if (s_axi_rvalid && s_axi_rready) begin
            s_axi_rvalid <= 1'b0;
        end

        if (fft_error || (frame_start && !accept_start) ||
            ((bin_valid || frame_done) && !capture_active))
            error_sticky <= 1'b1;

        if (capture_active) begin
            state <= CAPTURE;
            if (accept_start) begin
                captured_bins <= bin_valid && ordered_bin ? 12'd1 : 12'd0;
                snapshot_ready <= 1'b0;
                peak_count <= 5'd0;
                frame_bad <= capture_error;
            end else begin
                if (bin_valid && ordered_bin) captured_bins <= captured_bins + 1'b1;
                if (capture_error) frame_bad <= 1'b1;
            end
            if (capture_error) error_sticky <= 1'b1;
            if (frame_done) begin
                if (complete_frame && !capture_error && !frame_bad) begin
                    scan_index <= 11'd5;
                    state <= SCAN_ISSUE;
                end else begin
                    error_sticky <= 1'b1;
                    state <= IDLE;
                end
            end
        end else begin
            case (state)
                SCAN_ISSUE: state <= SCAN_WAIT;
                SCAN_WAIT: state <= SCAN_EVAL;
                SCAN_EVAL: begin
                    left_power <= center_power;
                    center_power <= ram_read_data;
                    if (scan_index >= 11'd7 && center_power > left_power &&
                        center_power > ram_read_data && center_power > PEAK_THRESHOLD) begin
                        candidate_bin <= scan_index - 1'b1;
                        candidate_power <= center_power;
                        insert_index <= 5'd0;
                        state <= INSERT_FIND;
                    end else begin
                        state <= SCAN_ADVANCE;
                    end
                end
                INSERT_FIND: begin
                    if (insert_index == peak_count) begin
                        peak_bin[insert_index] <= candidate_bin;
                        peak_power[insert_index] <= candidate_power;
                        peak_count <= peak_count + 1'b1;
                        state <= SCAN_ADVANCE;
                    end else if (candidate_power > peak_power[insert_index]) begin
                        shift_index <= peak_count < 5'd24 ? peak_count : 5'd23;
                        state <= INSERT_SHIFT;
                    end else if (insert_index == 5'd23) begin
                        state <= SCAN_ADVANCE;
                    end else begin
                        insert_index <= insert_index + 1'b1;
                    end
                end
                INSERT_SHIFT: begin
                    if (shift_index == insert_index) begin
                        peak_bin[insert_index] <= candidate_bin;
                        peak_power[insert_index] <= candidate_power;
                        if (peak_count < 5'd24) peak_count <= peak_count + 1'b1;
                        state <= SCAN_ADVANCE;
                    end else begin
                        peak_bin[shift_index] <= peak_bin[shift_index - 1'b1];
                        peak_power[shift_index] <= peak_power[shift_index - 1'b1];
                        shift_index <= shift_index - 1'b1;
                    end
                end
                SCAN_ADVANCE: begin
                    if (scan_index == 11'd601) begin
                        state <= PUBLISH;
                    end else begin
                        scan_index <= scan_index + 1'b1;
                        state <= SCAN_ISSUE;
                    end
                end
                PUBLISH: begin
                    snapshot_sequence <= snapshot_sequence + 1'b1;
                    snapshot_ready <= 1'b1;
                    state <= IDLE;
                end
                default: begin end
            endcase
        end
    end
end

endmodule
