`timescale 1ns / 1ps

// Burst acquisition -> Hann window -> AMD FFT -> unsigned positive-half power.
// The listening path is separate, so freezing PS readout never mutes audio.
module audio_fft_engine (
    input wire clk,
    input wire rst,
    input wire signed [15:0] pcm_sample,
    input wire pcm_valid,
    input wire freeze,
    input wire peaks_busy,
    output wire busy,
    output reg frame_start,
    output reg frame_done,
    output wire bin_valid,
    output wire [11:0] bin_index,
    output wire [31:0] bin_power,
    output wire [31:0] dropped_frames,
    output reg fft_error
);
localparam IDLE=0, CONFIGURE=1, READ=2, WAIT_READ=3,
           MULTIPLY=4, SEND=5, DRAIN=6, COOLDOWN1=7, COOLDOWN2=8;
reg [3:0] state;
wire [1:0] bank_ready;
reg active_bank;
reg release_valid;
reg [11:0] address;
wire signed [15:0] read_data;
reg [15:0] window_rom [0:4095];
reg [15:0] coefficient;
reg signed [32:0] window_product;
wire [31:0] input_word = {16'd0, window_product[30:15]};
wire input_ready;
wire config_ready;
wire [63:0] fft_data;
wire fft_valid;
wire fft_last;
wire event_unexpected_last;
wire event_missing_last;
wire event_input_halt;
wire event_output_halt;
wire event_status_halt;
reg [11:0] output_index;
reg [11:0] index1, index2, index3;
reg valid1, valid2, valid3;
reg signed [28:0] real1, imag1;
reg [57:0] real_square2, imag_square2;
reg [58:0] power3;
wire [58:0] scaled_power = power3 >> 20;

initial $readmemh("hann4096.mem", window_rom);
assign busy = state != IDLE;
assign bin_valid = valid3 && !index3[11];
assign bin_index = index3;
assign bin_power = |scaled_power[58:32] ? 32'hffffffff : scaled_power[31:0];

audio_fft_capture capture_inst (
    .clk(clk), .rst(rst), .sample_valid(pcm_valid), .sample_in(pcm_sample),
    .bank_ready(bank_ready), .release_valid(release_valid),
    .release_bank(active_bank), .read_enable(state == READ),
    .read_bank(active_bank), .read_address(address), .read_data(read_data),
    .dropped_frames(dropped_frames)
);

audio_fft_4096 fft_inst (
    .aclk(clk), .aresetn(!rst),
    .s_axis_config_tdata(8'h01),
    .s_axis_config_tvalid(state == CONFIGURE),
    .s_axis_config_tready(config_ready),
    .s_axis_data_tdata(input_word), .s_axis_data_tvalid(state == SEND),
    .s_axis_data_tready(input_ready),
    .s_axis_data_tlast(address == 12'd4095),
    .m_axis_data_tdata(fft_data), .m_axis_data_tvalid(fft_valid),
    .m_axis_data_tready(1'b1), .m_axis_data_tlast(fft_last),
    .event_frame_started(), .event_tlast_unexpected(event_unexpected_last),
    .event_tlast_missing(event_missing_last),
    .event_status_channel_halt(event_status_halt),
    .event_data_in_channel_halt(event_input_halt),
    .event_data_out_channel_halt(event_output_halt)
);

always @(posedge clk) begin
    if (rst) begin
        valid1 <= 0; valid2 <= 0; valid3 <= 0;
        output_index <= 0;
        index1 <= 0; index2 <= 0; index3 <= 0;
        real1 <= 0; imag1 <= 0;
        real_square2 <= 0; imag_square2 <= 0; power3 <= 0;
    end else begin
        valid1 <= fft_valid;
        valid2 <= valid1;
        valid3 <= valid2;
        if (fft_valid) begin
            real1 <= fft_data[28:0];
            imag1 <= fft_data[60:32];
            index1 <= output_index;
            output_index <= output_index + 1'b1;
        end
        index2 <= index1;
        index3 <= index2;
        real_square2 <= real1 * real1;
        imag_square2 <= imag1 * imag1;
        power3 <= {1'b0,real_square2} + {1'b0,imag_square2};
    end
end

always @(posedge clk) begin
    if (rst) begin
        state <= IDLE;
        active_bank <= 0;
        address <= 0;
        release_valid <= 0;
        frame_start <= 0;
        frame_done <= 0;
        coefficient <= 0;
        window_product <= 0;
        fft_error <= 0;
    end else begin
        release_valid <= 0;
        frame_start <= 0;
        frame_done <= 0;
        if (event_unexpected_last || event_missing_last || event_output_halt)
            fft_error <= 1;
        case (state)
            IDLE: if (!freeze && !peaks_busy && |bank_ready) begin
                active_bank <= bank_ready[0] ? 1'b0 : 1'b1;
                address <= 0;
                state <= CONFIGURE;
            end
            CONFIGURE: if (config_ready) begin
                frame_start <= 1;
                state <= READ;
            end
            READ: begin
                coefficient <= window_rom[address];
                state <= WAIT_READ;
            end
            WAIT_READ: state <= MULTIPLY;
            MULTIPLY: begin
                window_product <= read_data * $signed({1'b0,coefficient});
                state <= SEND;
            end
            SEND: if (input_ready) begin
                if (address == 12'd4095) begin
                    release_valid <= 1;
                    state <= DRAIN;
                end else begin
                    address <= address + 1'b1;
                    state <= READ;
                end
            end
            DRAIN: if (valid3 && index3 == 12'd4095) begin
                frame_done <= 1;
                state <= COOLDOWN1;
            end
            COOLDOWN1: state <= COOLDOWN2;
            COOLDOWN2: state <= IDLE;
            default: state <= IDLE;
        endcase
    end
end
endmodule
