#include "hand_conv.h"

enum {
    HC_MAX_CHANNELS = 1024,
    HC_MAX_KERNEL = 5,
    HC_TILE_H = 8,
    HC_TILE_W = 8,
    HC_IC_TILE = 8,
    HC_OC_TILE = 4,
    HC_MAX_PATCH = HC_TILE_H * 2 + HC_MAX_KERNEL
};

// Cache a complete output-channel tile of 1x1 weights across spatial tiles.
// Eight independent partial sums also hide FP32 feedback latency for dense
// layers, where there is only one spatial output to accumulate into.
static void hand_conv_pointwise(float *input, float *weights, float *bias,
                               float *output, int height, int width,
                               int channels, int outputs, int stride,
                               int pad_top, int pad_left, int out_h, int out_w,
                               int groups) {
    float wc[HC_OC_TILE][HC_MAX_CHANNELS];
    float pixels[HC_IC_TILE][HC_TILE_H][HC_TILE_W];
    float partial[HC_OC_TILE][8][HC_TILE_H * HC_TILE_W];
#pragma HLS ARRAY_PARTITION variable=wc complete dim=1
#pragma HLS ARRAY_PARTITION variable=partial complete dim=1
#pragma HLS ARRAY_PARTITION variable=partial complete dim=2
    int icpg = channels / groups;
    int ocpg = outputs / groups;
pw_group:
    for (int g = 0; g < groups; ++g) {
pw_oc:
        for (int oc0 = 0; oc0 < ocpg; oc0 += HC_OC_TILE) {
pw_weights:
            for (int lane = 0; lane < HC_OC_TILE; ++lane)
                for (int ic = 0; ic < icpg; ++ic) {
#pragma HLS PIPELINE II=1
                    wc[lane][ic] = (oc0 + lane < ocpg)
                        ? weights[(g * ocpg + oc0 + lane) * icpg + ic] : 0.0f;
                }
pw_row:
            for (int y0 = 0; y0 < out_h; y0 += HC_TILE_H) {
pw_col:
                for (int x0 = 0; x0 < out_w; x0 += HC_TILE_W) {
                    int vh = (out_h-y0 < HC_TILE_H) ? out_h-y0 : HC_TILE_H;
                    int vw = (out_w-x0 < HC_TILE_W) ? out_w-x0 : HC_TILE_W;
pw_init:
                    for (int p = 0; p < HC_TILE_H * HC_TILE_W; ++p) {
#pragma HLS PIPELINE II=1
                        for (int lane = 0; lane < HC_OC_TILE; ++lane) {
#pragma HLS UNROLL
                            for (int bank = 0; bank < 8; ++bank) {
#pragma HLS UNROLL
                                partial[lane][bank][p] = 0.0f;
                            }
                        }
                    }
pw_ic:
                    for (int ic0 = 0; ic0 < icpg; ic0 += HC_IC_TILE) {
                        int valid_ic = (icpg-ic0 < HC_IC_TILE) ? icpg-ic0 : HC_IC_TILE;
pw_load:
                        for (int ci = 0; ci < valid_ic; ++ci)
                            for (int y = 0; y < HC_TILE_H; ++y)
                                for (int x = 0; x < HC_TILE_W; ++x) {
#pragma HLS PIPELINE II=1
                                    int iy = (y0+y)*stride-pad_top;
                                    int ix = (x0+x)*stride-pad_left;
                                    pixels[ci][y][x] = (y < vh && x < vw && iy >= 0 && iy < height && ix >= 0 && ix < width)
                                        ? input[(g*icpg+ic0+ci)*height*width+iy*width+ix] : 0.0f;
                                }
pw_ci:
                        for (int ci = 0; ci < valid_ic; ++ci) {
pw_mac:
                            for (int p = 0; p < HC_TILE_H * HC_TILE_W; ++p) {
#pragma HLS PIPELINE II=1
                                float value = pixels[ci][p / HC_TILE_W][p % HC_TILE_W];
                                for (int lane = 0; lane < HC_OC_TILE; ++lane) {
#pragma HLS UNROLL
                                    partial[lane][ci][p] += value * wc[lane][ic0+ci];
                                }
                            }
                        }
                    }
pw_store:
                    for (int lane = 0; lane < HC_OC_TILE; ++lane)
                        for (int y = 0; y < vh; ++y)
                            for (int x = 0; x < vw; ++x) {
#pragma HLS PIPELINE II=1
                                int p = y*HC_TILE_W+x;
                                float s0 = partial[lane][0][p]+partial[lane][1][p];
                                float s1 = partial[lane][2][p]+partial[lane][3][p];
                                float s2 = partial[lane][4][p]+partial[lane][5][p];
                                float s3 = partial[lane][6][p]+partial[lane][7][p];
                                if (oc0+lane < ocpg) {
                                    int oc = g*ocpg+oc0+lane;
                                    output[oc*out_h*out_w+(y0+y)*out_w+x0+x] =
                                        ((s0+s1)+(s2+s3))+bias[oc];
                                }
                            }
                }
            }
        }
    }
}

static void hand_conv_depthwise(float *input, float *weights, float *bias,
                                float *output, int height, int width,
                                int channels, int kernel, int stride,
                                int pad_top, int pad_left, int out_h, int out_w) {
    float patch[HC_OC_TILE][HC_MAX_PATCH][HC_MAX_PATCH];
    float wcache[HC_OC_TILE][HC_MAX_KERNEL][HC_MAX_KERNEL];
    float accum[HC_OC_TILE][HC_TILE_H][HC_TILE_W];
#pragma HLS ARRAY_PARTITION variable=patch complete dim=1
#pragma HLS ARRAY_PARTITION variable=wcache complete dim=1
#pragma HLS ARRAY_PARTITION variable=accum complete dim=1

dw_oc_tile:
    for (int oc0 = 0; oc0 < channels; oc0 += HC_OC_TILE) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
dw_load_weights:
        for (int lane = 0; lane < HC_OC_TILE; ++lane)
            for (int ky = 0; ky < kernel; ++ky)
                for (int kx = 0; kx < kernel; ++kx) {
#pragma HLS PIPELINE II=1
                    wcache[lane][ky][kx] = (lane + oc0 < channels)
                        ? weights[(oc0 + lane) * kernel * kernel + ky * kernel + kx] : 0.0f;
                }
dw_row_tile:
        for (int oy0 = 0; oy0 < out_h; oy0 += HC_TILE_H) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
dw_col_tile:
            for (int ox0 = 0; ox0 < out_w; ox0 += HC_TILE_W) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
                int valid_h = (out_h - oy0 < HC_TILE_H) ? out_h - oy0 : HC_TILE_H;
                int valid_w = (out_w - ox0 < HC_TILE_W) ? out_w - ox0 : HC_TILE_W;
                int patch_h = (valid_h - 1) * stride + kernel;
                int patch_w = (valid_w - 1) * stride + kernel;

dw_load_patch:
                for (int lane = 0; lane < HC_OC_TILE; ++lane)
                    for (int py = 0; py < patch_h; ++py)
                        for (int px = 0; px < patch_w; ++px) {
#pragma HLS PIPELINE II=1
                            int iy = oy0 * stride - pad_top + py;
                            int ix = ox0 * stride - pad_left + px;
                            patch[lane][py][px] = (lane + oc0 < channels &&
                                                   iy >= 0 && iy < height && ix >= 0 && ix < width)
                                ? input[(oc0 + lane) * height * width + iy * width + ix] : 0.0f;
                        }
dw_init:
                for (int lane = 0; lane < HC_OC_TILE; ++lane)
                    for (int ty = 0; ty < HC_TILE_H; ++ty)
                        for (int tx = 0; tx < HC_TILE_W; ++tx)
#pragma HLS PIPELINE II=1
                            accum[lane][ty][tx] = (lane + oc0 < channels) ? bias[oc0 + lane] : 0.0f;
dw_mac:
                for (int ky = 0; ky < kernel; ++ky)
                    for (int kx = 0; kx < kernel; ++kx)
                        for (int ty = 0; ty < valid_h; ++ty)
                            for (int tx = 0; tx < valid_w; ++tx) {
#pragma HLS PIPELINE II=1
                                    for (int lane = 0; lane < HC_OC_TILE; ++lane) {
#pragma HLS UNROLL
                                        accum[lane][ty][tx] += patch[lane][ty * stride + ky][tx * stride + kx] * wcache[lane][ky][kx];
                                    }
                            }
dw_store:
                for (int lane = 0; lane < HC_OC_TILE; ++lane)
                    for (int ty = 0; ty < valid_h; ++ty)
                        for (int tx = 0; tx < valid_w; ++tx) {
#pragma HLS PIPELINE II=1
                            if (lane + oc0 < channels)
                                output[(oc0 + lane) * out_h * out_w + (oy0 + ty) * out_w + ox0 + tx] = accum[lane][ty][tx];
                        }
            }
    }
}
}

void hand_conv(float *input, float *weights, float *bias, float *output,
               int height, int width, int channels, int outputs, int kernel,
               int stride, int pad_top, int pad_left, int out_h, int out_w,
               int groups) {
#pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=weights offset=slave bundle=gmem max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=bias offset=slave bundle=gmem max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem max_write_burst_length=64
#pragma HLS INTERFACE s_axilite port=input bundle=control
#pragma HLS INTERFACE s_axilite port=weights bundle=control
#pragma HLS INTERFACE s_axilite port=bias bundle=control
#pragma HLS INTERFACE s_axilite port=output bundle=control
#pragma HLS INTERFACE s_axilite port=height bundle=control
#pragma HLS INTERFACE s_axilite port=width bundle=control
#pragma HLS INTERFACE s_axilite port=channels bundle=control
#pragma HLS INTERFACE s_axilite port=outputs bundle=control
#pragma HLS INTERFACE s_axilite port=kernel bundle=control
#pragma HLS INTERFACE s_axilite port=stride bundle=control
#pragma HLS INTERFACE s_axilite port=pad_top bundle=control
#pragma HLS INTERFACE s_axilite port=pad_left bundle=control
#pragma HLS INTERFACE s_axilite port=out_h bundle=control
#pragma HLS INTERFACE s_axilite port=out_w bundle=control
#pragma HLS INTERFACE s_axilite port=groups bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    if (height <= 0 || width <= 0 || channels <= 0 || outputs <= 0 || kernel < 1 || kernel > HC_MAX_KERNEL ||
        (kernel != 1 && kernel != 3 && kernel != 5) || (stride != 1 && stride != 2) || out_h <= 0 || out_w <= 0 ||
        groups <= 0 || channels > HC_MAX_CHANNELS || outputs > HC_MAX_CHANNELS || channels % groups || outputs % groups)
        return;

    if (groups == channels && outputs == channels) {
        hand_conv_depthwise(input, weights, bias, output, height, width, channels, kernel, stride,
                            pad_top, pad_left, out_h, out_w);
        return;
    }

    if (kernel == 1) {
        hand_conv_pointwise(input, weights, bias, output, height, width,
                            channels, outputs, stride, pad_top, pad_left,
                            out_h, out_w, groups);
        return;
    }

    float patch[HC_IC_TILE][HC_MAX_PATCH][HC_MAX_PATCH];
    float wcache[HC_OC_TILE][HC_IC_TILE][HC_MAX_KERNEL][HC_MAX_KERNEL];
    float accum[HC_OC_TILE][HC_TILE_H][HC_TILE_W];
#pragma HLS ARRAY_PARTITION variable=patch complete dim=1
#pragma HLS ARRAY_PARTITION variable=wcache complete dim=1
#pragma HLS ARRAY_PARTITION variable=accum complete dim=1
    int in_per_group = channels / groups;
    int out_per_group = outputs / groups;

group_loop:
    for (int group = 0; group < groups; ++group) {
oc_tile:
        for (int ocg0 = 0; ocg0 < out_per_group; ocg0 += HC_OC_TILE) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
row_tile:
            for (int oy0 = 0; oy0 < out_h; oy0 += HC_TILE_H) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
col_tile:
                for (int ox0 = 0; ox0 < out_w; ox0 += HC_TILE_W) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
                    int valid_h = (out_h - oy0 < HC_TILE_H) ? out_h - oy0 : HC_TILE_H;
                    int valid_w = (out_w - ox0 < HC_TILE_W) ? out_w - ox0 : HC_TILE_W;
                    int patch_h = (valid_h - 1) * stride + kernel;
                    int patch_w = (valid_w - 1) * stride + kernel;
init_accum:
                    for (int lane = 0; lane < HC_OC_TILE; ++lane)
                        for (int ty = 0; ty < valid_h; ++ty)
                            for (int tx = 0; tx < valid_w; ++tx)
#pragma HLS PIPELINE II=1
                                accum[lane][ty][tx] = (ocg0 + lane < out_per_group) ? bias[group * out_per_group + ocg0 + lane] : 0.0f;
ic_tile:
                    for (int ic0 = 0; ic0 < in_per_group; ic0 += HC_IC_TILE) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=128
load_patch:
                        for (int ic = 0; ic < HC_IC_TILE; ++ic)
                            for (int py = 0; py < patch_h; ++py)
                                for (int px = 0; px < patch_w; ++px) {
#pragma HLS PIPELINE II=1
                                    int iy = oy0 * stride - pad_top + py;
                                    int ix = ox0 * stride - pad_left + px;
                                    patch[ic][py][px] = (ic0 + ic < in_per_group &&
                                                         iy >= 0 && iy < height && ix >= 0 && ix < width)
                                        ? input[(group * in_per_group + ic0 + ic) * height * width + iy * width + ix] : 0.0f;
                                }
load_weights:
                        for (int lane = 0; lane < HC_OC_TILE; ++lane)
                            for (int ic = 0; ic < HC_IC_TILE; ++ic)
                                for (int ky = 0; ky < kernel; ++ky)
                                    for (int kx = 0; kx < kernel; ++kx) {
#pragma HLS PIPELINE II=1
                                        wcache[lane][ic][ky][kx] = (ocg0 + lane < out_per_group && ic0 + ic < in_per_group)
                                            ? weights[((group * out_per_group + ocg0 + lane) * in_per_group + ic0 + ic) * kernel * kernel + ky * kernel + kx] : 0.0f;
                                    }
mac_kernel:
                        for (int ky = 0; ky < kernel; ++ky)
                            for (int kx = 0; kx < kernel; ++kx)
                                for (int ic = 0; ic < in_per_group - ic0 && ic < HC_IC_TILE; ++ic)
                                    for (int ty = 0; ty < valid_h; ++ty)
                                        for (int tx = 0; tx < valid_w; ++tx) {
#pragma HLS PIPELINE II=4
                                                for (int lane = 0; lane < HC_OC_TILE; ++lane) {
#pragma HLS UNROLL
                                                    accum[lane][ty][tx] += patch[ic][ty * stride + ky][tx * stride + kx] * wcache[lane][ic][ky][kx];
                                                }
                                        }
                    }
store_output:
                    for (int lane = 0; lane < HC_OC_TILE; ++lane)
                        for (int ty = 0; ty < valid_h; ++ty)
                            for (int tx = 0; tx < valid_w; ++tx) {
#pragma HLS PIPELINE II=1
                                if (ocg0 + lane < out_per_group)
                                    output[(group * out_per_group + ocg0 + lane) * out_h * out_w + (oy0 + ty) * out_w + ox0 + tx] = accum[lane][ty][tx];
                            }
                }
            }
        }
    }
}
