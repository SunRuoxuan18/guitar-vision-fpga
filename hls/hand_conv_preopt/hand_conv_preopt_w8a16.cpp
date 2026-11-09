#include <ap_int.h>
#if defined(HAND_PW_FIRST_FOLD) && !defined(HAND_PW_FUSED)
#error HAND_PW_FIRST_FOLD requires the fused pointwise path
#endif
#ifdef HAND_BURST
#include <hls_burst_maxi.h>
typedef hls::burst_maxi<long long> output_t;
#else
typedef long long *output_t;
#endif

enum { VALUE_BITS = 16, WEIGHT_BITS = 8, SUM_BITS = 40 };
typedef ap_int<VALUE_BITS> value_t;
typedef ap_int<WEIGHT_BITS> weight_t;
typedef ap_int<SUM_BITS> sum_t;
typedef ap_uint<8*VALUE_BITS> packed_t;
typedef ap_uint<8*WEIGHT_BITS> weight_packed_t;
#ifdef HAND_REALTIME
enum { PW_SIMD = 8, SP_SIMD = 1 };
#elif defined(HAND_POINTWISE8)
enum { PW_SIMD = 8, SP_SIMD = 4 };
#else
enum { PW_SIMD = 4, SP_SIMD = 4 };
#endif
enum { LANES = 8, SIMD = SP_SIMD, TILE = 4, PIXELS = TILE*TILE, PATCH = 11 };
#ifdef HAND_PW_FUSED
enum { PW_WIDTH = 16, PW_HEIGHT = 2, PW_PIXELS = PW_WIDTH*PW_HEIGHT };
#elif defined(HAND_PW_CACHE)
enum { PW_WIDTH = 8, PW_HEIGHT = 2, PW_PIXELS = PW_WIDTH*PW_HEIGHT };
#elif defined(HAND_BANKED)
enum { PW_WIDTH = 8, PW_HEIGHT = 4, PW_PIXELS = PW_WIDTH*PW_HEIGHT };
#elif defined(HAND_BURST)
enum { PW_WIDTH = 16, PW_HEIGHT = 4, PW_PIXELS = PW_WIDTH*PW_HEIGHT };
#else
enum { PW_WIDTH = TILE, PW_HEIGHT = TILE, PW_PIXELS = PIXELS };
#endif

// Packed NHWC8 / OHWI8 operands, NCHW sums, symmetric dynamic activation
// and per-output weight quantization. Depthwise weights use HWO8 packing.
// Keep integer sums intact; PS applies scales and bias once per output.
static void pointwise(packed_t *input, weight_packed_t *weights, output_t &output,
                      int height, int width, int channels, int outputs,
                      int stride, int top, int left, int oh, int ow) {
    weight_t wc[LANES][1024];
#ifndef HAND_PW_FUSED
    value_t tile[PW_SIMD][PW_PIXELS];
#endif
    sum_t sums[LANES][PW_PIXELS];
#ifdef HAND_PW_CACHE
    packed_t input_tile[PW_PIXELS*128];
#pragma HLS BIND_STORAGE variable=input_tile type=ram_2p impl=bram
#endif
#pragma HLS ARRAY_PARTITION variable=wc complete dim=1
#ifdef HAND_COMPACT
#pragma HLS ARRAY_PARTITION variable=wc cyclic factor=4 dim=2
#pragma HLS BIND_STORAGE variable=wc type=ram_t2p impl=bram
#else
#pragma HLS ARRAY_PARTITION variable=wc cyclic factor=8 dim=2
#pragma HLS BIND_STORAGE variable=wc type=ram_2p impl=lutram
#endif
#ifndef HAND_PW_FUSED
#pragma HLS ARRAY_PARTITION variable=tile complete dim=1
#endif
#pragma HLS ARRAY_PARTITION variable=sums complete dim=1
    for (int oc0 = 0; oc0 < outputs; oc0 += LANES) {
        for (int lane = 0; lane < LANES; ++lane)
            for (int block = 0; block < (channels+7)/8; ++block) {
#pragma HLS PIPELINE II=1
                weight_packed_t word = oc0+lane < outputs ? weights[(oc0+lane)*((channels+7)/8)+block] : weight_packed_t(0);
                for (int ci = 0; ci < 8; ++ci) {
#pragma HLS UNROLL
                    wc[lane][block*8+ci] = word.range((ci+1)*WEIGHT_BITS-1,ci*WEIGHT_BITS);
                }
            }
        for (int y0 = 0; y0 < oh; y0 += PW_HEIGHT)
            for (int x0 = 0; x0 < ow; x0 += PW_WIDTH) {
#ifdef HAND_PW_BOUNDED
                // Preserve the fixed BRAM row pitch while trimming trailing
                // positions that cannot contribute to an output pixel.
                int valid_h = oh-y0 < PW_HEIGHT ? oh-y0 : PW_HEIGHT;
                int valid_w = ow-x0 < PW_WIDTH ? ow-x0 : PW_WIDTH;
                int active_pixels = (valid_h-1)*PW_WIDTH+valid_w;
#else
                const int active_pixels = PW_PIXELS;
#endif
#ifdef HAND_PW_CACHE
                // Read complete NHWC8 pixel vectors contiguously, then reuse
                // BRAM instead of issuing a strided DDR read per channel fold.
                for (int p = 0; p < PW_PIXELS; ++p) {
                    int y = y0+p/PW_WIDTH, x = x0+p%PW_WIDTH;
                    int iy = y*stride-top, ix = x*stride-left;
                    int blocks = (channels+7)/8;
                    if (y < oh && x < ow && iy >= 0 && iy < height && ix >= 0 && ix < width) {
                        for (int block = 0; block < blocks; ++block) {
#pragma HLS PIPELINE II=1
                            input_tile[p*128+block] = input[(iy*width+ix)*blocks+block];
                        }
                    } else {
                        for (int block = 0; block < blocks; ++block) {
#pragma HLS PIPELINE II=1
                            input_tile[p*128+block] = 0;
                        }
                    }
                }
#endif
#ifndef HAND_PW_FIRST_FOLD
                for (int p = 0; p < active_pixels; ++p) {
#pragma HLS PIPELINE II=1
                    for (int lane = 0; lane < LANES; ++lane) {
#pragma HLS UNROLL
                        sums[lane][p] = 0;
                    }
                }
#endif
                for (int ic0 = 0; ic0 < channels; ic0 += PW_SIMD) {
#ifdef HAND_PW_FUSED
pw_fused_mac:
                    for (int p = 0; p < active_pixels; ++p) {
#pragma HLS PIPELINE II=1
                        int y = y0+p/PW_WIDTH, x = x0+p%PW_WIDTH;
                        int iy = y*stride-top, ix = x*stride-left;
                        packed_t word = y < oh && x < ow && iy >= 0 && iy < height && ix >= 0 && ix < width ?
                            input[(iy*width+ix)*((channels+7)/8)+ic0/8] : packed_t(0);
                        // Each AXI operand feeds the MAC pipeline directly;
                        // there is no separate per-fold tile load/drain phase.
                        for (int lane = 0; lane < LANES; ++lane) {
#pragma HLS UNROLL
                            sum_t add = 0;
                            for (int ci = 0; ci < PW_SIMD; ++ci) {
#pragma HLS UNROLL
                                value_t pixel = word.range((ic0%8+ci+1)*VALUE_BITS-1,(ic0%8+ci)*VALUE_BITS);
                                weight_t weight = ic0+ci < channels ? wc[lane][ic0+ci] : weight_t(0);
                                ap_int<VALUE_BITS+WEIGHT_BITS> product = pixel*weight;
#ifdef PREOPT_FABRIC_MUL
#pragma HLS BIND_OP variable=product op=mul impl=fabric
#endif
                                add += product;
                            }
#ifdef HAND_PW_FIRST_FOLD
                            sums[lane][p] = ic0 == 0 ? add : sum_t(sums[lane][p]+add);
#else
                            sums[lane][p] += add;
#endif
                        }
                    }
#else
                    for (int p = 0; p < PW_PIXELS; ++p) {
#pragma HLS PIPELINE II=1
#ifdef HAND_PW_CACHE
                        packed_t word = input_tile[p*128+ic0/8];
#else
                        int y = y0+p/PW_WIDTH, x = x0+p%PW_WIDTH;
                        int iy = y*stride-top, ix = x*stride-left;
                        packed_t word = y < oh && x < ow && iy >= 0 && iy < height && ix >= 0 && ix < width ?
                            input[(iy*width+ix)*((channels+7)/8)+ic0/8] : packed_t(0);
#endif
                        for (int ci = 0; ci < PW_SIMD; ++ci) {
#pragma HLS UNROLL
                            tile[ci][p] = word.range((ic0%8+ci+1)*VALUE_BITS-1,(ic0%8+ci)*VALUE_BITS);
                        }
                    }
pw_mac:
                    for (int p = 0; p < PW_PIXELS; ++p) {
#pragma HLS PIPELINE II=1
                        for (int lane = 0; lane < LANES; ++lane) {
#pragma HLS UNROLL
                            sum_t add = 0;
                            for (int ci = 0; ci < PW_SIMD; ++ci) {
#pragma HLS UNROLL
                                weight_t weight = ic0+ci < channels ? wc[lane][ic0+ci] : weight_t(0);
                                ap_int<VALUE_BITS+WEIGHT_BITS> product = tile[ci][p]*weight;
#ifdef PREOPT_FABRIC_MUL
#pragma HLS BIND_OP variable=product op=mul impl=fabric
#endif
                                add += product;
                            }
                            sum_t accumulated = sums[lane][p]+add;
#ifdef HAND_REALTIME
#pragma HLS BIND_OP variable=accumulated op=add impl=fabric
#endif
                            sums[lane][p] = accumulated;
                        }
                    }
#endif
                }
#ifdef HAND_BURST
                for (int lane = 0; lane < LANES; ++lane) {
                    if (oc0+lane >= outputs) continue;
                    int vh = oh-y0 < PW_HEIGHT ? oh-y0 : PW_HEIGHT;
                    int vw = ow-x0 < PW_WIDTH ? ow-x0 : PW_WIDTH;
                    for (int y = 0; y < vh; ++y) {
                        output.write_request((oc0+lane)*oh*ow+(y0+y)*ow+x0, vw);
pw_write_row:
                        for (int x = 0; x < vw; ++x) {
#pragma HLS PIPELINE II=1
                            output.write((long long)sums[lane][y*PW_WIDTH+x]);
                        }
                        output.write_response();
                    }
                }
#else
                for (int lane = 0; lane < LANES; ++lane)
                    for (int p = 0; p < PW_PIXELS; ++p) {
#pragma HLS PIPELINE II=1
                        int y = y0+p/PW_WIDTH, x = x0+p%PW_WIDTH;
                        if (oc0+lane < outputs && y < oh && x < ow)
                            output[(oc0+lane)*oh*ow+y*ow+x] = (long long)sums[lane][p];
                    }
#endif
            }
    }
}

#if defined(HAND_REALTIME) || defined(HAND_DW_CACHE)
#ifdef HAND_DW_WIDE
enum { DW_WIDTH = 16, DW_HEIGHT = 2, DW_PIXELS = DW_WIDTH*DW_HEIGHT,
       DW_PATCH_H = (DW_HEIGHT-1)*2+5, DW_PATCH_W = (DW_WIDTH-1)*2+5 };
#else
enum { DW_WIDTH = TILE, DW_HEIGHT = TILE, DW_PIXELS = PIXELS,
       DW_PATCH_H = PATCH, DW_PATCH_W = PATCH };
#endif
// Cache depthwise weights once per channel block.
static void depthwise(packed_t *input, weight_packed_t *weights, output_t &output,
                      ap_uint<9> height, ap_uint<9> width, int channels, ap_uint<3> kernel,
                      ap_uint<2> stride, ap_uint<3> top, ap_uint<3> left,
                      ap_uint<9> oh, ap_uint<9> ow) {
    weight_t wc[LANES][5][5];
    value_t patch[LANES][DW_PATCH_H][DW_PATCH_W];
    sum_t sums[LANES][DW_PIXELS];
#ifdef HAND_DW_CACHE
    const int taps = 1;
#else
    const int taps = 2;
#endif
#pragma HLS ARRAY_PARTITION variable=wc complete dim=1
#ifdef HAND_BANKED
#pragma HLS ARRAY_PARTITION variable=wc cyclic factor=2 dim=3
#pragma HLS BIND_STORAGE variable=wc type=ram_2p impl=lutram
#else
#pragma HLS ARRAY_PARTITION variable=wc complete dim=3
#endif
#pragma HLS ARRAY_PARTITION variable=patch complete dim=1
#ifdef HAND_BANKED
#pragma HLS ARRAY_PARTITION variable=patch cyclic factor=2 dim=3
#pragma HLS BIND_STORAGE variable=patch type=ram_2p impl=lutram
#else
#pragma HLS ARRAY_PARTITION variable=patch complete dim=3
#endif
#pragma HLS ARRAY_PARTITION variable=sums complete dim=1
    for (int c0 = 0; c0 < channels; c0 += LANES) {
        for (int ky = 0; ky < kernel; ++ky)
            for (int kx = 0; kx < kernel; ++kx) {
#pragma HLS PIPELINE II=1
                weight_packed_t word = weights[(ky*kernel+kx)*((channels+7)/8)+c0/8];
                for (int lane = 0; lane < LANES; ++lane) {
#pragma HLS UNROLL
                    wc[lane][ky][kx] = word.range((lane+1)*WEIGHT_BITS-1,lane*WEIGHT_BITS);
                }
            }
        for (int y0 = 0; y0 < oh; y0 += DW_HEIGHT)
            for (int x0 = 0; x0 < ow; x0 += DW_WIDTH) {
                int vh = oh-y0 < DW_HEIGHT ? oh-y0 : DW_HEIGHT;
                int vw = ow-x0 < DW_WIDTH ? ow-x0 : DW_WIDTH;
                int ph = (vh-1)*stride+kernel, pw = (vw-1)*stride+kernel;
                for (int y = 0; y < ph; ++y)
                    for (int x = 0; x < pw; ++x) {
#pragma HLS PIPELINE II=1
                        int iy = y0*stride-top+y, ix = x0*stride-left+x;
                        packed_t word = iy >= 0 && iy < height && ix >= 0 && ix < width ?
                            input[(iy*width+ix)*((channels+7)/8)+c0/8] : packed_t(0);
                        for (int lane = 0; lane < LANES; ++lane) {
#pragma HLS UNROLL
                            patch[lane][y][x] = word.range((lane+1)*VALUE_BITS-1,lane*VALUE_BITS);
                        }
                    }
                for (int p = 0; p < DW_PIXELS; ++p) {
#pragma HLS PIPELINE II=1
                    for (int lane = 0; lane < LANES; ++lane) {
#pragma HLS UNROLL
                        sums[lane][p] = 0;
                    }
                }
                for (int ky = 0; ky < kernel; ++ky)
                    for (int k0 = 0; k0 < kernel; k0 += taps)
dw_fast_mac:
                    for (int p = 0; p < DW_PIXELS; ++p) {
#pragma HLS PIPELINE II=1
                        for (int lane = 0; lane < LANES; ++lane) {
#pragma HLS UNROLL
                            sum_t add = 0;
                            for (int tap = 0; tap < taps; ++tap) {
#pragma HLS UNROLL
                                int kx = k0+tap;
                                if (kx < kernel) {
                                    ap_int<VALUE_BITS+WEIGHT_BITS> product = patch[lane][(p/DW_WIDTH)*stride+ky][(p%DW_WIDTH)*stride+kx]*wc[lane][ky][kx];
#ifdef PREOPT_FABRIC_MUL
#pragma HLS BIND_OP variable=product op=mul impl=fabric
#endif
                                    add += product;
                                }
                            }
                            sum_t accumulated = sums[lane][p]+add;
#pragma HLS BIND_OP variable=accumulated op=add impl=fabric
                            sums[lane][p] = accumulated;
                        }
                    }
                for (int lane = 0; lane < LANES; ++lane) {
#ifdef HAND_BURST
                    if (c0+lane >= channels) continue;
#endif
                    for (int y = 0; y < vh; ++y) {
#ifdef HAND_BURST
                        output.write_request((c0+lane)*oh*ow+(y0+y)*ow+x0, vw);
#endif
                        for (int x = 0; x < vw; ++x) {
#pragma HLS PIPELINE II=1
#ifdef HAND_BURST
                            output.write((long long)sums[lane][y*DW_WIDTH+x]);
#else
                            if (c0+lane < channels)
                                output[(c0+lane)*oh*ow+(y0+y)*ow+x0+x] = (long long)sums[lane][y*DW_WIDTH+x];
#endif
                        }
#ifdef HAND_BURST
                        output.write_response();
#endif
                    }
                }
            }
    }
}
#endif

// Shared regular/depthwise datapath. Integer feedback needs no FP32 sum banks.
static void spatial(packed_t *input, weight_packed_t *weights, output_t &output,
                    int height, int width, int channels, int outputs, int kernel,
                    int stride, int top, int left, int oh, int ow, int groups) {
    value_t patch[LANES][PATCH][PATCH];
    weight_t wc[LANES][SIMD][5][5];
    sum_t sums[LANES][PIXELS];
#pragma HLS ARRAY_PARTITION variable=patch complete dim=1
#pragma HLS ARRAY_PARTITION variable=wc complete dim=1
#pragma HLS ARRAY_PARTITION variable=wc complete dim=2
#pragma HLS BIND_STORAGE variable=wc type=ram_2p impl=lutram
#pragma HLS ARRAY_PARTITION variable=sums complete dim=1
    bool dw = groups == channels && channels == outputs;
    int icpg = dw ? 1 : channels;
    for (int oc0 = 0; oc0 < outputs; oc0 += LANES)
        for (int y0 = 0; y0 < oh; y0 += TILE)
            for (int x0 = 0; x0 < ow; x0 += TILE) {
                int vh = oh-y0 < TILE ? oh-y0 : TILE;
                int vw = ow-x0 < TILE ? ow-x0 : TILE;
                int ph = (vh-1)*stride+kernel, pw = (vw-1)*stride+kernel;
                for (int p = 0; p < PIXELS; ++p) {
#pragma HLS PIPELINE II=1
                    for (int lane = 0; lane < LANES; ++lane) {
#pragma HLS UNROLL
                        sums[lane][p] = 0;
                    }
                }
                for (int ic0 = 0; ic0 < icpg; ic0 += SIMD) {
                    for (int y = 0; y < ph; ++y)
                        for (int x = 0; x < pw; ++x) {
#pragma HLS PIPELINE II=1
                            int block = dw ? oc0/8 : ic0/8;
                            int iy = y0*stride-top+y, ix = x0*stride-left+x;
                            packed_t word = iy >= 0 && iy < height && ix >= 0 && ix < width ?
                                input[(iy*width+ix)*((channels+7)/8)+block] : packed_t(0);
                            for (int ci = 0; ci < LANES; ++ci) {
#pragma HLS UNROLL
                                patch[ci][y][x] = word.range(((dw ? ci : (ic0%8+ci)%8)+1)*VALUE_BITS-1,(dw ? ci : (ic0%8+ci)%8)*VALUE_BITS);
                            }
                        }
                    for (int lane = 0; lane < LANES; ++lane)
                        for (int ky = 0; ky < kernel; ++ky)
                            for (int kx = 0; kx < kernel; ++kx) {
#pragma HLS PIPELINE II=1
                                weight_packed_t word = oc0+lane < outputs ? weights[dw ?
                                    (ky*kernel+kx)*((outputs+7)/8)+oc0/8 :
                                    ((oc0+lane)*kernel*kernel+ky*kernel+kx)*((channels+7)/8)+ic0/8] : weight_packed_t(0);
                                for (int ci = 0; ci < SIMD; ++ci) {
#pragma HLS UNROLL
                                    wc[lane][ci][ky][kx] = dw ?
                                        (ci == 0 ? weight_t(word.range((lane+1)*WEIGHT_BITS-1,lane*WEIGHT_BITS)) : weight_t(0)) :
                                        weight_t(word.range((ic0%8+ci+1)*WEIGHT_BITS-1,(ic0%8+ci)*WEIGHT_BITS));
                                }
                            }
                    for (int ky = 0; ky < kernel; ++ky)
                        for (int kx = 0; kx < kernel; ++kx) {
spatial_mac:
                            for (int p = 0; p < PIXELS; ++p) {
#pragma HLS PIPELINE II=1
                                for (int lane = 0; lane < LANES; ++lane) {
#pragma HLS UNROLL
                                    sum_t add = 0;
                                    for (int ci = 0; ci < SIMD; ++ci) {
#pragma HLS UNROLL
                                        value_t pixel = patch[dw ? lane : ci][(p/TILE)*stride+ky][(p%TILE)*stride+kx];
                                        ap_int<VALUE_BITS+WEIGHT_BITS> product = pixel*wc[lane][ci][ky][kx];
#if defined(HAND_REALTIME) || defined(PREOPT_FABRIC_MUL) || defined(PREOPT_SPATIAL_FABRIC)
#pragma HLS BIND_OP variable=product op=mul impl=fabric
#endif
                                        add += product;
                                    }
                                    sum_t accumulated = sums[lane][p]+add;
#ifdef HAND_REALTIME
#pragma HLS BIND_OP variable=accumulated op=add impl=fabric
#endif
                                    sums[lane][p] = accumulated;
                                }
                            }
                        }
                }
                for (int lane = 0; lane < LANES; ++lane) {
#ifdef HAND_BURST
                    if (oc0+lane >= outputs) continue;
#endif
                    for (int y = 0; y < vh; ++y) {
#ifdef HAND_BURST
                        output.write_request((oc0+lane)*oh*ow+(y0+y)*ow+x0, vw);
#endif
                        for (int x = 0; x < vw; ++x) {
#pragma HLS PIPELINE II=1
#ifdef HAND_BURST
                            output.write((long long)sums[lane][y*TILE+x]);
#else
                            if (oc0+lane < outputs)
                                output[(oc0+lane)*oh*ow+(y0+y)*ow+x0+x] = (long long)sums[lane][y*TILE+x];
#endif
                        }
#ifdef HAND_BURST
                        output.write_response();
#endif
                    }
                }
            }
}

void hand_conv_preopt(packed_t *input, weight_packed_t *weights, output_t output,
                   int height, int width, int channels, int outputs, int kernel,
                   int stride, int pad_top, int pad_left, int out_h, int out_w, int groups) {
#pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem max_read_burst_length=64 depth=4096
#pragma HLS INTERFACE m_axi port=weights offset=slave bundle=gmem max_read_burst_length=64 depth=4096
#ifdef HAND_BURST
// Manual bursts count bus beats. Keep 64-bit sums off the 128-bit operand bus.
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=results max_write_burst_length=64 max_widen_bitwidth=64 depth=4096
#else
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem max_write_burst_length=64 depth=4096
#endif
#pragma HLS INTERFACE s_axilite port=input bundle=control
#pragma HLS INTERFACE s_axilite port=weights bundle=control
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
    // The raw YOLO DFL head uses a 4x2100 pointwise input at 320 resolution.
    bool pointwise_head = height <= 4 && width <= 4096 && out_h == height && out_w == width &&
                          kernel == 1 && stride == 1 && pad_top == 0 && pad_left == 0 && groups == 1;
    if (height <= 0 || width <= 0 || height > 320 || (width > 320 && !pointwise_head) || channels <= 0 || channels > 1024 ||
        outputs <= 0 || outputs > 1024 || (kernel != 1 && kernel != 3 && kernel != 5) ||
        (stride != 1 && stride != 2) || out_h <= 0 || out_w <= 0 || out_h > 328 || (out_w > 328 && !pointwise_head) ||
        pad_top < 0 || pad_top > 4 || pad_left < 0 || pad_left > 4 ||
        (groups != 1 && !(groups == channels && outputs == channels))) return;
    if (kernel == 1 && groups == 1)
        pointwise(input,weights,output,height,width,channels,outputs,stride,pad_top,pad_left,out_h,out_w);
    else
#if defined(HAND_REALTIME) || defined(HAND_DW_CACHE)
    if (groups == channels && outputs == channels)
        depthwise(input,weights,output,height,width,channels,kernel,stride,pad_top,pad_left,out_h,out_w);
    else
#endif
        spatial(input,weights,output,height,width,channels,outputs,kernel,stride,pad_top,pad_left,out_h,out_w,groups);
}
