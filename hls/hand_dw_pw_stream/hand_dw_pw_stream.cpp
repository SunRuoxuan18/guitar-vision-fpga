#include <ap_int.h>
#include <hls_stream.h>

typedef ap_int<16> value_t;
typedef ap_uint<128> packed_t;
enum { MAX_C = 144, MAX_O = 24, ROW = 4096, LANES = 2 };

static int address_product(int a, int b) {
#pragma HLS INLINE off
    ap_uint<16> left=a, right=b;
    ap_uint<16> result=left*right;
#pragma HLS BIND_OP variable=result op=mul impl=fabric
    return result;
}

static value_t activate(ap_int<48> sum, float scale, float bias) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II=1
    float product=(float)sum*scale;
#pragma HLS BIND_OP variable=product op=fmul impl=meddsp
    float value=product+bias;
#pragma HLS BIND_OP variable=value op=fadd impl=fabric
    float clipped=value<0.f ? 0.f : (value>6.f ? 6.f : value);
    float scaled=clipped*(32767.f/6.f);
#pragma HLS BIND_OP variable=scaled op=fmul impl=fabric
    float rounded=scaled+.5f;
#pragma HLS BIND_OP variable=rounded op=fadd impl=fabric
    int q=(int)rounded;
    return value_t(q>32767 ? 32767 : q);
}

// Each input row is read once. The ring retains all real rows needed by
// the next output window; bounds checks explicitly produce bottom/right pad.
static void depthwise(packed_t *input, packed_t *weights,
                      hls::stream<ap_int<48> > &sums, int h, int w, int c,
                      int k, int stride, int top, int left, int oh, int ow) {
#pragma HLS INLINE off
    packed_t rows[5][512];
    packed_t kernel[25][18];
#pragma HLS BIND_STORAGE variable=rows type=ram_2p impl=bram
#pragma HLS ARRAY_PARTITION variable=rows complete dim=1
#pragma HLS BIND_STORAGE variable=kernel type=ram_2p impl=bram
    const int blocks = (c+7)/8;
    for (int tap=0; tap<address_product(k,k); ++tap)
        for (int block=0; block<blocks; ++block) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_FLATTEN off
            kernel[tap][block]=weights[address_product(tap,blocks)+block];
        }
    int loaded=-1;
    for (int y=0; y<oh; ++y) {
        int needed=address_product(y,stride)-top+k-1;
        if (needed>=h) needed=h-1;
        while (loaded<needed) {
            ++loaded;
            for (int x=0; x<w; ++x)
                for (int block=0; block<blocks; ++block) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_FLATTEN off
                    rows[ap_uint<6>(loaded)%ap_uint<3>(5)][address_product(x,blocks)+block]=input[address_product(address_product(loaded,w)+x,blocks)+block];
                }
        }
        for (int x=0; x<ow; ++x)
            for (int ch=0; ch<c; ch+=2) {
                ap_int<48> sum[2]={0,0};
#pragma HLS ARRAY_PARTITION variable=sum complete
                int ky=0,kx=0;
                for (int tap=0; tap<address_product(k,k); ++tap) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_FLATTEN off
                        int iy=address_product(y,stride)-top+ky, ix=address_product(x,stride)-left+kx;
                        packed_t word=(iy>=0 && iy<h && ix>=0 && ix<w) ? rows[ap_uint<6>(iy)%ap_uint<3>(5)][address_product(ix,blocks)+ch/8] : packed_t(0);
                        packed_t weight=kernel[tap][ch/8];
                        for(int lane=0;lane<2;++lane) {
#pragma HLS UNROLL
                        int lo=(ch%8+lane)*16;
                        value_t v=word.range(lo+15,lo), wv=weight.range(lo+15,lo);
                        ap_int<32> product=v*wv;
#pragma HLS BIND_OP variable=product op=mul impl=dsp
                        sum[lane]+=product;
                        }
                        if (kx+1==k) {kx=0;++ky;} else ++kx;
                    }
                sums.write(sum[0]);
                if(ch+1<c) sums.write(sum[1]);
            }
    }
}

static void activation(const float scales[MAX_C], const float biases[MAX_C], hls::stream<ap_int<48> > &sums,
                       hls::stream<value_t> &features, int c, int elements) {
#pragma HLS INLINE off
    int ch=0;
    for (int i=0;i<elements;++i) {
#pragma HLS PIPELINE II=1
        // Separate consumer overlaps conversion with the next DW window.
        features.write(activate(sums.read(),scales[ch],biases[ch]));
        ch=ch+1==c ? 0 : ch+1;
    }
}

static void pointwise(packed_t *weights, hls::stream<value_t> &features,
                      long long *output, int c, int outputs, int pixels) {
#pragma HLS INLINE off
    value_t cached[MAX_O][MAX_C];
    value_t vector[MAX_C];
#pragma HLS ARRAY_PARTITION variable=cached cyclic factor=LANES dim=1
#pragma HLS BIND_STORAGE variable=cached type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=vector type=ram_2p impl=lutram
    const int blocks=(c+7)/8;
    for (int o=0; o<outputs; ++o)
        for (int block=0; block<blocks; ++block) {
            packed_t word=weights[address_product(o,blocks)+block];
            for (int lane=0; lane<8; ++lane) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_FLATTEN off
                int ch=block*8+lane;
                if (ch<c) cached[o][ch]=word.range(lane*16+15,lane*16);
            }
        }
    for (int pixel=0; pixel<pixels; ++pixel) {
        for (int ch=0; ch<c; ++ch) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_FLATTEN off
            vector[ch]=features.read();
        }
        for (int o=0; o<outputs; o+=LANES) {
            ap_int<48> sums[LANES];
#pragma HLS ARRAY_PARTITION variable=sums complete
            for (int lane=0; lane<LANES; ++lane) {
#pragma HLS UNROLL
                sums[lane]=0;
            }
            for (int ch=0; ch<c; ++ch) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_FLATTEN off
                for (int lane=0; lane<LANES; ++lane) {
#pragma HLS UNROLL
                    value_t weight=o+lane<outputs ? cached[o+lane][ch] : value_t(0);
                    ap_int<32> product=vector[ch]*weight;
#pragma HLS BIND_OP variable=product op=mul impl=dsp
                    sums[lane]+=product;
                }
            }
            for (int lane=0; lane<LANES; ++lane) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_FLATTEN off
                if (o+lane<outputs) output[address_product(o+lane,pixels)+pixel]=(long long)sums[lane];
            }
        }
    }
}

static void fused(packed_t *input, packed_t *dw_weights, packed_t *pw_weights,
                  const float scales[MAX_C], const float biases[MAX_C], long long *output, int h, int w, int c, int o,
                  int k, int stride, int top, int left, int oh, int ow, int pixels, int elements) {
#pragma HLS DATAFLOW
    hls::stream<value_t> features("features");
    hls::stream<ap_int<48> > sums("sums");
#pragma HLS STREAM variable=features depth=288
#pragma HLS STREAM variable=sums depth=16
    depthwise(input,dw_weights,sums,h,w,c,k,stride,top,left,oh,ow);
    activation(scales,biases,sums,features,c,elements);
    pointwise(pw_weights,features,output,c,o,pixels);
}

void hand_dw_pw_stream(packed_t *input, packed_t *dw_weights, packed_t *pw_weights,
                       float *params, long long *output, int height, int width,
                       int channels, int outputs, int kernel, int stride,
                       int pad_top, int pad_left, int out_h, int out_w) {
#pragma HLS INTERFACE m_axi port=input offset=slave bundle=operand depth=18432 max_read_burst_length=16 num_read_outstanding=1
#pragma HLS INTERFACE m_axi port=dw_weights offset=slave bundle=operand depth=450 max_read_burst_length=16 num_read_outstanding=1
#pragma HLS INTERFACE m_axi port=params offset=slave bundle=operand depth=288 max_read_burst_length=16 num_read_outstanding=1
#pragma HLS INTERFACE m_axi port=pw_weights offset=slave bundle=pointweights depth=864 max_read_burst_length=16 num_read_outstanding=1
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=results depth=49152 max_write_burst_length=16 max_widen_bitwidth=64 num_write_outstanding=1
#pragma HLS INTERFACE s_axilite port=input bundle=control
#pragma HLS INTERFACE s_axilite port=dw_weights bundle=control
#pragma HLS INTERFACE s_axilite port=pw_weights bundle=control
#pragma HLS INTERFACE s_axilite port=params bundle=control
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
#pragma HLS INTERFACE s_axilite port=return bundle=control
    if (height<1 || height>32 || width<1 || width>32 || channels<1 || channels>MAX_C ||
        address_product(width,channels)>ROW || address_product(width,(channels+7)/8)>512 || outputs<1 || outputs>MAX_O ||
        (kernel!=3 && kernel!=5) || (stride!=1 && stride!=2) ||
        pad_top<0 || pad_top>2 || pad_left<0 || pad_left>2 ||
        out_h<1 || out_h>32 || out_w<1 || out_w>32 ||
        address_product(out_h-1,stride)-pad_top>=height || address_product(out_w-1,stride)-pad_left>=width) return;
    float scales[MAX_C],biases[MAX_C];
#pragma HLS BIND_STORAGE variable=scales type=ram_2p impl=lutram
#pragma HLS BIND_STORAGE variable=biases type=ram_2p impl=lutram
    for (int ch=0;ch<channels;++ch) {
#pragma HLS PIPELINE II=1
        scales[ch]=params[ch];biases[ch]=params[channels+ch];
    }
    int pixels=address_product(out_h,out_w);
    ap_uint<18> elements=ap_uint<11>(pixels)*ap_uint<8>(channels);
#pragma HLS BIND_OP variable=elements op=mul impl=fabric
    fused(input,dw_weights,pw_weights,scales,biases,output,height,width,channels,outputs,
          kernel,stride,pad_top,pad_left,out_h,out_w,pixels,elements);
}
