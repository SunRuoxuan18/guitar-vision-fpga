#include <ap_int.h>
#include <hls_burst_maxi.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

#ifndef PREOPT_VALUE_BITS
#define PREOPT_VALUE_BITS 8
#endif
enum { BITS = PREOPT_VALUE_BITS };
typedef ap_int<BITS> value_t;
typedef ap_uint<8*BITS> packed_t;
typedef hls::burst_maxi<long long> output_t;
void hand_conv_preopt(packed_t*,packed_t*,output_t,int,int,int,int,int,int,int,int,int,int,int);

static bool check(int h, int w, int c, int o, int k, int s, int top, int left,
                  int bottom, int right, int groups) {
    const int oh = (h+top+bottom-k)/s+1, ow = (w+left+right-k)/s+1;
    const int per_group = c/groups;
    std::vector<value_t> input(c*h*w), weights(o*per_group*k*k);
    const long long guard = 0x123456789ABCDEFLL;
    std::vector<long long> output(std::max(o*oh*ow+16,4096),guard);
    for (unsigned i=0; i<input.size(); ++i) input[i] = i*1789+13;
    for (unsigned i=0; i<weights.size(); ++i) weights[i] = i*2347+17;
    const int blocks = (c+7)/8;
    std::vector<packed_t> x(std::max(h*w*blocks,4096),0);
    std::vector<packed_t> weight_words(std::max(groups==1 ? o*k*k*blocks : k*k*((o+7)/8),4096),0);
    for (int ic=0; ic<c; ++ic)
        for (int p=0; p<h*w; ++p)
            x[p*blocks+ic/8].range((ic%8+1)*BITS-1,ic%8*BITS) = input[ic*h*w+p];
    for (int oc=0; oc<o; ++oc)
        for (int ic=0; ic<per_group; ++ic)
            for (int p=0; p<k*k; ++p) {
                int index = groups==1 ? (oc*k*k+p)*blocks+ic/8 : p*((o+7)/8)+oc/8;
                int lane = groups==1 ? ic%8 : oc%8;
                weight_words[index].range((lane+1)*BITS-1,lane*BITS) = weights[(oc*per_group+ic)*k*k+p];
            }
    hand_conv_preopt(x.data(),weight_words.data(),output.data(),h,w,c,o,k,s,top,left,oh,ow,groups);
    for (int oc=0; oc<o; ++oc)
        for (int y=0; y<oh; ++y)
            for (int z=0; z<ow; ++z) {
                long long expected = 0;
                for (int ic=0; ic<per_group; ++ic)
                    for (int ky=0; ky<k; ++ky)
                        for (int kx=0; kx<k; ++kx) {
                            int iy=y*s-top+ky, ix=z*s-left+kx;
                            int channel=(oc/(o/groups))*per_group+ic;
                            if (iy>=0 && iy<h && ix>=0 && ix<w)
                                expected += (long long)input[(channel*h+iy)*w+ix].to_int()*
                                            weights[((oc*per_group+ic)*k+ky)*k+kx].to_int();
                        }
                if (output[(oc*oh+y)*ow+z] != expected) {
                    std::cerr << "Mismatch " << h << "x" << w << " k=" << k << " groups=" << groups << "\n";
                    return false;
                }
            }
    for (unsigned i=o*oh*ow; i<output.size(); ++i)
        if (output[i] != guard) { std::cerr << "Output guard overwritten\n"; return false; }
    std::cout << "PASS bits=" << BITS << " " << h << "x" << w << " " << c << "->" << o
              << " k=" << k << " g=" << groups << "\n";
    return true;
}

int main(int argc, char**) {
    bool ok=true;
    ok &= check(7,9,3,11,3,2,0,1,1,0,1);
    ok &= check(9,7,13,19,1,1,0,0,0,0,1);
    ok &= check(9,7,13,19,1,2,1,0,0,1,1);
    ok &= check(7,9,11,11,3,1,1,1,1,1,11);
    ok &= check(11,13,9,9,5,2,1,2,2,1,9);
    if (argc>1) return ok ? 0 : 1;
    ok &= check(1,1,1024,63,1,1,0,0,0,0,1);
    ok &= check(1,1,9,9,1,1,0,0,0,0,9);
    ok &= check(15,13,8,17,5,1,2,2,2,2,1);
    ok &= check(5,37,13,19,1,1,0,0,0,0,1);
    ok &= check(7,39,11,17,1,2,1,1,0,0,1);
    ok &= check(320,320,3,16,3,2,1,1,1,1,1);
    ok &= check(4,2100,16,1,1,1,0,0,0,0,1);
    // Max supported reduction length: verify 32-bit accumulation range.
    ok &= check(5,5,1024,1,5,1,0,0,0,0,1);
    return ok ? 0 : 1;
}
