#include "hand_conv.h"
#include <cmath>
#include <cstdio>
#include <vector>

static unsigned state = 1;
static float sample() { state = state * 1664525u + 1013904223u; return ((state >> 8) & 0xffff) / 32768.0f - 1.0f; }

static void reference(const std::vector<float>& input, const std::vector<float>& weights, const std::vector<float>& bias,
                      std::vector<float>& output, int h, int w, int c, int o, int k, int s, int pt, int pl, int oh, int ow, int g) {
    int icpg = c / g, ocpg = o / g;
    for (int oc = 0; oc < o; ++oc) for (int y = 0; y < oh; ++y) for (int x = 0; x < ow; ++x) {
        float sum = bias[oc]; int group = oc / ocpg;
        for (int ic = 0; ic < icpg; ++ic) for (int ky = 0; ky < k; ++ky) for (int kx = 0; kx < k; ++kx) {
            int iy = y * s - pt + ky, ix = x * s - pl + kx;
            if (iy >= 0 && iy < h && ix >= 0 && ix < w)
                sum += input[(group * icpg + ic) * h * w + iy * w + ix] * weights[(oc * icpg + ic) * k * k + ky * k + kx];
        }
        output[oc * oh * ow + y * ow + x] = sum;
    }
}

static int run_case(const char *name, int h, int w, int c, int o, int k, int s, int pt, int pl, int oh, int ow, int g) {
    std::vector<float> in(h*w*c), wt(o*(c/g)*k*k), bias(o), got(o*oh*ow, 99.0f), ref(o*oh*ow);
    for (size_t i=0;i<in.size();++i) in[i]=sample(); for(size_t i=0;i<wt.size();++i) wt[i]=sample(); for(size_t i=0;i<bias.size();++i) bias[i]=sample();
    reference(in,wt,bias,ref,h,w,c,o,k,s,pt,pl,oh,ow,g);
    hand_conv(in.data(),wt.data(),bias.data(),got.data(),h,w,c,o,k,s,pt,pl,oh,ow,g);
    float max_abs=0.0f; int bad=-1;
    for(size_t i=0;i<got.size();++i) { float d=std::fabs(got[i]-ref[i]); if(d>max_abs) max_abs=d; if((!std::isfinite(got[i]) || d>2e-4f+2e-5f*std::fabs(ref[i])) && bad<0) bad=(int)i; }
    std::printf("%s: max_abs=%g %s\n",name,max_abs,bad<0?"PASS":"FAIL");
    if(bad>=0) { std::printf(" index=%d got=%g expected=%g\n",bad,got[bad],ref[bad]); return 1; }
    return 0;
}

int main() {
    int fail=0;
    fail+=run_case("regular_3x3_stride2_asymmetric",9,11,7,10,3,2,1,2,5,6,1);
    fail+=run_case("pointwise_many_channels_tails",5,7,19,13,1,1,0,0,5,7,1);
    fail+=run_case("depthwise_3x3_stride1",6,9,9,9,3,1,1,2,7,9,9);
    fail+=run_case("depthwise_5x5_stride2",10,11,7,7,5,2,2,1,5,5,7);
    fail+=run_case("grouped_3x3_borders",7,8,12,16,3,1,2,1,9,8,4);
    fail+=run_case("pointwise_dense_1024",1,1,1024,21,1,1,0,0,1,1,1);
    fail+=run_case("pointwise_stride2_padded_tiles",19,23,33,17,1,2,1,2,11,14,1);
    fail+=run_case("pointwise_grouped_tails",9,11,18,30,1,1,0,0,9,11,3);
    fail+=run_case("depthwise_single_pixel",1,1,3,3,5,1,2,2,1,1,3);
    return fail ? 1 : 0;
}
