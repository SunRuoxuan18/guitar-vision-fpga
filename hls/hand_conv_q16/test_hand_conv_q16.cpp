#include <ap_int.h>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <algorithm>
#ifdef HAND_BURST
#include <hls_burst_maxi.h>
typedef hls::burst_maxi<long long> output_t;
#else
typedef long long *output_t;
#endif
void hand_conv_q16(ap_uint<128>*,ap_uint<128>*,output_t,int,int,int,int,int,int,int,int,int,int,int);

static bool check(int h,int w,int c,int o,int k,int s,int t,int l,int b,int r,int g) {
    int oh=(h+t+b-k)/s+1, ow=(w+l+r-k)/s+1, icpg=c/g;
    std::vector<ap_int<16> > x(c*h*w), weights(o*icpg*k*k);
    const long long guard = 0x123456789ABCDEFLL;
    std::vector<long long> out(std::max(o*oh*ow+16,4096),guard);
    for (unsigned i=0;i<x.size();++i) x[i]=(i*1789+13)%65535-32767;
    for (unsigned i=0;i<weights.size();++i) weights[i]=(i*2347+17)%65535-32767;
    // Match the fixed co-simulation capture depths; runtime dimensions still
    // determine every accelerator read and write.
    std::vector<ap_uint<128> > packed_x(std::max(h*w*((c+7)/8),4096),0);
    std::vector<ap_uint<128> > packed_w(std::max(g==1 ? o*k*k*((c+7)/8) : k*k*((o+7)/8),4096),0);
    for(int ic=0;ic<c;++ic) for(int p=0;p<h*w;++p)
        packed_x[p*((c+7)/8)+ic/8].range(ic%8*16+15,ic%8*16)=x[ic*h*w+p];
    for(int oc=0;oc<o;++oc) for(int ic=0;ic<icpg;++ic) for(int p=0;p<k*k;++p) {
        int index=g==1 ? (oc*k*k+p)*((c+7)/8)+ic/8 : p*((o+7)/8)+oc/8;
        int lane=g==1 ? ic%8 : oc%8;
        packed_w[index].range(lane*16+15,lane*16)=weights[(oc*icpg+ic)*k*k+p];
    }
    hand_conv_q16(packed_x.data(),packed_w.data(),out.data(),h,w,c,o,k,s,t,l,oh,ow,g);
    for (int oc=0;oc<o;++oc)
        for(int y=0;y<oh;++y)
            for(int z=0;z<ow;++z) {
                long long expected=0;
                for(int ic=0;ic<icpg;++ic)
                    for(int ky=0;ky<k;++ky)
                        for(int kx=0;kx<k;++kx) {
                            int iy=y*s-t+ky,ix=z*s-l+kx;
                            if(iy>=0&&iy<h&&ix>=0&&ix<w)
                                expected+=(long long)x[((oc/(o/g))*icpg+ic)*h*w+iy*w+ix].to_int()*
                                    weights[((oc*icpg+ic)*k+ky)*k+kx].to_int();
                        }
                if(out[oc*oh*ow+y*ow+z]!=expected) {
                    std::cerr<<"Mismatch k="<<k<<" groups="<<g<<" oc="<<oc<<" "<<out[oc*oh*ow+y*ow+z]<<" != "<<expected<<"\n";
                    return false;
                }
            }
    for (unsigned i=o*oh*ow;i<out.size();++i) {
        if (out[i] != guard) { std::cerr<<"Output guard overwritten at "<<i<<"\n"; return false; }
    }
    std::cout<<"PASS "<<h<<"x"<<w<<" "<<c<<"->"<<o<<" k="<<k<<" g="<<g<<"\n";
    return true;
}
int main() {
    bool ok=true;
    ok&=check(7,9,3,11,3,2,0,1,1,0,1);
    ok&=check(9,7,13,19,1,1,0,0,0,0,1);
    ok&=check(9,7,13,19,1,2,1,0,0,1,1);
    ok&=check(7,9,11,11,3,1,1,1,1,1,11);
    ok&=check(11,13,9,9,5,2,1,2,2,1,9);
    ok&=check(1,1,1024,63,1,1,0,0,0,0,1);
    ok&=check(1,1,9,9,1,1,0,0,0,0,9);
    ok&=check(15,13,8,17,5,1,2,2,2,2,1);
    ok&=check(5,37,13,19,1,1,0,0,0,0,1);
    ok&=check(7,39,11,17,1,2,1,1,0,0,1);
#ifdef HAND_EXPERIMENTAL_YOLO320
    ok&=check(320,320,3,16,3,2,1,1,1,1,1);
#endif
    return ok?0:1;
}
