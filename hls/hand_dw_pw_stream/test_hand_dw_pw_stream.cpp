#include <ap_int.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
typedef ap_uint<128> packed_t;
void hand_dw_pw_stream(packed_t*,packed_t*,packed_t*,float*,long long*,int,int,int,int,int,int,int,int,int,int);

static bool check(int h,int w,int c,int o,int k,int stride,int top,int left,int bottom,int right,bool zero) {
    int oh=(h+top+bottom-k)/stride+1, ow=(w+left+right-k)/stride+1, blocks=(c+7)/8;
    // Co-simulation marshals the declared AXI depths, including unused tails.
    std::vector<packed_t> input(18432),dw(450),pw(864);
    std::vector<short> x(h*w*c),a(k*k*c),b(o*c);
    std::vector<float> params(288);
    for (int i=0;i<h*w*c;++i) x[i]=zero ? 0 : (i*17+3)%127-63;
    for (int i=0;i<k*k*c;++i) a[i]=(i*7+1)%43-21;
    for (int i=0;i<o*c;++i) b[i]=(i*11+7)%53-26;
    for (int ch=0;ch<c;++ch) { params[ch]=.00017f*(1+ch%3);params[c+ch]=zero ? 0.f : (ch%5-2)*.4f; }
    for (int p=0;p<h*w;++p) for(int ch=0;ch<c;++ch)
        input[p*blocks+ch/8].range((ch%8)*16+15,(ch%8)*16)=(ap_uint<16>)x[p*c+ch];
    for (int t=0;t<k*k;++t) for(int ch=0;ch<c;++ch)
        dw[t*blocks+ch/8].range((ch%8)*16+15,(ch%8)*16)=(ap_uint<16>)a[t*c+ch];
    for (int oc=0;oc<o;++oc) for(int ch=0;ch<c;++ch)
        pw[oc*blocks+ch/8].range((ch%8)*16+15,(ch%8)*16)=(ap_uint<16>)b[oc*c+ch];
    std::vector<long long> output(49152,-77),gold(o*oh*ow);
    for (int y=0;y<oh;++y) for(int xx=0;xx<ow;++xx) for(int ch=0;ch<c;++ch) {
        long long sum=0;
        for (int ky=0;ky<k;++ky) for(int kx=0;kx<k;++kx) {
            int iy=y*stride-top+ky,ix=xx*stride-left+kx;
            if(iy>=0 && iy<h && ix>=0 && ix<w) sum+=(long long)x[(iy*w+ix)*c+ch]*a[(ky*k+kx)*c+ch];
        }
        float value=(float)sum*params[ch]; value+=params[c+ch];
        value=std::min(6.f,std::max(0.f,value));
        int q=std::min(32767,(int)(value*(32767.f/6.f)+.5f));
        for(int oc=0;oc<o;++oc) gold[oc*oh*ow+y*ow+xx]+=(long long)q*b[oc*c+ch];
    }
    hand_dw_pw_stream(input.data(),dw.data(),pw.data(),params.data(),output.data(),h,w,c,o,k,stride,top,left,oh,ow);
    if(!std::equal(gold.begin(),gold.end(),output.begin())) {
        for(size_t i=0;i<gold.size();++i) if(output[i]!=gold[i]) {
            std::cerr<<"Mismatch "<<i<<": "<<output[i]<<" != "<<gold[i]<<"\n";break;
        }
        return false;
    }
    for(size_t i=gold.size();i<output.size();++i) if(output[i]!=-77) return false;
    std::cout<<"PASS "<<h<<"x"<<w<<" C"<<c<<" O"<<o<<" K"<<k<<" stride"<<stride<<" zero"<<zero<<"\n";
    return true;
}
int main() {
    bool ok=true;
    ok &= check(7,9,13,17,3,1,1,1,1,1,false);
    ok &= check(8,10,16,8,5,2,1,1,2,2,false);
    ok &= check(5,7,8,9,3,1,1,1,1,1,true);
    ok &= check(28,28,144,24,5,1,2,2,2,2,false);
    // Consecutive invocation and bottom/right padding expose retained-row bugs.
    ok &= check(4,6,9,7,5,1,2,2,2,2,false);
    std::vector<packed_t> input(18432),dw(450),pw(864);
    std::vector<float> params(288);
    std::vector<long long> output(49152,-99);
    hand_dw_pw_stream(input.data(),dw.data(),pw.data(),params.data(),output.data(),33,1,1,1,3,1,1,1,1,1);
    ok &= std::all_of(output.begin(),output.end(),[](long long v){return v==-99;});
    return ok ? 0 : 1;
}
