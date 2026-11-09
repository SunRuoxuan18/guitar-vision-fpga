#include "lk_sparse.h"
#include <cmath>
#include <iostream>
#include <vector>

static double texture(double x, double y) {
    return 125 + 30*std::sin(x*.23) + 29*std::cos(y*.27)
        + 23*std::sin(x*.13+y*.17);
}
int main() {
    const unsigned w=320, h=200, n=48;
    // Co-simulation marshals the declared AXI depths, including unused tails.
    std::vector<ap_uint<8> > a(64000), b(64000);
    ap_uint<32> points[64] = {}; ap_uint<64> result[64] = {};
    for (unsigned p=0;p<n;++p) points[p]=(25+p%8*38) | ((25+p/8*28)<<16);
    const double shifts[][2]={{0,0},{.5,0},{0,-.5},{.5,.5},{-.5,-.5},{1,0},{0,1}};
    for (auto &shift:shifts) {
        for (unsigned y=0;y<h;++y) for(unsigned x=0;x<w;++x) {
            a[y*w+x]=(unsigned)std::lround(texture(x,y));
            b[y*w+x]=(unsigned)std::lround(texture(x-shift[0],y-shift[1]));
        }
        lk_sparse(a.data(),b.data(),points,result,w,h,n);
        for (unsigned p=0;p<n;++p) {
            double dx=(ap_int<16>)result[p].range(15,0)/64.0;
            double dy=(ap_int<16>)result[p].range(31,16)/64.0;
            if(!result[p][32] || std::hypot(dx-shift[0],dy-shift[1])>.22) {
                std::cerr<<"shift failure "<<shift[0]<<","<<shift[1]<<" p="<<p
                    <<" got "<<dx<<","<<dy<<std::endl; return 1;
            }
        }
    }
    for(unsigned i=0;i<w*h;++i) a[i]=b[i]=100;
    lk_sparse(a.data(),b.data(),points,result,w,h,n);
    for(unsigned p=0;p<n;++p) if(result[p][32]) return 2;
    points[0]=1 | (1<<16);
    lk_sparse(a.data(),b.data(),points,result,w,h,n);
    if(result[0]!=0) return 3;
    // Horizontal-only edges cannot constrain both components of motion.
    for(unsigned y=0;y<h;++y) for(unsigned x=0;x<w;++x) a[y*w+x]=b[y*w+x]=y*2;
    lk_sparse(a.data(),b.data(),points,result,w,h,n);
    for(unsigned p=0;p<n;++p) if(result[p][32]) return 4;
    std::cout<<"LK_HLS_TEST_PASS: 7 known translations, blank, border, aperture"<<std::endl;
    return 0;
}
