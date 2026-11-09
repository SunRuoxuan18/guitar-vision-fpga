#include "roi_projection.h"
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

static int reflect101_ref(int v, int n) { return v < 0 ? -v : (v >= n ? 2*n-v-2 : v); }
static void reference(const std::vector<ap_uint<8> >& im, const float h[9], std::vector<uint32_t>& out) {
    std::vector<unsigned char> raw(ROI_PIXELS), blur(ROI_PIXELS);
    for (int y=0;y<ROI_HEIGHT;++y) for(int x=0;x<ROI_WIDTH;++x) {
        float d=h[6]*x+h[7]*y+h[8]; bool valid=d!=0.0f;
        float sx=valid?(h[0]*x+h[1]*y+h[2])/d:0, sy=valid?(h[3]*x+h[4]*y+h[5])/d:0;
        int ix=(int)std::floor(sx), iy=(int)std::floor(sy); float fx=sx-ix, fy=sy-iy;
        auto get=[&](int px,int py){return px<0||px>=ROI_SOURCE_WIDTH||py<0||py>=ROI_SOURCE_HEIGHT?0.0f:(float)im[py*ROI_SOURCE_WIDTH+px];};
        int v=valid?(int)((1-fy)*((1-fx)*get(ix,iy)+fx*get(ix+1,iy))+fy*((1-fx)*get(ix,iy+1)+fx*get(ix+1,iy+1))+.5f):0;
        raw[y*ROI_WIDTH+x]=(unsigned char)(v<0?0:(v>255?255:v));
    }
    for(int y=0;y<ROI_HEIGHT;++y) for(int x=0;x<ROI_WIDTH;++x) {
        auto p=[&](int px,int py){return (int)raw[reflect101_ref(py,ROI_HEIGHT)*ROI_WIDTH+reflect101_ref(px,ROI_WIDTH)];};
        blur[y*ROI_WIDTH+x]=(unsigned char)((p(x-1,y-1)+2*p(x,y-1)+p(x+1,y-1)+2*p(x-1,y)+4*p(x,y)+2*p(x+1,y)+p(x-1,y+1)+2*p(x,y+1)+p(x+1,y+1)+8)>>4);
    }
    for(int y=0;y<ROI_HEIGHT;++y) for(int x=0;x<ROI_WIDTH;++x) {
        auto p=[&](int px,int py){return (int)blur[reflect101_ref(py,ROI_HEIGHT)*ROI_WIDTH+reflect101_ref(px,ROI_WIDTH)];};
        int gx=p(x+1,y-1)+2*p(x+1,y)+p(x+1,y+1)-p(x-1,y-1)-2*p(x-1,y)-p(x-1,y+1);
        int gy=p(x-1,y+1)+2*p(x,y+1)+p(x+1,y+1)-p(x-1,y-1)-2*p(x,y-1)-p(x+1,y-1);
        out[y*ROI_WIDTH+x]=(uint32_t)raw[y*ROI_WIDTH+x]|((uint32_t)std::abs(gx)<<8)|((uint32_t)std::abs(gy)<<19);
    }
}
static bool check(const char* tag,const std::vector<ap_uint<8> >& im,const float h[9]) {
    std::vector<ap_uint<32> > got(ROI_PIXELS); std::vector<uint32_t> expected(ROI_PIXELS); roi_projection(im.data(),h,got.data()); reference(im,h,expected);
    for(int i=0;i<ROI_PIXELS;++i) if((uint32_t)got[i]!=expected[i]) { std::cerr<<tag<<" mismatch at "<<i<<" got="<<(uint32_t)got[i]<<" expected="<<expected[i]<<'\n'; return false; }
    return true;
}
int main() {
    std::vector<ap_uint<8> > image(ROI_SOURCE_WIDTH*ROI_SOURCE_HEIGHT);
    for(int y=0;y<ROI_SOURCE_HEIGHT;++y) for(int x=0;x<ROI_SOURCE_WIDTH;++x) image[y*ROI_SOURCE_WIDTH+x]=(x*13+y*29+(x*y>>4))&255;
    const float identity[9]={1,0,0,0,1,0,0,0,1};
    const float translation[9]={1,0,17.375f,0,1,39.625f,0,0,1};
    const float projective[9]={1.31f,.072f,12.25f,.041f,.91f,28.75f,.00033f,-.00021f,1};
    if(!check("identity",image,identity)||!check("translation",image,translation)||!check("projective",image,projective)) return 1;
    std::cout<<"ROI_PROJECTION_TEST_PASS: identity, fractional translation, projective"<<std::endl; return 0;
}
