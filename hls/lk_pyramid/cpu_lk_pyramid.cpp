#include "cpu_lk_pyramid.h"
#include <vector>

namespace {
const unsigned kMaxWidth=320, kMaxHeight=200, kMaxPoints=64, kLevels=4, kRadius=10, kIters=10;
static float bilinear(const uint8_t* image, unsigned stride, float x, float y) {
    unsigned ix=(unsigned)x, iy=(unsigned)y; float fx=x-ix, fy=y-iy;
    float a=image[iy*stride+ix], b=image[iy*stride+ix+1];
    float c=image[(iy+1)*stride+ix], d=image[(iy+1)*stride+ix+1];
    return (a+fx*(b-a))+fy*((c+fx*(d-c))-(a+fx*(b-a)));
}
static bool inside(float x, float y, unsigned w, unsigned h) {
    const float lo=(float)(kRadius+1);
    return x>=lo && y>=lo && x<(float)(w-kRadius-2) && y<(float)(h-kRadius-2);
}
static uint8_t gaussian5(const uint8_t* source, unsigned width, unsigned height, unsigned x, unsigned y) {
    const unsigned wt[5]={1,4,6,4,1}; unsigned sum=0;
    for(int j=-2;j<=2;++j)for(int i=-2;i<=2;++i) { int sx=(int)(x<<1)+i,sy=(int)(y<<1)+j;
        if(sx<0)sx=0;else if(sx>=(int)width)sx=width-1;if(sy<0)sy=0;else if(sy>=(int)height)sy=height-1;
        sum+=wt[i+2]*wt[j+2]*source[sy*width+sx]; }
    return (uint8_t)((sum+128)>>8);
}
}

extern "C" void cpu_lk_pyramid(const uint8_t* previous, const uint8_t* current,
                                const float* points, float* result,
                                unsigned width, unsigned height, unsigned count) {
    if (width>kMaxWidth || height>kMaxHeight || width<16 || height<16 || count>kMaxPoints) return;
    const unsigned l0=kMaxWidth*kMaxHeight, l1=(kMaxWidth/2)*(kMaxHeight/2);
    const unsigned l2=(kMaxWidth/4)*(kMaxHeight/4);
    const unsigned offsets[kLevels]={0,l0,l0+l1,l0+l1+l2};
    std::vector<uint8_t> a(l0+l1+l2+(kMaxWidth/8)*(kMaxHeight/8));
    std::vector<uint8_t> b(a.size());
    for(unsigned i=0;i<width*height;++i) { a[i]=previous[i]; b[i]=current[i]; }
    for(unsigned level=1;level<kLevels;++level) {
        unsigned pw=width>>(level-1), ph=height>>(level-1), nw=width>>level, nh=height>>level;
        (void)ph;
        for(unsigned y=0;y<nh;++y) for(unsigned x=0;x<nw;++x) {
            unsigned q=offsets[level]+y*nw+x;
            a[q]=gaussian5(a.data()+offsets[level-1],pw,ph,x,y);
            b[q]=gaussian5(b.data()+offsets[level-1],pw,ph,x,y);
        }
    }
    for(unsigned p=0;p<count;++p) {
        float x0=points[2*p],y0=points[2*p+1],dx=0,dy=0,residual=0; bool valid=x0==x0&&y0==y0;
        int start=-1;
        for(int level=3;level>=0;--level) { unsigned w=width>>level,h=height>>level; float s=(float)(1<<level);
            if(w>=12&&h>=12&&inside(x0/s,y0/s,w,h)){start=level;break;} }
        if(start<0) valid=false; bool first=true;
        for(int level=start;level>=0;--level) {
            unsigned w=width>>level,h=height>>level; float s=(float)(1<<level),rx=x0/s,ry=y0/s;
            if(!first){dx*=2;dy*=2;} first=false;
            if(!valid||!inside(rx,ry,w,h)){valid=false;continue;}
            const uint8_t* ref=a.data()+offsets[level]; const uint8_t* img=b.data()+offsets[level]; float xx=0,xy=0,yy=0, rp[21][21], gxv[21][21], gyv[21][21];
            for(int j=-10;j<=10;++j)for(int i=-10;i<=10;++i){int r=j+10,c=i+10;float gx=.5f*(bilinear(ref,w,rx+i+1,ry+j)-bilinear(ref,w,rx+i-1,ry+j));float gy=.5f*(bilinear(ref,w,rx+i,ry+j+1)-bilinear(ref,w,rx+i,ry+j-1));rp[r][c]=bilinear(ref,w,rx+i,ry+j);gxv[r][c]=gx;gyv[r][c]=gy;xx+=gx*gx;xy+=gx*gy;yy+=gy*gy;}
            float det=xx*yy-xy*xy;if(det<1000){valid=false;continue;}
            for(unsigned it=0;it<kIters;++it){float qx=rx+dx,qy=ry+dy;if(!inside(qx,qy,w,h)){valid=false;break;}float bx=0,by=0;
                for(int j=-10;j<=10;++j)for(int i=-10;i<=10;++i){int r=j+10,c=i+10;float e=bilinear(img,w,qx+i,qy+j)-rp[r][c];bx+=gxv[r][c]*e;by+=gyv[r][c]*e;}
                float ux=(xy*by-yy*bx)/det,uy=(xy*bx-xx*by)/det;dx+=ux;dy+=uy;if(ux*ux+uy*uy<.0001f)break;}
        }
        if(valid&&inside(x0+dx,y0+dy,width,height)){float sum=0;for(int j=-10;j<=10;++j)for(int i=-10;i<=10;++i){float e=bilinear(b.data(),width,x0+dx+i,y0+dy+j)-bilinear(a.data(),width,x0+i,y0+j);sum+=e<0?-e:e;}residual=sum/441.f;}else valid=false;
        result[4*p]=valid?x0+dx:x0;result[4*p+1]=valid?y0+dy:y0;result[4*p+2]=valid?1.f:0.f;result[4*p+3]=valid?residual:0.f;
    }
}
