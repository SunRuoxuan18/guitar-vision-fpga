#include "lk_pyramid.h"
#include "cpu_lk_pyramid.h"
#include <cmath>
#include <iostream>
#include <vector>

static const unsigned W = 320, H = 200, N = 40;

static float texture(float x, float y) {
    return 123.0f + 31.0f * std::sin(0.123f * x + 0.071f * y) +
           27.0f * std::cos(0.181f * x - 0.113f * y) +
           19.0f * std::sin(0.057f * x + 0.097f * y);
}

int main() {
    std::vector<ap_uint<8> > before(W*H), after(W*H);
    std::vector<uint8_t> before_cpu(W*H), after_cpu(W*H);
    std::vector<float> points(2*N), got(4*N), cpu(4*N);
    for(unsigned p=0;p<N;++p) {
        // Exercise adaptive safe-level selection with subpixel interior points.
        points[2*p]=45.25f+(p%8)*27.1f;
        points[2*p+1]=45.75f+(p/8)*22.3f;
    }
    const float shifts[][2]={{1,0},{4,0},{8,0},{0,1},{0,4},{0,8},{.35f,-.45f},{2.5f,-1.75f}};
    unsigned checked=0;
    for(unsigned s=0;s<sizeof(shifts)/sizeof(shifts[0]);++s) {
        for(unsigned y=0;y<H;++y) for(unsigned x=0;x<W;++x) {
            before[y*W+x]=(unsigned)std::lround(texture(x,y));
            after[y*W+x]=(unsigned)std::lround(texture(x-shifts[s][0],y-shifts[s][1]));
            before_cpu[y*W+x]=(uint8_t)before[y*W+x];
            after_cpu[y*W+x]=(uint8_t)after[y*W+x];
        }
        lk_pyramid(before.data(),after.data(),points.data(),got.data(),W,H,N);
        cpu_lk_pyramid(before_cpu.data(),after_cpu.data(),points.data(),cpu.data(),W,H,N);
        unsigned valid_count=0;
        for(unsigned p=0;p<N;++p) {
            float err=std::hypot(got[4*p]-points[2*p]-shifts[s][0],got[4*p+1]-points[2*p+1]-shifts[s][1]);
            float cpu_err=std::hypot(got[4*p]-cpu[4*p],got[4*p+1]-cpu[4*p+1]);
            if(got[4*p+2] != cpu[4*p+2] || cpu_err>.001f) {
                std::cerr << "translation failure shift=" << shifts[s][0] << "," << shifts[s][1]
                          << " point=" << p << " got=" << got[4*p]-points[2*p] << ","
                          << got[4*p+1]-points[2*p+1] << " err=" << err << " cpu_delta=" << cpu_err << "\n";
                return 1;
            }
            if (got[4*p+2] == 1.0f) {
                if (err > .7f) {
                    std::cerr << "accuracy failure shift=" << shifts[s][0] << "," << shifts[s][1]
                              << " point=" << p << " err=" << err << "\n";
                    return 1;
                }
                ++valid_count;
                ++checked;
            }
        }
        if (valid_count < 32) {
            std::cerr << "insufficient valid tracks shift=" << shifts[s][0] << "," << shifts[s][1]
                      << " valid=" << valid_count << "\n";
            return 1;
        }
        std::vector<float> reverse_cached(4*N), reverse_fresh(4*N);
        lk_pyramid(after.data(),before.data(),points.data(),reverse_cached.data(),W,H,N,1);
        lk_pyramid(after.data(),before.data(),points.data(),reverse_fresh.data(),W,H,N,0);
        for(unsigned i=0;i<4*N;++i) {
            if(std::abs(reverse_cached[i]-reverse_fresh[i])>.001f) {
                std::cerr << "cached reverse differs from fresh: " << i << "\n";
                return 4;
            }
        }
        lk_pyramid(before.data(),after.data(),points.data(),reverse_cached.data(),W,H,N,1);
        lk_pyramid(before.data(),after.data(),points.data(),reverse_cached.data(),W,H,N,1);
        for(unsigned p=0;p<N;++p) if(reverse_cached[4*p+2] != 0) return 5;
    }
    for(unsigned i=0;i<W*H;++i) before[i]=after[i]=100;
    lk_pyramid(before.data(),after.data(),points.data(),got.data(),W,H,N);
    for(unsigned p=0;p<N;++p) if(got[4*p+2] != 0.0f) return 2;
    points[0]=2.25f; points[1]=2.75f;
    lk_pyramid(before.data(),after.data(),points.data(),got.data(),W,H,N);
    if(got[2] != 0.0f) return 3;
    std::cout << "LK_PYRAMID_TEST_PASS: " << checked
              << " textured subpixel-point checks; 1/4/8px and subpixel translations; blank/boundary rejected\n";
    return 0;
}
