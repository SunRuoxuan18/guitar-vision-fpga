#include "roi_projection.h"

enum { ROW_CACHE_LINES = 16 };

static ap_uint<8> source_sample(const ap_uint<8> image[ROI_SOURCE_WIDTH * ROI_SOURCE_HEIGHT],
                                ap_uint<8> row_cache[ROW_CACHE_LINES][ROI_SOURCE_WIDTH],
                                int row_tag[ROW_CACHE_LINES], int x, int y) {
#pragma HLS INLINE off
    if (x < 0 || x >= ROI_SOURCE_WIDTH || y < 0 || y >= ROI_SOURCE_HEIGHT)
        return ap_uint<8>(0);
    int slot = y & (ROW_CACHE_LINES - 1);
    if (row_tag[slot] != y) {
fill_cached_row:
        for (int column = 0; column < ROI_SOURCE_WIDTH; ++column) {
#pragma HLS PIPELINE II=1
            row_cache[slot][column] = image[y * ROI_SOURCE_WIDTH + column];
        }
        row_tag[slot] = y;
    }
    return row_cache[slot][x];
}

static int reflect101(int index, int limit) {
#pragma HLS INLINE
    // Callers only request index +/- 1, so this is exactly BORDER_REFLECT_101
    // without a variable-bound loop in the pixel datapath.
    return index < 0 ? -index : (index >= limit ? 2 * limit - index - 2 : index);
}

void roi_projection(const ap_uint<8> image[ROI_SOURCE_WIDTH * ROI_SOURCE_HEIGHT],
                    const float inverse_homography[9], ap_uint<32> packed[ROI_PIXELS]) {
#pragma HLS INTERFACE m_axi port=image offset=slave bundle=image depth=307200 max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=inverse_homography offset=slave bundle=inverse depth=9 max_read_burst_length=16
#pragma HLS INTERFACE m_axi port=packed offset=slave bundle=output depth=40960 max_write_burst_length=64
#pragma HLS INTERFACE s_axilite port=image bundle=control offset=0x10
#pragma HLS INTERFACE s_axilite port=inverse_homography bundle=control offset=0x1c
#pragma HLS INTERFACE s_axilite port=packed bundle=control offset=0x28
#pragma HLS INTERFACE s_axilite port=return bundle=control
    ap_uint<8> rectified[ROI_HEIGHT][ROI_WIDTH];
    ap_uint<8> blurred[ROI_HEIGHT][ROI_WIDTH];
    ap_uint<8> row_cache[ROW_CACHE_LINES][ROI_SOURCE_WIDTH];
    int row_tag[ROW_CACHE_LINES];
    float h[9];
#pragma HLS BIND_STORAGE variable=rectified type=ram_t2p impl=bram
#pragma HLS BIND_STORAGE variable=blurred type=ram_t2p impl=bram
#pragma HLS BIND_STORAGE variable=row_cache type=ram_t2p impl=bram
#pragma HLS ARRAY_PARTITION variable=h complete dim=1

init_row_tags:
    for (int i=0; i<ROW_CACHE_LINES; ++i) {
#pragma HLS UNROLL
        row_tag[i] = -1;
    }

load_homography:
    for (int i=0; i<9; ++i) {
#pragma HLS UNROLL
        h[i] = inverse_homography[i];
    }

warp:
    for (int y=0; y<ROI_HEIGHT; ++y) for (int x=0; x<ROI_WIDTH; ++x) {
        // A tagged 16-row cache changes common row reuse into AXI bursts while
        // retaining direct DDR behavior on every cache miss.
        float d=h[6]*x+h[7]*y+h[8];
        float sx=0.0f, sy=0.0f;
        bool valid = d != 0.0f;
        if (valid) {
            sx=(h[0]*x+h[1]*y+h[2])/d;
            sy=(h[3]*x+h[4]*y+h[5])/d;
        }
        int x0=(int)sx, y0=(int)sy;
        if (sx < 0.0f && (float)x0 != sx) --x0;
        if (sy < 0.0f && (float)y0 != sy) --y0;
        float fx=sx-(float)x0, fy=sy-(float)y0;
        float top=(1.0f-fx)*(float)source_sample(image,row_cache,row_tag,x0,y0)+fx*(float)source_sample(image,row_cache,row_tag,x0+1,y0);
        float bottom=(1.0f-fx)*(float)source_sample(image,row_cache,row_tag,x0,y0+1)+fx*(float)source_sample(image,row_cache,row_tag,x0+1,y0+1);
        int value=valid ? (int)(top*(1.0f-fy)+bottom*fy+0.5f) : 0;
        rectified[y][x]=(ap_uint<8>)(value < 0 ? 0 : (value > 255 ? 255 : value));
    }
gaussian:
    for (int y=0; y<ROI_HEIGHT; ++y) for (int x=0; x<ROI_WIDTH; ++x) {
        int xm=reflect101(x-1,ROI_WIDTH), xp=reflect101(x+1,ROI_WIDTH);
        int ym=reflect101(y-1,ROI_HEIGHT), yp=reflect101(y+1,ROI_HEIGHT);
        int sum=rectified[ym][xm]+2*rectified[ym][x]+rectified[ym][xp]
               +2*rectified[y][xm]+4*rectified[y][x]+2*rectified[y][xp]
               +rectified[yp][xm]+2*rectified[yp][x]+rectified[yp][xp];
        blurred[y][x]=(ap_uint<8>)((sum+8)>>4);
    }
sobel_and_pack:
    for (int y=0; y<ROI_HEIGHT; ++y) for (int x=0; x<ROI_WIDTH; ++x) {
        int xm=reflect101(x-1,ROI_WIDTH), xp=reflect101(x+1,ROI_WIDTH);
        int ym=reflect101(y-1,ROI_HEIGHT), yp=reflect101(y+1,ROI_HEIGHT);
        int gx=(int)blurred[ym][xp]+2*(int)blurred[y][xp]+(int)blurred[yp][xp]
              -(int)blurred[ym][xm]-2*(int)blurred[y][xm]-(int)blurred[yp][xm];
        int gy=(int)blurred[yp][xm]+2*(int)blurred[yp][x]+(int)blurred[yp][xp]
              -(int)blurred[ym][xm]-2*(int)blurred[ym][x]-(int)blurred[ym][xp];
        unsigned ax=gx < 0 ? (unsigned)-gx : (unsigned)gx;
        unsigned ay=gy < 0 ? (unsigned)-gy : (unsigned)gy;
        ap_uint<32> word=rectified[y][x]; word.range(18,8)=ax; word.range(29,19)=ay;
        packed[y*ROI_WIDTH+x]=word;
    }
}
