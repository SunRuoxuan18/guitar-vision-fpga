#include "lk_sparse.h"

void lk_sparse(const ap_uint<8>* previous, const ap_uint<8>* current,
               const ap_uint<32>* points, ap_uint<64>* result,
               unsigned width, unsigned height, unsigned count) {
#pragma HLS INTERFACE m_axi port=previous offset=slave bundle=before depth=64000 max_read_burst_length=16
#pragma HLS INTERFACE m_axi port=current offset=slave bundle=after depth=64000 max_read_burst_length=16
#pragma HLS INTERFACE m_axi port=points offset=slave bundle=features depth=64
#pragma HLS INTERFACE m_axi port=result offset=slave bundle=features depth=64
#pragma HLS INTERFACE s_axilite port=previous bundle=control offset=0x10
#pragma HLS INTERFACE s_axilite port=current bundle=control offset=0x1c
#pragma HLS INTERFACE s_axilite port=points bundle=control offset=0x28
#pragma HLS INTERFACE s_axilite port=result bundle=control offset=0x34
#pragma HLS INTERFACE s_axilite port=width bundle=control offset=0x40
#pragma HLS INTERFACE s_axilite port=height bundle=control offset=0x48
#pragma HLS INTERFACE s_axilite port=count bundle=control offset=0x50
#pragma HLS INTERFACE s_axilite port=return bundle=control
    if (width < 16 || width > 320 || height < 16 || height > 200 || count > 64)
        return;
points_loop:
    for (unsigned p = 0; p < count; ++p) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=64 avg=48
        ap_uint<32> point = points[p];
        unsigned x = point.range(15, 0), y = point.range(31, 16);
        ap_uint<64> output = 0;
        if (x >= 4 && x + 4 < width && y >= 4 && y + 4 < height) {
            ap_uint<8> a[9][9], b[9][9];
#pragma HLS ARRAY_PARTITION variable=a complete dim=0
#pragma HLS ARRAY_PARTITION variable=b complete dim=0
load_rows:
            for (int row = 0; row < 9; ++row) {
load_columns:
                for (int col = 0; col < 9; ++col) {
#pragma HLS PIPELINE II=1
                    unsigned address = (y + row - 4) * width + x + col - 4;
                    a[row][col] = previous[address];
                    b[row][col] = current[address];
                }
            }
            ap_int<24> xx=0, xy=0, yy=0, xt=0, yt=0;
window_rows:
            for (int row=1; row<8; ++row) {
window_columns:
                for (int col=1; col<8; ++col) {
#pragma HLS PIPELINE II=1
                    ap_int<10> gx = ((int)a[row][col+1] - (int)a[row][col-1]
                                   + (int)b[row][col+1] - (int)b[row][col-1])/2;
                    ap_int<10> gy = ((int)a[row+1][col] - (int)a[row-1][col]
                                   + (int)b[row+1][col] - (int)b[row-1][col])/2;
                    ap_int<10> gt = 2*((int)b[row][col] - (int)a[row][col]);
                    xx += gx*gx; xy += gx*gy; yy += gy*gy;
                    xt += gx*gt; yt += gy*gt;
                }
            }
            ap_int<50> determinant = (ap_int<50>)xx*yy - (ap_int<50>)xy*xy;
            ap_int<25> trace = xx+yy;
            ap_int<52> quality = (ap_int<52>)trace*trace;
            if (trace > 1024 && determinant > (quality >> 8)) {
                ap_int<56> nx = ((ap_int<56>)xy*yt - (ap_int<56>)yy*xt) << 6;
                ap_int<56> ny = ((ap_int<56>)xy*xt - (ap_int<56>)xx*yt) << 6;
                ap_int<56> dx = nx/determinant, dy = ny/determinant;
                if (dx >= -256 && dx <= 256 && dy >= -256 && dy <= 256) {
                    output.range(15,0) = (ap_int<16>)dx;
                    output.range(31,16) = (ap_int<16>)dy;
                    output[32] = 1;
                }
            }
        }
        result[p] = output;
    }
}
