#include "lk_pyramid.h"

static const unsigned LK_L0_PIXELS = LK_MAX_WIDTH * LK_MAX_HEIGHT;
static const unsigned LK_L1_PIXELS = (LK_MAX_WIDTH / 2) * (LK_MAX_HEIGHT / 2);
static const unsigned LK_L2_PIXELS = (LK_MAX_WIDTH / 4) * (LK_MAX_HEIGHT / 4);
static const unsigned LK_L3_PIXELS = (LK_MAX_WIDTH / 8) * (LK_MAX_HEIGHT / 8);

static float bilinear_at(const ap_uint<8>* image, unsigned stride, unsigned ix, unsigned iy,
                         float fx, float fy) {
#pragma HLS INLINE
    float a = (float)image[iy * stride + ix];
    float b = (float)image[iy * stride + ix + 1];
    float c = (float)image[(iy + 1) * stride + ix];
    float d = (float)image[(iy + 1) * stride + ix + 1];
    return (a + fx * (b - a)) + fy * ((c + fx * (d - c)) - (a + fx * (b - a)));
}

static float bilinear(const ap_uint<8>* image, unsigned stride, float x, float y) {
#pragma HLS INLINE
    unsigned ix = (unsigned)x, iy = (unsigned)y;
    return bilinear_at(image, stride, ix, iy, x - (float)ix, y - (float)iy);
}

static bool patch_inside(float x, float y, unsigned width, unsigned height) {
#pragma HLS INLINE
    const float lo = (float)(LK_PATCH_RADIUS + 1);
    return x >= lo && y >= lo &&
           x < (float)(width - LK_PATCH_RADIUS - 2) &&
           y < (float)(height - LK_PATCH_RADIUS - 2);
}

static ap_uint<8> gaussian5_down(const ap_uint<8>* source, unsigned width,
                                  unsigned height, unsigned x, unsigned y) {
#pragma HLS INLINE
    const unsigned weights[5] = {1, 4, 6, 4, 1};
    unsigned sum = 0;
gaussian_rows:
    for (int j = -2; j <= 2; ++j) {
gaussian_columns:
        for (int i = -2; i <= 2; ++i) {
            int sx = (int)(x << 1) + i, sy = (int)(y << 1) + j;
            if (sx < 0) sx = 0; else if (sx >= (int)width) sx = width - 1;
            if (sy < 0) sy = 0; else if (sy >= (int)height) sy = height - 1;
            sum += weights[i + 2] * weights[j + 2] * (unsigned)source[sy * width + sx];
        }
    }
    return (sum + 128) >> 8;
}

void lk_pyramid(const ap_uint<8>* previous, const ap_uint<8>* current,
                const float* points, float* result,
                unsigned width, unsigned height, unsigned count, unsigned reuse_reverse) {
#pragma HLS INTERFACE m_axi port=previous offset=slave bundle=previous depth=64000 max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=current offset=slave bundle=current depth=64000 max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=points offset=slave bundle=points depth=128 max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=result offset=slave bundle=result depth=256 max_write_burst_length=64
#pragma HLS INTERFACE s_axilite port=previous bundle=control offset=0x10
#pragma HLS INTERFACE s_axilite port=current bundle=control offset=0x1c
#pragma HLS INTERFACE s_axilite port=points bundle=control offset=0x28
#pragma HLS INTERFACE s_axilite port=result bundle=control offset=0x34
#pragma HLS INTERFACE s_axilite port=width bundle=control offset=0x40
#pragma HLS INTERFACE s_axilite port=height bundle=control offset=0x48
#pragma HLS INTERFACE s_axilite port=count bundle=control offset=0x50
#pragma HLS INTERFACE s_axilite port=reuse_reverse bundle=control offset=0x58
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Separate exact-depth memories prevent 85k-element depth rounding.
    static ap_uint<8> prev0[LK_L0_PIXELS], curr0[LK_L0_PIXELS];
    static ap_uint<8> prev1[LK_L1_PIXELS], curr1[LK_L1_PIXELS];
    static ap_uint<8> prev2[LK_L2_PIXELS], curr2[LK_L2_PIXELS];
    static ap_uint<8> prev3[LK_L3_PIXELS], curr3[LK_L3_PIXELS];
#pragma HLS BIND_STORAGE variable=prev0 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=curr0 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=prev1 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=curr1 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=prev2 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=curr2 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=prev3 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=curr3 type=ram_2p impl=bram

    static unsigned cached_width = 0, cached_height = 0;
    static bool cache_valid = false;
    if (width > LK_MAX_WIDTH || height > LK_MAX_HEIGHT || width < 16 ||
        height < 16 || count > LK_MAX_POINTS)
        return;

    bool reverse = reuse_reverse == 1 && cache_valid && cached_width == width && cached_height == height;
    if (reuse_reverse != 0 && !reverse) {
        for (unsigned p = 0; p < count; ++p) {
            result[p*4] = points[p*2]; result[p*4+1] = points[p*2+1];
            result[p*4+2] = 0; result[p*4+3] = 0;
        }
        cache_valid = false;
        return;
    }
    if (!reverse) {
copy_input:
    for (unsigned i = 0; i < width * height; ++i) {
#pragma HLS PIPELINE II=1
        prev0[i] = previous[i];
        curr0[i] = current[i];
    }

pyramid_levels:
    for (unsigned level = 1; level < LK_LEVELS; ++level) {
        unsigned pw = width >> (level - 1), ph = height >> (level - 1);
        unsigned nw = width >> level, nh = height >> level;
        ap_uint<8>* ps = level == 1 ? prev0 : (level == 2 ? prev1 : prev2);
        ap_uint<8>* cs = level == 1 ? curr0 : (level == 2 ? curr1 : curr2);
        ap_uint<8>* pd = level == 1 ? prev1 : (level == 2 ? prev2 : prev3);
        ap_uint<8>* cd = level == 1 ? curr1 : (level == 2 ? curr2 : curr3);
down_rows:
        for (unsigned y = 0; y < nh; ++y) {
down_columns:
            for (unsigned x = 0; x < nw; ++x) {
#pragma HLS PIPELINE II=1
                unsigned q = y * nw + x;
                pd[q] = gaussian5_down(ps, pw, ph, x, y);
                cd[q] = gaussian5_down(cs, pw, ph, x, y);
            }
        }
    }

    cached_width = width; cached_height = height; cache_valid = true;
    }
point_loop:
    for (unsigned p = 0; p < count; ++p) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=64 avg=32
        float x0 = points[2 * p];
        float y0 = points[2 * p + 1];
        float dx = 0.0f, dy = 0.0f, residual = 0.0f;
        bool valid = x0 == x0 && y0 == y0;
        int start_level = -1;
find_start_level:
        for (int level = LK_LEVELS - 1; level >= 0; --level) {
            unsigned lw = width >> level, lh = height >> level;
            float scale = (float)(1 << level);
            if (lw >= 12 && lh >= 12 && patch_inside(x0 / scale, y0 / scale, lw, lh)) {
                start_level = level;
                break;
            }
        }
        if (start_level < 0) valid = false;
        bool first_level = true;

level_loop:
        for (int level = start_level; level >= 0; --level) {
            unsigned lw = width >> level, lh = height >> level;
            float scale = (float)(1 << level);
            float rx = x0 / scale, ry = y0 / scale;
            if (!first_level) { dx *= 2.0f; dy *= 2.0f; }
            first_level = false;
            if (!valid || lw < 12 || lh < 12 || !patch_inside(rx, ry, lw, lh)) {
                valid = false;
                continue;
            }

            const ap_uint<8>* before = level == 0 ? prev0 : (level == 1 ? prev1 : (level == 2 ? prev2 : prev3));
            const ap_uint<8>* after = level == 0 ? curr0 : (level == 1 ? curr1 : (level == 2 ? curr2 : curr3));
            const ap_uint<8>* ref = reverse ? after : before;
            const ap_uint<8>* img = reverse ? before : after;
            unsigned rxi = (unsigned)rx, ryi = (unsigned)ry;
            float rfx = rx - (float)rxi, rfy = ry - (float)ryi;
            float ref_patch[21][21], gx_patch[21][21], gy_patch[21][21];
// The larger patch remains storage-efficient; do not fully partition it.
#pragma HLS ARRAY_PARTITION variable=ref_patch cyclic factor=3 dim=2
#pragma HLS ARRAY_PARTITION variable=gx_patch cyclic factor=3 dim=2
#pragma HLS ARRAY_PARTITION variable=gy_patch cyclic factor=3 dim=2
            float hxx_lane[4] = {0,0,0,0}, hxy_lane[4] = {0,0,0,0}, hyy_lane[4] = {0,0,0,0};
#pragma HLS ARRAY_PARTITION variable=hxx_lane complete
#pragma HLS ARRAY_PARTITION variable=hxy_lane complete
#pragma HLS ARRAY_PARTITION variable=hyy_lane complete
gradient_rows:
            for (int j = -(int)LK_PATCH_RADIUS; j <= (int)LK_PATCH_RADIUS; ++j) {
gradient_columns:
                for (int i = -(int)LK_PATCH_RADIUS; i <= (int)LK_PATCH_RADIUS; ++i) {
#pragma HLS PIPELINE II=4
                    int r = j + (int)LK_PATCH_RADIUS, c = i + (int)LK_PATCH_RADIUS;
                    unsigned lane = c & 3;
                    float gx = 0.5f * (bilinear_at(ref, lw, (unsigned)((int)rxi+i+1), (unsigned)((int)ryi+j), rfx, rfy) -
                                       bilinear_at(ref, lw, (unsigned)((int)rxi+i-1), (unsigned)((int)ryi+j), rfx, rfy));
                    float gy = 0.5f * (bilinear_at(ref, lw, (unsigned)((int)rxi+i), (unsigned)((int)ryi+j+1), rfx, rfy) -
                                       bilinear_at(ref, lw, (unsigned)((int)rxi+i), (unsigned)((int)ryi+j-1), rfx, rfy));
                    ref_patch[r][c] = bilinear_at(ref, lw, (unsigned)((int)rxi+i), (unsigned)((int)ryi+j), rfx, rfy);
                    gx_patch[r][c] = gx;
                    gy_patch[r][c] = gy;
                    hxx_lane[lane] += gx * gx; hxy_lane[lane] += gx * gy; hyy_lane[lane] += gy * gy;
                }
            }
            float hxx = hxx_lane[0] + hxx_lane[1] + hxx_lane[2] + hxx_lane[3];
            float hxy = hxy_lane[0] + hxy_lane[1] + hxy_lane[2] + hxy_lane[3];
            float hyy = hyy_lane[0] + hyy_lane[1] + hyy_lane[2] + hyy_lane[3];
            float det = hxx * hyy - hxy * hxy;
            if (det < 1000.0f) { valid = false; continue; }

iteration_loop:
            for (unsigned iter = 0; iter < LK_MAX_ITERS; ++iter) {
                float qx = rx + dx, qy = ry + dy;
                if (!patch_inside(qx, qy, lw, lh)) { valid = false; break; }
                unsigned qxi = (unsigned)qx, qyi = (unsigned)qy;
                float qfx = qx - (float)qxi, qfy = qy - (float)qyi;
                float bx_lane[4] = {0,0,0,0}, by_lane[4] = {0,0,0,0};
#pragma HLS ARRAY_PARTITION variable=bx_lane complete
#pragma HLS ARRAY_PARTITION variable=by_lane complete
residual_rows:
                for (int j = -(int)LK_PATCH_RADIUS; j <= (int)LK_PATCH_RADIUS; ++j) {
residual_columns:
                    for (int i = -(int)LK_PATCH_RADIUS; i <= (int)LK_PATCH_RADIUS; ++i) {
#pragma HLS PIPELINE II=4
                        int r = j + (int)LK_PATCH_RADIUS, c = i + (int)LK_PATCH_RADIUS;
                        unsigned lane = c & 3;
                        float gx = gx_patch[r][c];
                        float gy = gy_patch[r][c];
                        float e = bilinear_at(img, lw, (unsigned)((int)qxi+i), (unsigned)((int)qyi+j), qfx, qfy) - ref_patch[r][c];
                        bx_lane[lane] += gx * e; by_lane[lane] += gy * e;
                    }
                }
                float bx = bx_lane[0] + bx_lane[1] + bx_lane[2] + bx_lane[3];
                float by = by_lane[0] + by_lane[1] + by_lane[2] + by_lane[3];
                float ux = (hxy * by - hyy * bx) / det;
                float uy = (hxy * bx - hxx * by) / det;
                dx += ux; dy += uy;
                if (ux * ux + uy * uy < 0.0001f) break;
            }
        }

        if (valid && patch_inside(x0 + dx, y0 + dy, width, height)) {
            unsigned xbi = (unsigned)x0, ybi = (unsigned)y0;
            float xbf = x0-(float)xbi, ybf=y0-(float)ybi;
            float xt=x0+dx, yt=y0+dy; unsigned xti=(unsigned)xt, yti=(unsigned)yt;
            float xtf=xt-(float)xti, ytf=yt-(float)yti;
            float sum_lane[4] = {0,0,0,0};
#pragma HLS ARRAY_PARTITION variable=sum_lane complete
error_rows:
            for (int j = -(int)LK_PATCH_RADIUS; j <= (int)LK_PATCH_RADIUS; ++j) {
error_columns:
                for (int i = -(int)LK_PATCH_RADIUS; i <= (int)LK_PATCH_RADIUS; ++i) {
#pragma HLS PIPELINE II=4
                    unsigned lane = (i + LK_PATCH_RADIUS) & 3;
                    float e = bilinear_at(reverse ? prev0 : curr0, width, (unsigned)((int)xti+i), (unsigned)((int)yti+j), xtf, ytf) -
                              bilinear_at(reverse ? curr0 : prev0, width, (unsigned)((int)xbi+i), (unsigned)((int)ybi+j), xbf, ybf);
                    sum_lane[lane] += e < 0.0f ? -e : e;
                }
            }
            residual = (sum_lane[0] + sum_lane[1] + sum_lane[2] + sum_lane[3]) / 441.0f;
        } else valid = false;

        result[4 * p] = valid ? x0 + dx : x0;
        result[4 * p + 1] = valid ? y0 + dy : y0;
        result[4 * p + 2] = valid ? 1.0f : 0.0f;
        result[4 * p + 3] = valid ? residual : 0.0f;
    }
    if (reverse) cache_valid = false;
}
