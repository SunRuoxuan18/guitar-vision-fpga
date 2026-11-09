#ifndef LK_PYRAMID_H
#define LK_PYRAMID_H

#include <ap_int.h>

// Fixed accelerator limits. Points and results are tightly packed float arrays.
static const unsigned LK_MAX_WIDTH = 320;
static const unsigned LK_MAX_HEIGHT = 200;
static const unsigned LK_MAX_POINTS = 64;
static const unsigned LK_LEVELS = 4;
static const unsigned LK_PATCH_RADIUS = 10;  // 21x21 patch
static const unsigned LK_MAX_ITERS = 10;

// points: [x0, y0, x1, y1, ...], count * 2 floats.
// result: [x0, y0, valid0, mean_abs_residual0, ...], count * 4 floats.
// Coordinates are in full-resolution pixels. valid is exactly 0.0f or 1.0f.
void lk_pyramid(const ap_uint<8>* previous, const ap_uint<8>* current,
                const float* points, float* result,
                unsigned width, unsigned height, unsigned count, unsigned reuse_reverse = 0);

#endif
