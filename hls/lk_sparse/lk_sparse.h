#ifndef LK_SPARSE_H
#define LK_SPARSE_H
#include <ap_int.h>

// Pixel coordinates are unsigned x[15:0], y[31:16].
// Results: signed Q6 dx[15:0], dy[31:16], valid[32].
void lk_sparse(const ap_uint<8>* previous, const ap_uint<8>* current,
               const ap_uint<32>* points, ap_uint<64>* result,
               unsigned width, unsigned height, unsigned count);
#endif
