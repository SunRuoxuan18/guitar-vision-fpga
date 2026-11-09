#ifndef HAND_REQUANT_H
#define HAND_REQUANT_H

#include <ap_int.h>

enum {
    HAND_REQUANT_MAX_CHANNELS = 1024,
    HAND_REQUANT_MAX_ELEMENTS = 1048576,
    HAND_REQUANT_PARAM_COUNT = HAND_REQUANT_MAX_CHANNELS * 2,
    HAND_REQUANT_METADATA_SCALE = 0,
    HAND_REQUANT_METADATA_STATUS = 1
};

// params[0..1023] are per-channel scales; params[1024..2047] are biases.
// metadata[0] receives the dynamic activation scale. metadata[1] is zero on
// success, -1 for invalid dimensions or Clip bounds, and -2 for nonfinite
// scales/biases or conversion results.
void hand_requant(const long long *input, const float *params,
                  ap_uint<128> *output, float *metadata,
                  int channels, int pixels, float clip_min, float clip_max);

#endif
