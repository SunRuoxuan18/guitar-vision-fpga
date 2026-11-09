#ifndef ROI_PROJECTION_H
#define ROI_PROJECTION_H

#include <ap_int.h>

enum { ROI_SOURCE_WIDTH = 640, ROI_SOURCE_HEIGHT = 480, ROI_WIDTH = 320,
       ROI_HEIGHT = 128, ROI_PIXELS = ROI_WIDTH * ROI_HEIGHT };

// packed[i]: rectified [7:0], abs(Sobel x) [18:8], abs(Sobel y) [29:19].
void roi_projection(const ap_uint<8> image[ROI_SOURCE_WIDTH * ROI_SOURCE_HEIGHT],
                    const float inverse_homography[9], ap_uint<32> packed[ROI_PIXELS]);
#endif
