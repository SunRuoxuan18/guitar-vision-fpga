#ifndef HAND_CONV_H
#define HAND_CONV_H

// Activations and outputs are NCHW for one image. Weights are OIHW, with the
// input-channel dimension local to each group (OpenCV DNN convolution layout).
void hand_conv(float *input, float *weights, float *bias, float *output,
               int height, int width, int channels, int outputs, int kernel,
               int stride, int pad_top, int pad_left, int out_h, int out_w,
               int groups);

#endif
