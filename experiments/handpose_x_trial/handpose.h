#ifndef HANDPOSE_H
#define HANDPOSE_H

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <net.h>

#define HANDPOSE_PARAM_PATH "/home/dell/Code/c++/handpose/weights/resnet_50_size_256_handposeX.ncnn.param"
#define HANDPOSE_BIN_PATH "/home/dell/Code/c++/handpose/weights/resnet_50_size_256_handposeX.ncnn.bin"

struct Object_handpose {
    std::vector<cv::Point2f> kpts;  // 21关键点
};

class Handpose
{
public:
    Handpose();

    ~Handpose();

    int detect(const cv::Mat& rgb, Object_handpose& obj);
    
    int draw(cv::Mat& rgb, const Object_handpose& obj);

private:
    ncnn::Net handpose;

    const float mean_vals[3] = {128.f, 128.f, 128.f};
    const float norm_vals[3] = {1.f / 256.f, 1.f / 256.f, 1.f / 256.f};
    const int target_size = 256;
    const bool use_gpu = true;

    ncnn::UnlockedPoolAllocator blob_pool_allocator;
    ncnn::PoolAllocator workspace_pool_allocator;
};

#endif // HANDPOSE_H