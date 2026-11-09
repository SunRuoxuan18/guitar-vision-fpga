#include "handpose.h"
#include "cpu.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

Handpose::Handpose() {
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);

    handpose.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    handpose.opt = ncnn::Option();

#if NCNN_VULKAN
    handpose.opt.use_vulkan_compute = use_gpu;
#endif

    handpose.opt.num_threads = ncnn::get_big_cpu_count();
    handpose.opt.blob_allocator = &blob_pool_allocator;
    handpose.opt.workspace_allocator = &workspace_pool_allocator;

    handpose.load_param(HANDPOSE_PARAM_PATH);
    handpose.load_model(HANDPOSE_BIN_PATH);
}

Handpose::~Handpose()
{
    handpose.clear();
}

int Handpose::detect(const cv::Mat& rgb, Object_handpose& obj)
{
    obj.kpts.clear();
    
    const int img_w = rgb.cols;
    const int img_h = rgb.rows;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(
        rgb.data, ncnn::Mat::PIXEL_BGR,
        img_w, img_h,
        target_size, target_size
    );

    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = handpose.create_extractor();
    ex.input("in0", in);

    ncnn::Mat out;
    ex.extract("out0", out);

    if (out.dims != 1 || out.w != 42) {
        fprintf(stderr, "ERROR: Unexpected output shape - dims=%d, w=%d (expected dims=1, w=42)\n", 
                out.dims, out.w);
        return -1;
    }

    obj.kpts.resize(21);

    const float* out_data = (const float*)out.data;

    // 提取21个关键点坐标
    for (int i = 0; i < 21; i++)
    {
        float x = out_data[i * 2];
        float y = out_data[i * 2 + 1];

        // 映射到输入图像的实际尺寸
        x *= img_w;
        y *= img_h;

        obj.kpts[i] = cv::Point2f(x, y);
    }

    return 0;
}

int Handpose::draw(cv::Mat& rgb, const Object_handpose& obj)
{
    if (obj.kpts.size() != 21) {
        fprintf(stderr, "ERROR: Invalid keypoints count: %zu\n", obj.kpts.size());
        return -1;
    }

    // 绘制21个关键点
    for (int k = 0; k < 21; k++)
    {
        cv::circle(rgb, obj.kpts[k], 4, cv::Scalar(255, 155, 0), -1);
    }

    // 手部骨架连接关系
    static const int connections[20][2] = {
        // 大拇指
        {0, 1}, {1, 2}, {2, 3}, {3, 4},
        // 食指
        {0, 5}, {5, 6}, {6, 7}, {7, 8},
        // 中指
        {0, 9}, {9, 10}, {10, 11}, {11, 12},
        // 无名指
        {0, 13}, {13, 14}, {14, 15}, {15, 16},
        // 小指
        {0, 17}, {17, 18}, {18, 19}, {19, 20}
    };

    // 每个手指使用不同颜色
    static const cv::Scalar colors[5] = {
        cv::Scalar(0, 0, 255),      // 大拇指 - 红色
        cv::Scalar(255, 0, 255),    // 食指 - 品红
        cv::Scalar(0, 255, 255),    // 中指 - 黄色
        cv::Scalar(0, 255, 0),      // 无名指 - 绿色
        cv::Scalar(255, 0, 0)       // 小指 - 蓝色
    };

    // 绘制骨架连接线
    for (int i = 0; i < 20; i++)
    {
        int idx1 = connections[i][0];
        int idx2 = connections[i][1];
        int finger_idx = i / 4;
        
        cv::line(rgb, obj.kpts[idx1], obj.kpts[idx2], 
                 colors[finger_idx], 2, cv::LINE_AA);
    }

    return 0;
}