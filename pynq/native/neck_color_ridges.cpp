// Bounded PS verification of segmented strips; no camera or PL access.
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace {
struct Support {
    bool supported = false;
    int run = 0;
    std::vector<int> peaks;
    std::vector<float> support, strength;
};

float median(std::vector<float> values) {
    const size_t middle = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + middle, values.end());
    float result = values[middle];
    if (!(values.size() % 2)) {
        result = (result + *std::max_element(values.begin(), values.begin() + middle)) * .5f;
    }
    return result;
}

bool better(const Support& a, const Support& b) {
    if (a.supported != b.supported) return a.supported;
    if (a.run != b.run) return a.run > b.run;
    return a.peaks.size() > b.peaks.size();
}

Support ridges(const cv::Mat& patch) {
    Support best;
    for (int sign : {1, -1}) {
        std::array<float, 256> profile{};
        std::array<int, 256> votes{};
        for (int y = 5; y < 35; ++y) {
            std::array<float, 256> ridge{};
            const unsigned char* row = patch.ptr<unsigned char>(y);
            for (int x = 2; x < 254; ++x) {
                ridge[x] = std::min(sign * (int(row[x]) - row[x - 2]),
                                    sign * (int(row[x]) - row[x + 2]));
                profile[x] += std::max(0.f, ridge[x]) / 30.f;
            }
            for (int x = 0; x < 256; ++x) {
                float maximum = ridge[x];
                if (x > 0) maximum = std::max(maximum, ridge[x - 1]);
                if (x < 255) maximum = std::max(maximum, ridge[x + 1]);
                votes[x] += maximum > 8;
            }
        }
        float baseline = median(std::vector<float>(profile.begin(), profile.end()));
        std::vector<float> deviation;
        for (float value : profile) deviation.push_back(std::abs(value - baseline));
        float threshold = std::max(4.f, baseline + 2 * median(deviation));
        std::vector<int> maxima;
        for (int x = 1; x < 255; ++x) {
            if (profile[x] >= profile[x - 1] && profile[x] > profile[x + 1] &&
                profile[x] >= threshold && votes[x] >= 14) maxima.push_back(x);
        }
        std::stable_sort(maxima.begin(), maxima.end(), [&](int a, int b) {
            return profile[a] > profile[b];
        });
        Support candidate;
        for (int x : maxima) {
            if (x < 5 || x > 250) continue;
            bool separate = true;
            for (int old : candidate.peaks) if (std::abs(x - old) < 6) separate = false;
            if (separate) candidate.peaks.push_back(x);
        }
        std::sort(candidate.peaks.begin(), candidate.peaks.end());
        for (int x : candidate.peaks) {
            candidate.support.push_back(votes[x] / 30.f);
            candidate.strength.push_back(profile[x]);
        }
        for (int count = int(candidate.peaks.size()); count >= 5 && !candidate.run; --count) {
            for (int start = 0; start + count <= int(candidate.peaks.size()); ++start) {
                int smallest = 256, largest = 0, previous = 0;
                bool valid = true;
                for (int k = start; k + 1 < start + count; ++k) {
                    int gap = candidate.peaks[k + 1] - candidate.peaks[k];
                    smallest = std::min(smallest, gap);
                    largest = std::max(largest, gap);
                    if (previous && (float(gap) / previous < .60f || float(gap) / previous > 1.60f)) valid = false;
                    previous = gap;
                }
                if (valid && largest <= smallest * 2.2f &&
                    candidate.peaks[start + count - 1] - candidate.peaks[start] >= 64) {
                    candidate.run = count;
                    break;
                }
            }
        }
        candidate.supported = candidate.run >= 5;
        if (better(candidate, best)) best = candidate;
    }
    return best;
}
}

extern "C" int neck_color_ridges(const unsigned char* gray, int width, int height,
        const float* quad, int shear, float* output) {
    if (!gray || !quad || !output || width < 24 || height < 24 ||
        width > 4096 || height > 4096 || (shear != 0 && shear != 1)) return -1;
    for (int i = 0; i < 8; ++i) if (!std::isfinite(quad[i])) return -1;
    try {
        std::vector<cv::Point2f> source, target{{0, 0}, {255, 0}, {255, 39}, {0, 39}};
        for (int i = 0; i < 4; ++i) source.emplace_back(quad[2 * i], quad[2 * i + 1]);
        cv::Mat transform = cv::getPerspectiveTransform(source, target), patch;
        cv::warpPerspective(cv::Mat(height, width, CV_8U, const_cast<unsigned char*>(gray)),
                            patch, transform, cv::Size(256, 40), cv::INTER_LINEAR);
        Support best = ridges(patch);
        if (shear && !best.supported) {
            for (float amount : {-.5f, .5f, -1.f, 1.f}) {
                cv::Mat matrix = (cv::Mat_<float>(2, 3) << 1, amount, -19.5f * amount, 0, 1, 0);
                cv::Mat adjusted;
                cv::warpAffine(patch, adjusted, matrix, cv::Size(256, 40),
                               cv::INTER_LINEAR, cv::BORDER_REPLICATE);
                Support candidate = ridges(adjusted);
                if (better(candidate, best)) best = candidate;
                if (best.supported) break;
            }
        }
        output[0] = float(best.peaks.size());
        output[1] = float(best.run);
        output[2] = best.peaks.empty() ? 0.f : (best.peaks.back() - best.peaks.front()) / 255.f;
        for (size_t i = 0; i < best.peaks.size(); ++i) {
            output[3 + i] = best.peaks[i] / 255.f;
            output[64 + i] = best.support[i];
            output[106 + i] = best.strength[i];
        }
        return best.supported ? 1 : 0;
    } catch (...) { return -1; }
}
