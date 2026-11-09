#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

namespace {
struct Border {
    cv::Point2f first, second;
};
struct Geometry {
    cv::Point2f axis, normal, center;
};
float dot(const cv::Point2f& first, const cv::Point2f& second) {
    return first.x * second.x + first.y * second.y;
}
float length(const Border& border) {
    const cv::Point2f vector = border.second - border.first;
    return std::sqrt(dot(vector, vector));
}
Geometry geometry(const Border& border) {
    const cv::Point2f vector = border.second - border.first;
    const float magnitude = std::sqrt(dot(vector, vector));
    if (!(magnitude > 0.f) || !std::isfinite(magnitude)) {
        CV_Error(cv::Error::StsBadArg, "Invalid border length");
    }
    const cv::Point2f axis(vector.x / magnitude, vector.y / magnitude);
    return Geometry{axis, cv::Point2f(-axis.y, axis.x),
                    (border.first + border.second) * .5f};
}
template <typename Value>
std::vector<std::size_t> descending_order(const std::vector<Value>& values) {
    std::vector<std::size_t> order(values.size());
    std::iota(order.begin(), order.end(), std::size_t(0));
    std::sort(order.begin(), order.end(), [&values](std::size_t first, std::size_t second) {
        if (values[first] != values[second]) return values[first] > values[second];
        return first < second;
    });
    return order;
}

std::vector<Border> fit_borders(const std::vector<Border>& segments, int side) {
    std::vector<float> lengths;
    for (const Border& border : segments) lengths.push_back(length(border));
    const std::vector<std::size_t> order = descending_order(lengths);
    const std::size_t count = std::min(std::size_t(128), order.size());
    std::vector<Border> lines;
    std::vector<Geometry> shapes;
    for (std::size_t index = 0; index < count; ++index) {
        lines.push_back(segments[order[index]]);
        shapes.push_back(geometry(lines.back()));
    }
    std::vector<Border> combined;
    combined.reserve(count * 2);
    const float distance_limit = static_cast<float>(side * .01);
    const float extension = static_cast<float>(side * .2);
    for (std::size_t index = 0; index < count; ++index) {
        const Geometry& shape = shapes[index];
        const float own_first = dot(lines[index].first, shape.axis);
        const float own_second = dot(lines[index].second, shape.axis);
        const float own_min = std::min(own_first, own_second);
        const float own_max = std::max(own_first, own_second);
        std::vector<cv::Point2f> points;
        points.reserve(count * 2);
        for (std::size_t other = 0; other < count; ++other) {
            const float agreement = std::abs(dot(shape.axis, shapes[other].axis));
            const float distance = std::abs(dot(shapes[other].center - shape.center, shape.normal));
            const float first = dot(lines[other].first, shape.axis);
            const float second = dot(lines[other].second, shape.axis);
            if (agreement > .993f && distance <= distance_limit &&
                std::min(first, second) <= own_max + extension &&
                std::max(first, second) >= own_min - extension) {
                points.push_back(lines[other].first);
                points.push_back(lines[other].second);
            }
        }
        if (points.empty()) CV_Error(cv::Error::StsBadArg, "Missing border support");
        cv::Vec4f fitted;
        cv::fitLine(points, fitted, cv::DIST_HUBER, 0., .01, .01);
        const cv::Point2f axis(fitted[0], fitted[1]), center(fitted[2], fitted[3]);
        float low = std::numeric_limits<float>::infinity();
        float high = -std::numeric_limits<float>::infinity();
        for (const cv::Point2f& point : points) {
            const float extent = dot(point - center, axis);
            low = std::min(low, extent);
            high = std::max(high, extent);
        }
        combined.push_back(Border{center + axis * low, center + axis * high});
    }
    combined.insert(combined.end(), lines.begin(), lines.end());
    lengths.clear();
    for (const Border& border : combined) lengths.push_back(length(border));
    const std::vector<std::size_t> combined_order = descending_order(lengths);
    std::vector<Border> sorted;
    shapes.clear();
    for (std::size_t index : combined_order) {
        sorted.push_back(combined[index]);
        shapes.push_back(geometry(sorted.back()));
    }
    std::vector<unsigned char> removed(sorted.size(), 0);
    std::vector<Border> selected;
    selected.reserve(64);
    for (std::size_t index = 0; index < sorted.size(); ++index) {
        if (removed[index]) continue;
        selected.push_back(sorted[index]);
        for (std::size_t other = 0; other < sorted.size(); ++other) {
            const float agreement = std::abs(dot(shapes[index].axis, shapes[other].axis));
            const float distance = std::abs(dot(shapes[other].center - shapes[index].center,
                                                shapes[index].normal));
            const float first = dot(sorted[other].first, shapes[index].axis);
            const float second = dot(sorted[other].second, shapes[index].axis);
            const float own_first = dot(sorted[index].first, shapes[index].axis);
            const float own_second = dot(sorted[index].second, shapes[index].axis);
            const bool overlap = std::max(first, second) >= std::min(own_first, own_second) &&
                                 std::min(first, second) <= std::max(own_first, own_second);
            if (agreement >= .999f && distance <= 1.5f && overlap) removed[other] = 1;
        }
        if (selected.size() == 64) break;
    }
    return selected;
}

std::vector<Border> region_borders(const cv::Mat& lab) {
    cv::Mat smooth;
    cv::GaussianBlur(lab, smooth, cv::Size(3, 3), .6);
    std::vector<std::int32_t> counts(1859, 0);
    for (int row = 0; row < smooth.rows; ++row) {
        const cv::Vec3b* pixels = smooth.ptr<cv::Vec3b>(row);
        for (int column = 0; column < smooth.cols; ++column) {
            const cv::Vec3b& pixel = pixels[column];
            const std::int32_t code =
                (static_cast<std::int32_t>(pixel[0]) / 24) * 169 +
                (static_cast<std::int32_t>(pixel[1]) / 20) * 13 +
                static_cast<std::int32_t>(pixel[2]) / 20;
            ++counts[code];
        }
    }
    const std::vector<std::size_t> palette = descending_order(counts);
    std::vector<cv::Vec3i> centers;
    std::vector<Border> segments;
    const int tolerance[3] = {26, 16, 16};
    const int side = std::max(lab.rows, lab.cols);
    const double minimum = side * .08;
    const double population_limit = std::max(40., static_cast<double>(lab.total()) * .004);
    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
    for (std::size_t code : palette) {
        if (counts[code] < population_limit) break;
        const int lightness = static_cast<int>(code) / 169;
        const int remainder = static_cast<int>(code) % 169;
        const cv::Vec3i center(lightness * 24 + 12,
                              (remainder / 13) * 20 + 10,
                              (remainder % 13) * 20 + 10);
        bool duplicate = false;
        for (const cv::Vec3i& previous : centers) {
            if (std::abs(center[0] - previous[0]) < tolerance[0] &&
                std::abs(center[1] - previous[1]) < tolerance[1] &&
                std::abs(center[2] - previous[2]) < tolerance[2]) {
                duplicate = true;
                break;
            }
        }
        if (duplicate) continue;
        centers.push_back(center);
        cv::Scalar lower, upper;
        for (int channel = 0; channel < 3; ++channel) {
            lower[channel] = std::max(0, std::min(255, center[channel] - tolerance[channel]));
            upper[channel] = std::max(0, std::min(255, center[channel] + tolerance[channel]));
        }
        cv::Mat mask;
        cv::inRange(smooth, lower, upper, mask);
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
        std::vector<std::vector<cv::Point> > contours;
        cv::findContours(mask, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
        for (const std::vector<cv::Point>& contour : contours) {
            if (cv::arcLength(contour, true) < side * .35) continue;
            std::vector<cv::Point> polygon;
            cv::approxPolyDP(contour, polygon, 2.5, true);
            for (std::size_t index = 0; index < polygon.size(); ++index) {
                const cv::Point& first = polygon[index];
                const cv::Point& second = polygon[(index + 1) % polygon.size()];
                const double delta_x = static_cast<double>(second.x) - first.x;
                const double delta_y = static_cast<double>(second.y) - first.y;
                if (delta_x * delta_x + delta_y * delta_y >= minimum * minimum) {
                    segments.push_back(Border{
                        cv::Point2f(static_cast<float>(first.x), static_cast<float>(first.y)),
                        cv::Point2f(static_cast<float>(second.x), static_cast<float>(second.y))});
                }
            }
        }
        if (centers.size() == 10) break;
    }
    if (segments.empty()) return std::vector<Border>();
    return fit_borders(segments, side);
}
}

extern "C" int neck_color_borders(const std::uint8_t* lab, int w, int h, float* out) {
    if (!lab || !out || w <= 0 || h <= 0) return -1;
    const std::size_t width = static_cast<std::size_t>(w);
    const std::size_t height = static_cast<std::size_t>(h);
    if (width > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) / height ||
        width > std::numeric_limits<std::size_t>::max() / 3 / height) return -1;
    try {
        const cv::Mat image(h, w, CV_8UC3, const_cast<std::uint8_t*>(lab));
        const std::vector<Border> borders = region_borders(image);
        for (const Border& border : borders) {
            if (!std::isfinite(border.first.x) || !std::isfinite(border.first.y) ||
                !std::isfinite(border.second.x) || !std::isfinite(border.second.y)) return -1;
        }
        for (std::size_t index = 0; index < borders.size(); ++index) {
            out[index * 4] = borders[index].first.x;
            out[index * 4 + 1] = borders[index].first.y;
            out[index * 4 + 2] = borders[index].second.x;
            out[index * 4 + 3] = borders[index].second.y;
        }
        return static_cast<int>(borders.size());
    } catch (...) {
        return -1;
    }
}
