#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

extern "C" int neck_color_borders(const std::uint8_t*, int, int, float*);
extern "C" int neck_color_ridges(const std::uint8_t*, int, int, const float*, int, float*);

namespace {
using Quad = std::array<cv::Point2f, 4>;
using DoubleQuad = std::array<cv::Point2d, 4>;

double dot(const cv::Point2d& first, const cv::Point2d& second) {
    return first.x * second.x + first.y * second.y;
}
float magnitude(const cv::Point2f& point) {
    return std::sqrt(point.x * point.x + point.y * point.y);
}
cv::Point2f as_float(const cv::Point2d& point) {
    return cv::Point2f(static_cast<float>(point.x), static_cast<float>(point.y));
}
float color_distance(const cv::Vec3b& first, const cv::Vec3b& second) {
    const float lightness = float(first[0]) - second[0];
    const float channel_a = float(first[1]) - second[1];
    const float channel_b = float(first[2]) - second[2];
    return std::sqrt(lightness * lightness + channel_a * channel_a + channel_b * channel_b);
}
double overlap(const Quad& first, const Quad& second) {
    const std::vector<cv::Point2f> first_points(first.begin(), first.end());
    const std::vector<cv::Point2f> second_points(second.begin(), second.end());
    std::vector<cv::Point2f> intersection;
    const float area = cv::intersectConvexConvex(first_points, second_points, intersection);
    return area / std::max(std::min(cv::contourArea(first_points),
                                    cv::contourArea(second_points)), 1e-9);
}
float median(std::array<float, 40> values) {
    std::sort(values.begin(), values.end());
    return (values[19] + values[20]) * .5f;
}

Quad refine_quad(const cv::Mat& lab, const DoubleQuad& quad) {
    Quad refined;
    for (int index = 0; index < 4; ++index) refined[index] = as_float(quad[index]);
    for (int side = 0; side < 2; ++side) {
        const int first = side == 0 ? 0 : 3;
        const int last = side == 0 ? 1 : 2;
        const cv::Point2d start = quad[first], end = quad[last];
        const cv::Point2d axis = end - start;
        const double length = std::sqrt(dot(axis, axis));
        if (!(length > 0.) || !std::isfinite(length)) throw std::runtime_error("Invalid quad edge");
        const cv::Point2d normal = cv::Point2d(as_float(cv::Point2d(-axis.y, axis.x))) / length;
        std::array<std::array<cv::Point2d, 9>, 48> locations;
        std::array<cv::Mat, 2> colors;
        for (int sign = 0; sign < 2; ++sign) {
            cv::Mat map_x(48, 9, CV_32F), map_y(48, 9, CV_32F);
            const float shift = sign == 0 ? -1.5f : 1.5f;
            for (int sample = 0; sample < 48; ++sample) {
                const cv::Point2d position = start + axis * (.03 + (.97 - .03) * sample / 47.);
                for (int offset = 0; offset < 9; ++offset) {
                    const float distance = -2.f + .5f * offset;
                    const cv::Point2d point = position + normal * distance;
                    locations[sample][offset] = point;
                    const cv::Point2d shifted = point + normal * shift;
                    map_x.at<float>(sample, offset) = static_cast<float>(shifted.x);
                    map_y.at<float>(sample, offset) = static_cast<float>(shifted.y);
                }
            }
            cv::remap(lab, colors[sign], map_x, map_y, cv::INTER_LINEAR, cv::BORDER_REPLICATE);
        }
        std::vector<cv::Point2f> points;
        for (int sample = 0; sample < 48; ++sample) {
            int best = 0;
            float best_strength = -std::numeric_limits<float>::infinity();
            for (int offset = 0; offset < 9; ++offset) {
                const float strength = color_distance(colors[1].at<cv::Vec3b>(sample, offset),
                    colors[0].at<cv::Vec3b>(sample, offset)) - std::abs(-2.f + .5f * offset) * 3.f;
                if (strength > best_strength) { best = offset; best_strength = strength; }
            }
            if (best_strength >= 18.f) points.push_back(as_float(locations[sample][best]));
        }
        if (points.size() < 24) continue;
        cv::Vec4f fitted;
        cv::fitLine(points, fitted, cv::DIST_HUBER, 0., .01, .01);
        const cv::Point2f direction(fitted[0], fitted[1]), center(fitted[2], fitted[3]);
        std::array<cv::Point2f, 2> moved;
        const std::array<cv::Point2f, 2> ends = {{as_float(start), as_float(end)}};
        for (int index = 0; index < 2; ++index) {
            const cv::Point2f relative = ends[index] - center;
            const float extent = relative.x * direction.x + relative.y * direction.y;
            moved[index] = center + direction * extent;
        }
        if (cv::norm(cv::Point2d(moved[0]) - start) <= 2.5 &&
            cv::norm(cv::Point2d(moved[1]) - end) <= 2.5) {
            refined[first] = moved[0]; refined[last] = moved[1];
        }
    }
    return refined;
}

Quad extend_quad(const cv::Mat& lab, const Quad& quad) {
    cv::Point2f axis = (quad[1] + quad[2] - quad[0] - quad[3]) * .5f;
    const float length = magnitude(axis);
    if (!(length > 0.f) || !std::isfinite(length)) throw std::runtime_error("Invalid quad axis");
    axis /= length;
    const cv::Point2f normal(-axis.y, axis.x);
    std::array<float, 160> parameters;
    std::array<bool, 160> valid, support;
    cv::Mat map_x(4, 160, CV_32F), map_y(4, 160, CV_32F), colors;
    for (int sample = 0; sample < 160; ++sample) {
        const float parameter = static_cast<float>(-.8 + (1.8 - (-.8)) * sample / 159.);
        parameters[sample] = parameter;
        const cv::Point2f upper = quad[0] + (quad[1] - quad[0]) * parameter;
        const cv::Point2f lower = quad[3] + (quad[2] - quad[3]) * parameter;
        const float distance = std::max(1.5f, std::min(4.f, magnitude(lower - upper) * .14f));
        const std::array<cv::Point2f, 4> points = {{upper - normal * distance, upper + normal * distance,
                                                  lower - normal * distance, lower + normal * distance}};
        valid[sample] = true;
        for (int row = 0; row < 4; ++row) {
            const cv::Point2f& point = points[row];
            map_x.at<float>(row, sample) = point.x;
            map_y.at<float>(row, sample) = point.y;
            valid[sample] = valid[sample] && point.x >= 1 && point.x < lab.cols - 1 &&
                                             point.y >= 1 && point.y < lab.rows - 1;
        }
    }
    cv::remap(lab, colors, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    for (int sample = 0; sample < 160; ++sample) {
        support[sample] = valid[sample] &&
            color_distance(colors.at<cv::Vec3b>(0, sample), colors.at<cv::Vec3b>(1, sample)) >= 18 &&
            color_distance(colors.at<cv::Vec3b>(3, sample), colors.at<cv::Vec3b>(2, sample)) >= 18 &&
            color_distance(colors.at<cv::Vec3b>(1, sample), colors.at<cv::Vec3b>(2, sample)) <= 55;
    }
    float bounds[2] = {0.f, 1.f};
    for (int side = 0; side < 2; ++side) {
        const int step = side == 0 ? -1 : 1;
        const int initial = static_cast<int>(std::lower_bound(parameters.begin(), parameters.end(),
                                              side == 0 ? 0.f : 1.f) - parameters.begin()) - side;
        int misses = 0, selected = initial;
        for (int index = initial; index >= 0 && index < 160; index += step) {
            if (!valid[index]) break;
            misses = support[index] ? 0 : misses + 1;
            if (misses >= 6) break;
            if (support[index]) selected = index;
        }
        bounds[side] = side == 0 ? std::min(0.f, parameters[selected]) : std::max(1.f, parameters[selected]);
    }
    return Quad{{quad[0] + (quad[1] - quad[0]) * bounds[0],
                 quad[0] + (quad[1] - quad[0]) * bounds[1],
                 quad[3] + (quad[2] - quad[3]) * bounds[1],
                 quad[3] + (quad[2] - quad[3]) * bounds[0]}};
}

struct Candidate {
    cv::Point2d axis, normal;
    double low, high, mid, separation;
    double slopes[2], offsets[2];
    double balance = 0., inside = 0., score = 0.;
    float strength = 0.f;
};

struct Crossbars {
    bool supported = false;
    int count = 0, run = 0;
    float span = 0.f;
    std::array<float, 42> positions{};
    std::array<float, 42> support{}, strength{};
};

float median_edge(const std::array<float, 3>& values, int count) {
    std::array<float, 3> sorted = values;
    std::sort(sorted.begin(), sorted.begin() + count);
    return count == 2 ? (sorted[0] + sorted[1]) * .5f : sorted[count / 2];
}

struct RegularRun {
    int start = 0;
    int count = 0;
};

RegularRun regular_crossbar_run(const Crossbars& bars) {
    RegularRun best;
    float best_span = 0.f;
    for (int start = 0; start + 4 < bars.count; ++start) {
        float previous_gap = 0.f, smallest_gap = 1.f, largest_gap = 0.f;
        for (int end = start + 1; end < bars.count; ++end) {
            const float gap = bars.positions[end] - bars.positions[end - 1];
            if (previous_gap > 0.f) {
                const float ratio = gap / previous_gap;
                if (ratio < .60f || ratio > 1.60f) break;
            }
            previous_gap = gap;
            smallest_gap = std::min(smallest_gap, gap);
            largest_gap = std::max(largest_gap, gap);
            if (largest_gap > smallest_gap * 2.2f) break;
            const int count = end - start + 1;
            const float span = bars.positions[end] - bars.positions[start];
            if (count >= 5 && span >= .25f &&
                (count > best.count || (count == best.count && span > best_span))) {
                best.start = start;
                best.count = count;
                best_span = span;
            }
        }
    }
    return best;
}

int headstock_side(const Crossbars& bars, const RegularRun& run) {
    if (!bars.supported || run.count < 6) return -1;
    const int count = run.count - 1;
    float sum_x = 0.f, sum_y = 0.f, sum_xx = 0.f, sum_xy = 0.f;
    for (int index = 0; index < count; ++index) {
        const int position = run.start + index;
        const float gap = bars.positions[position + 1] - bars.positions[position];
        sum_x += index;
        sum_y += gap;
        sum_xx += index * index;
        sum_xy += index * gap;
    }
    const float denominator = count * sum_xx - sum_x * sum_x;
    const float mean_gap = sum_y / count;
    if (denominator <= 0.f || mean_gap <= 0.f) return -1;
    const float slope = (count * sum_xy - sum_x * sum_y) / denominator;
    const float relative_slope = slope / mean_gap;
    // Fret spacing shrinks towards the nut under perspective projection.
    if (relative_slope < -.0075f) return 1;
    if (relative_slope > .0075f) return 0;
    return -1;
}

int nut_ridge_boundary(const Crossbars& bars, int headstock) {
    if (headstock < 0) return -1;
    const RegularRun run = regular_crossbar_run(bars);
    if (run.count < 5) return -1;
    const float first = bars.positions[run.start];
    const float last = bars.positions[run.start + run.count - 1];
    if (headstock == 1) {
        int selected = -1;
        for (int index = run.start + run.count - 1; index < bars.count; ++index) {
            if (bars.positions[index] >= last - 1e-4f &&
                bars.support[index] >= .52f && bars.strength[index] >= 18.f) {
                selected = index;
            }
        }
        return selected;
    }
    for (int index = run.start; index >= 0; --index) {
        if (bars.positions[index] <= first + 1e-4f &&
            bars.support[index] >= .52f && bars.strength[index] >= 18.f) {
            return index;
        }
    }
    return -1;
}

std::pair<float, float> crossbar_limits(const Crossbars& bars) {
    const RegularRun run = regular_crossbar_run(bars);
    if (run.count < 5) {
        return {bars.positions[0], bars.positions[bars.count - 1]};
    }
    const float first = bars.positions[run.start];
    const float last = bars.positions[run.start + run.count - 1];
    const int headstock = headstock_side(bars, run);
    const int edge_count = std::min(3, run.count - 1);
    std::array<float, 3> leading{{0.f, 0.f, 0.f}}, trailing{{0.f, 0.f, 0.f}};
    for (int index = 0; index < edge_count; ++index) {
        leading[index] = bars.positions[run.start + index + 1] -
                         bars.positions[run.start + index];
        const int end = run.start + run.count - 1 - index;
        trailing[index] = bars.positions[end] - bars.positions[end - 1];
    }
    const float leading_gap = median_edge(leading, edge_count);
    const float trailing_gap = median_edge(trailing, edge_count);
    if (headstock == 1) {
        float start = std::max(0.f, bars.positions[0]);
        float end = last;
        const int boundary = nut_ridge_boundary(bars, headstock);
        if (boundary >= 0 && bars.positions[boundary] >= end) {
            return {start, std::min(1.f, bars.positions[boundary])};
        }
        const float nut_limit = std::min(1.f, last + trailing_gap * 1.5f);
        int used_occlusion_bridge = 0;
        for (int index = run.start + run.count; index < bars.count; ++index) {
            const float candidate = bars.positions[index];
            if (candidate > nut_limit + 1e-4f) break;
            end = candidate;
        }
        if (end == last) {
            for (int index = run.start + run.count; index < bars.count; ++index) {
                const float candidate = bars.positions[index];
                if (candidate - end > trailing_gap * 2.8f) break;
                end = candidate;
                if (++used_occlusion_bridge >= 2) break;
            }
        }
        return {start, end};
    }
    if (headstock == 0) {
        float start = first;
        const float end = std::min(1.f, bars.positions[bars.count - 1]);
        const int boundary = nut_ridge_boundary(bars, headstock);
        if (boundary >= 0 && bars.positions[boundary] <= start) {
            return {std::max(0.f, bars.positions[boundary]), end};
        }
        const float nut_limit = std::max(0.f, first - leading_gap * 1.5f);
        int used_occlusion_bridge = 0;
        for (int index = run.start - 1; index >= 0; --index) {
            const float candidate = bars.positions[index];
            if (candidate < nut_limit - 1e-4f) break;
            start = candidate;
        }
        if (start == first) {
            for (int index = run.start - 1; index >= 0; --index) {
                const float candidate = bars.positions[index];
                if (start - candidate > leading_gap * 2.8f) break;
                start = candidate;
                if (++used_occlusion_bridge >= 2) break;
            }
        }
        return {start, end};
    }
    return {first, last};
}

Crossbars crossbars(const std::uint8_t* gray, int width, int height, const Quad& quad) {
    std::array<float, 8> points;
    for (int index = 0; index < 4; ++index) {
        points[2 * index] = quad[index].x; points[2 * index + 1] = quad[index].y;
    }
    std::array<float, 259> result{};
    const int supported = neck_color_ridges(gray, width, height, points.data(), 1, result.data());
    if ((supported != 0 && supported != 1) || !std::isfinite(result[0]) ||
        result[0] < 0 || result[0] > 42 || std::floor(result[0]) != result[0] ||
        !std::isfinite(result[1]) || result[1] < 0 || result[1] > result[0] ||
        std::floor(result[1]) != result[1] || !std::isfinite(result[2])) {
        throw std::runtime_error("Invalid native ridge result");
    }
    Crossbars bars;
    bars.supported = supported != 0; bars.count = static_cast<int>(result[0]);
    bars.run = static_cast<int>(result[1]); bars.span = result[2];
    if (bars.supported && bars.count == 0) throw std::runtime_error("Missing ridge positions");
    for (int index = 0; index < bars.count; ++index) {
        const float position = result[3 + index];
        const float support = result[64 + index], strength = result[106 + index];
        if (!std::isfinite(position) || position < 0 || position > 1 ||
            (index && position < bars.positions[index - 1]) ||
            !std::isfinite(support) || support < 0 || support > 1 ||
            !std::isfinite(strength) || strength < 0) {
            throw std::runtime_error("Invalid ridge position");
        }
        bars.positions[index] = position;
        bars.support[index] = support;
        bars.strength[index] = strength;
    }
    return bars;
}

std::vector<Candidate> pair_borders(const float* lines, int count, const cv::Mat& lab) {
    std::vector<cv::Point2f> axes;
    for (int index = 0; index < count; ++index) {
        cv::Point2f axis(lines[4 * index + 2] - lines[4 * index],
                         lines[4 * index + 3] - lines[4 * index + 1]);
        const float length = magnitude(axis);
        if (!(length > 0.f) || !std::isfinite(length)) throw std::runtime_error("Invalid border axis");
        axis /= length;
        if (axis.x < 0 || (axis.x == 0 && axis.y < 0)) axis *= -1.f;
        axes.push_back(axis);
    }
    std::vector<Candidate> candidates;
    for (int first = 0; first < count; ++first) {
        for (int second = first + 1; second < count; ++second) {
            const float agreement = axes[first].x * axes[second].x + axes[first].y * axes[second].y;
            if (!(std::abs(agreement) > .992f)) continue;
            Candidate candidate;
            candidate.axis = cv::Point2d(axes[first]) + cv::Point2d(axes[second]) * (agreement >= 0 ? 1. : -1.);
            candidate.axis /= std::sqrt(dot(candidate.axis, candidate.axis));
            candidate.normal = cv::Point2d(-candidate.axis.y, candidate.axis.x);
            double projections[2][2], transverse[2][2];
            const int indices[2] = {first, second};
            for (int side = 0; side < 2; ++side) {
                for (int end = 0; end < 2; ++end) {
                    const int offset = 4 * indices[side] + 2 * end;
                    const cv::Point2d point(lines[offset], lines[offset + 1]);
                    projections[side][end] = dot(point, candidate.axis);
                    transverse[side][end] = dot(point, candidate.normal);
                }
            }
            const double first_low = std::min(projections[0][0], projections[0][1]);
            const double first_high = std::max(projections[0][0], projections[0][1]);
            const double second_low = std::min(projections[1][0], projections[1][1]);
            const double second_high = std::max(projections[1][0], projections[1][1]);
            candidate.low = std::min(first_low, second_low);
            candidate.high = std::max(first_high, second_high);
            candidate.mid = (candidate.low + candidate.high) * .5;
            const double common = std::min(first_high, second_high) - std::max(first_low, second_low);
            for (int side = 0; side < 2; ++side) {
                const double denominator = projections[side][1] - projections[side][0];
                if (std::abs(denominator) < 1e-12) throw std::runtime_error("Invalid border projection");
                candidate.slopes[side] = (transverse[side][1] - transverse[side][0]) / denominator;
                candidate.offsets[side] = transverse[side][0] + candidate.slopes[side] *
                                         (candidate.mid - projections[side][0]);
            }
            if (candidate.offsets[0] > candidate.offsets[1]) {
                std::swap(candidate.slopes[0], candidate.slopes[1]);
                std::swap(candidate.offsets[0], candidate.offsets[1]);
            }
            candidate.separation = candidate.offsets[1] - candidate.offsets[0];
            const double first_width = candidate.separation + (candidate.slopes[1] - candidate.slopes[0]) *
                                                             (candidate.low - candidate.mid);
            const double last_width = candidate.separation + (candidate.slopes[1] - candidate.slopes[0]) *
                                                            (candidate.high - candidate.mid);
            const double length = candidate.high - candidate.low;
            if (candidate.separation >= lab.cols * .018 && candidate.separation <= lab.cols * .19 &&
                length >= lab.cols * .28 && length / std::max(candidate.separation, 1.) >= 4 &&
                common >= lab.cols * .09 && std::min(first_width, last_width) >= lab.cols * .015 &&
                std::max(first_width, last_width) <= lab.cols * .22) candidates.push_back(candidate);
        }
    }
    std::stable_sort(candidates.begin(), candidates.end(), [](const Candidate& first, const Candidate& second) {
        return first.high - first.low > second.high - second.low;
    });
    if (candidates.size() > 96) candidates.resize(96);
    std::vector<Candidate> supported;
    for (Candidate& candidate : candidates) {
        cv::Mat map_x(5, 40, CV_32F), map_y(5, 40, CV_32F), colors;
        std::array<bool, 40> valid;
        const double distance = std::max(1.5, std::min(4., candidate.separation * .14));
        for (int sample = 0; sample < 40; ++sample) {
            const double parameter = candidate.low + (candidate.high - candidate.low) *
                                                      (.04 + (.96 - .04) * sample / 39.);
            const double upper = candidate.offsets[0] + candidate.slopes[0] * (parameter - candidate.mid);
            const double lower = candidate.offsets[1] + candidate.slopes[1] * (parameter - candidate.mid);
            const double offsets[5] = {upper - distance, upper + distance, (upper + lower) * .5,
                                       lower - distance, lower + distance};
            valid[sample] = true;
            for (int row = 0; row < 5; ++row) {
                const cv::Point2d point = candidate.axis * parameter + candidate.normal * offsets[row];
                valid[sample] = valid[sample] && point.x >= 1 && point.x < lab.cols - 1 &&
                                                 point.y >= 1 && point.y < lab.rows - 1;
                map_x.at<float>(row, sample) = static_cast<float>(point.x);
                map_y.at<float>(row, sample) = static_cast<float>(point.y);
            }
        }
        cv::remap(lab, colors, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_REPLICATE);
        std::array<float, 40> delta_first, delta_second;
        int valid_count = 0, first_count = 0, second_count = 0, inside_count = 0;
        for (int sample = 0; sample < 40; ++sample) {
            delta_first[sample] = color_distance(colors.at<cv::Vec3b>(0, sample), colors.at<cv::Vec3b>(1, sample));
            delta_second[sample] = color_distance(colors.at<cv::Vec3b>(4, sample), colors.at<cv::Vec3b>(3, sample));
            const float interior = color_distance(colors.at<cv::Vec3b>(1, sample), colors.at<cv::Vec3b>(3, sample));
            if (!valid[sample]) continue;
            ++valid_count;
            first_count += delta_first[sample] >= 18;
            second_count += delta_second[sample] >= 18;
            inside_count += interior <= 45;
        }
        candidate.balance = double(std::min(first_count, second_count)) / std::max(valid_count, 1);
        candidate.inside = double(inside_count) / std::max(valid_count, 1);
        candidate.strength = std::min(median(delta_first), median(delta_second));
        candidate.score = (candidate.high - candidate.low) * candidate.balance * candidate.inside *
                          std::min(candidate.strength, 70.f);
        if (valid_count >= 20 && candidate.balance >= .30 && candidate.inside >= .30) supported.push_back(candidate);
    }
    std::stable_sort(supported.begin(), supported.end(), [](const Candidate& first, const Candidate& second) {
        return first.score > second.score;
    });
    if (supported.size() > 32) supported.resize(32);
    return supported;
}

bool clipped_quad(const Candidate& candidate, int width, int height, DoubleQuad& quad) {
    double start = candidate.low, end = candidate.high;
    for (int side = 0; side < 2; ++side) {
        const cv::Point2d direction = candidate.axis + candidate.normal * candidate.slopes[side];
        const cv::Point2d origin = candidate.normal *
            (candidate.offsets[side] - candidate.slopes[side] * candidate.mid);
        const double directions[2] = {direction.x, direction.y};
        const double origins[2] = {origin.x, origin.y};
        const double maximums[2] = {double(width - 1), double(height - 1)};
        for (int dimension = 0; dimension < 2; ++dimension) {
            if (std::abs(directions[dimension]) < 1e-8) continue;
            const double first = -origins[dimension] / directions[dimension];
            const double last = (maximums[dimension] - origins[dimension]) / directions[dimension];
            start = std::max(start, std::min(first, last));
            end = std::min(end, std::max(first, last));
        }
    }
    if (end - start < width * .28) return false;
    const double parameters[4] = {start, end, end, start};
    const int sides[4] = {0, 0, 1, 1};
    for (int index = 0; index < 4; ++index) {
        const int side = sides[index];
        const double transverse = candidate.offsets[side] + candidate.slopes[side] *
                                                            (parameters[index] - candidate.mid);
        cv::Point2d point = candidate.axis * parameters[index] + candidate.normal * transverse;
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || point.x < -1e-4 || point.y < -1e-4 ||
            point.x > width - 1 + 1e-4 || point.y > height - 1 + 1e-4) return false;
        point.x = std::max(0., std::min(double(width - 1), point.x));
        point.y = std::max(0., std::min(double(height - 1), point.y));
        quad[index] = point;
    }
    return true;
}

bool tracker_geometry(const Quad& normalized) {
    std::vector<cv::Point2f> points;
    for (const cv::Point2f& point : normalized) points.emplace_back(point.x * 319.f, point.y * 199.f);
    if (!cv::isContourConvex(points)) return false;
    float shortest = std::numeric_limits<float>::infinity(), longest = 0.f;
    for (int index = 0; index < 4; ++index) {
        const float length = magnitude(points[index] - points[(index + 3) % 4]);
        shortest = std::min(shortest, length); longest = std::max(longest, length);
    }
    const double area = cv::contourArea(points);
    return shortest >= 8 && area >= 319 * 199 * .002 && longest > 0 && area / longest >= 8;
}

int candidates_impl(const std::uint8_t* lab, const std::uint8_t* gray_small, int sw, int sh,
                    const std::uint8_t* gray, int width, int height, int limit,
                    bool require_crossbars, float* output) {
    std::array<float, 256> borders{};
    const int count = neck_color_borders(lab, sw, sh, borders.data());
    if (count < 0 || count > 64) throw std::runtime_error("Invalid native border count");
    for (int index = 0; index < count * 4; ++index) {
        if (!std::isfinite(borders[index])) throw std::runtime_error("Invalid native border coordinate");
    }
    if (count < 2) return 0;
    const cv::Mat colors(sh, sw, CV_8UC3, const_cast<std::uint8_t*>(lab));
    const std::vector<Candidate> candidates = pair_borders(borders.data(), count, colors);
    const float scale_x = float(width) / sw, scale_y = float(height) / sh;
    std::vector<Quad> examined, proposals;
    std::vector<std::array<float, 64> > records;
    for (const Candidate& candidate : candidates) {
        DoubleQuad initial;
        if (!clipped_quad(candidate, sw, sh, initial)) continue;
        Quad initial_float;
        for (int index = 0; index < 4; ++index) initial_float[index] = as_float(initial[index]);
        bool duplicate = false;
        for (const Quad& previous : examined) if (overlap(initial_float, previous) > .97) duplicate = true;
        if (duplicate) continue;
        examined.push_back(initial_float);
        if (examined.size() > 16) break;
        const Quad quad = extend_quad(colors, refine_quad(colors, initial));
        Quad full, normalized;
        bool valid = true;
        for (int index = 0; index < 4; ++index) {
            full[index] = cv::Point2f((quad[index].x + .5f) * scale_x - .5f,
                                      (quad[index].y + .5f) * scale_y - .5f);
            normalized[index] = cv::Point2f(full[index].x / float(width - 1), full[index].y / float(height - 1));
            const cv::Point2f& point = normalized[index];
            valid = valid && std::isfinite(point.x) && std::isfinite(point.y) &&
                    point.x >= 0 && point.x <= 1 && point.y >= 0 && point.y <= 1;
        }
        if (!valid) continue;
        duplicate = false;
        for (const Quad& previous : proposals) if (overlap(normalized, previous) > .8) duplicate = true;
        if (duplicate) continue;
        Crossbars bars = crossbars(gray, width, height, full);
        if (!bars.supported) bars = crossbars(gray_small, sw, sh, quad);
        if (require_crossbars && !bars.supported) continue;
        if (bars.supported) {
            const std::pair<float, float> limits = crossbar_limits(bars);
            const double start = limits.first, end = limits.second;
            const Quad previous = normalized;
            normalized = Quad{{previous[0] * static_cast<float>(1. - start) + previous[1] * static_cast<float>(start),
                               previous[0] * static_cast<float>(1. - end) + previous[1] * static_cast<float>(end),
                               previous[3] * static_cast<float>(1. - end) + previous[2] * static_cast<float>(end),
                               previous[3] * static_cast<float>(1. - start) + previous[2] * static_cast<float>(start)}};
        }
        if (!tracker_geometry(normalized)) continue;
        std::array<float, 64> record{};
        for (int index = 0; index < 4; ++index) {
            record[2 * index] = normalized[index].x; record[2 * index + 1] = normalized[index].y;
        }
        record[8] = static_cast<float>(candidate.score); record[9] = static_cast<float>(candidate.balance);
        record[10] = static_cast<float>(candidate.inside); record[11] = candidate.strength;
        record[12] = bars.supported ? 1.f : 0.f; record[13] = float(bars.count);
        record[14] = float(bars.run); record[15] = bars.span;
        for (int index = 0; index < bars.count; ++index) record[16 + index] = bars.positions[index];
        records.push_back(record); proposals.push_back(normalized);
        if (static_cast<int>(proposals.size()) == limit) break;
    }
    for (std::size_t index = 0; index < records.size(); ++index) {
        std::copy(records[index].begin(), records[index].end(), output + index * 64);
    }
    return static_cast<int>(records.size());
}
}

extern "C" int neck_color_candidates(const std::uint8_t* lab, const std::uint8_t* gray_small,
        int sw, int sh, const std::uint8_t* gray, int w, int h, int limit,
        int require_crossbars, float* output) {
    if (!lab || !gray_small || !gray || !output || sw < 24 || sh < 24 || sw > 240 || sh > 180 ||
        w < 24 || h < 24 || w > 4096 || h > 4096 || limit < 1 || limit > 6 ||
        (require_crossbars != 0 && require_crossbars != 1)) return -1;
    try {
        return candidates_impl(lab, gray_small, sw, sh, gray, w, h, limit,
                               require_crossbars != 0, output);
    } catch (...) {
        return -1;
    }
}
