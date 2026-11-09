/*
 * Native scalar port of neck_region._find_neck_regions after HoughLinesP.
 *
 * Build on PYNQ:
 *   g++ -O3 -std=c++11 -ffp-contract=off -fPIC -shared -o libneck_region.so \
 *       native/neck_region.cpp
 *
 * Do not enable -ffast-math: its relaxed comparisons change proposal gates.
 */
#include <algorithm>
#include <cmath>
#include <vector>


extern "C" int tracking_skin_component(const int* labels, const unsigned char* cr,
        const unsigned char* neck, const int* stats, const float* quad, int count, int width, int height,
        float predicted_x, float predicted_y, int right_only, int width_hint, float* out) {
    if (!labels || !cr || !neck || !stats || !quad || !out || count < 1 ||
        width < 1 || height < 1 || width > 160 || height > 120 || count > width*height+1) return 0;
    struct Sum { double x=0, y=0, warm=0; int contact=0; };
    std::vector<Sum> sums(count);
    for (int y=0; y<height; ++y) for (int x=0; x<width; ++x) {
        int p=y*width+x, label=labels[p];
        if (label <= 0 || label >= count) continue;
        sums[label].warm += cr[p];
        if (neck[p]) { sums[label].x += x; sums[label].y += y; ++sums[label].contact; }
    }
    double best=-1;
    for (int i=1; i<count; ++i) {
        const int* s=stats+5*i;
        int area=s[4], w=s[2], h=s[3];
        if (area < 18 || area > width*height*.20 || std::min(w,h) < 1 ||
            double(std::max(w,h))/std::min(w,h) > 5 || double(area)/(w*h) > .92 ||
            sums[i].contact < 8) continue;
        double x=sums[i].x/sums[i].contact, y=sums[i].y/sums[i].contact;
        if (right_only && x < (width-1)*.5) continue;
        double temporal=1;
        if (std::isfinite(predicted_x) && std::isfinite(predicted_y)) {
            double jump=std::hypot(x-predicted_x,y-predicted_y);
            temporal=.5+.5*std::max(0.,1.-jump/std::max(1.,width_hint*1.5));
        }
        bool positive=false, negative=false;
        double distance=1e30;
        for (int edge=0; edge<4; ++edge) {
            int next=(edge+1)%4;
            double ax=quad[2*edge], ay=quad[2*edge+1];
            double dx=quad[2*next]-ax, dy=quad[2*next+1]-ay;
            double cross=dx*(y-ay)-dy*(x-ax);
            positive |= cross > 0; negative |= cross < 0;
            double t=std::max(0.,std::min(1.,((x-ax)*dx+(y-ay)*dy)/std::max(1e-20,dx*dx+dy*dy)));
            distance=std::min(distance,std::hypot(x-ax-t*dx,y-ay-t*dy));
        }
        double locality=positive && negative ? std::max(0.,1.-distance/std::max(1,width_hint)) : 1.;
        double score=std::min(1.,sums[i].contact/180.)*temporal*locality*(sums[i].warm/area/180.);
        if (score <= best) continue;
        best=score;
        out[0]=x; out[1]=y; out[2]=score;
        out[3]=s[0]; out[4]=s[1]; out[5]=s[0]+w-1; out[6]=s[1]+h-1;
    }
    return best >= 0;
}

// Same normalized support gates as FretboardTracker._well_spread. Keeping
// the small sort/hull operation together avoids many ARM NumPy dispatches.
extern "C" int tracking_well_spread(const float* board, int count) {
    if (!board || count < 16 || count > 240) return 0;
    struct XY { double x, y; };
    std::vector<XY> points;
    std::vector<double> xs, ys;
    unsigned cells = 0;
    for (int i = 0; i < count; ++i) {
        double x = board[2*i], y = board[2*i+1];
        if (!std::isfinite(x) || !std::isfinite(y) ||
            x < -.03 || y < -.03 || x > 1.03 || y > 1.03) return 0;
        points.push_back(XY{x,y}); xs.push_back(x); ys.push_back(y);
        int cx = std::max(0, std::min(2, static_cast<int>(x*3)));
        int cy = std::max(0, std::min(1, static_cast<int>(y*2)));
        cells |= 1u << (cx+3*cy);
    }
    std::sort(xs.begin(), xs.end()); std::sort(ys.begin(), ys.end());
    auto quantile = [count](const std::vector<double>& v, double q) {
        double index = q*(count-1);
        int lo = static_cast<int>(index), hi = static_cast<int>(std::ceil(index));
        return v[lo]+(v[hi]-v[lo])*(index-lo);
    };
    int occupied = 0;
    for (unsigned bits = cells; bits; bits >>= 1) occupied += bits & 1;
    if (quantile(xs,.9)-quantile(xs,.1) < .55 ||
        quantile(ys,.9)-quantile(ys,.1) < .45 || occupied < 4) return 0;
    std::sort(points.begin(), points.end(), [](const XY& a, const XY& b) {
        return a.x < b.x || (a.x == b.x && a.y < b.y);
    });
    auto cross = [](const XY& a, const XY& b, const XY& c) {
        return (b.x-a.x)*(c.y-a.y)-(b.y-a.y)*(c.x-a.x);
    };
    std::vector<XY> hull(2*count);
    int k = 0;
    for (const XY& p : points) {
        while (k >= 2 && cross(hull[k-2],hull[k-1],p) <= 0) --k;
        hull[k++] = p;
    }
    for (int i = count-2, lower = k+1; i >= 0; --i) {
        while (k >= lower && cross(hull[k-2],hull[k-1],points[i]) <= 0) --k;
        hull[k++] = points[i];
    }
    double twice_area = 0;
    for (int i = 1; i < k; ++i)
        twice_area += hull[i-1].x*hull[i].y-hull[i].x*hull[i-1].y;
    return std::abs(twice_area)*.5 >= .28;
}

namespace {

struct Point { double x, y; };
struct Segment { Point p[2]; Point vector; double length; };
struct Border { double slope, value; };
struct Proposal { double score; Point quad[4]; };

static double dot(const Point& a, const Point& b) { return a.x * b.x + a.y * b.y; }

static Point add(const Point& a, const Point& b) { return Point{a.x + b.x, a.y + b.y}; }
static Point subtract(const Point& a, const Point& b) { return Point{a.x - b.x, a.y - b.y}; }
static Point scale(const Point& point, double scalar) {
    return Point{point.x * scalar, point.y * scalar};
}

static double linear_quantile(std::vector<double> values, double fraction) {
    std::sort(values.begin(), values.end());
    const double index = fraction * static_cast<double>(values.size() - 1);
    const size_t lower = static_cast<size_t>(std::floor(index));
    const size_t upper = static_cast<size_t>(std::ceil(index));
    return values[lower] + (values[upper] - values[lower]) * (index - lower);
}

static void unique_positions(std::vector<double>* positions) {
    std::sort(positions->begin(), positions->end());
    std::vector<double> unique;
    for (size_t i = 0; i < positions->size(); ++i) {
        if (unique.empty() || (*positions)[i] - unique.back() > 5.0)
            unique.push_back((*positions)[i]);
    }
    positions->swap(unique);
}

static double median_gaps(const std::vector<double>& positions) {
    std::vector<double> gaps;
    for (size_t i = 1; i < positions.size(); ++i)
        gaps.push_back(positions[i] - positions[i - 1]);
    return linear_quantile(gaps, 0.5);
}

static bool in_frame(const Point quad[4], int width, int height) {
    for (int i = 0; i < 4; ++i) {
        if (quad[i].x < 0 || quad[i].y < 0 || quad[i].x > width - 1 || quad[i].y > height - 1)
            return false;
    }
    return true;
}

static double mean_corner_distance(const Point a[4], const Point b[4]) {
    double total = 0;
    for (int i = 0; i < 4; ++i) {
        const Point delta = subtract(a[i], b[i]);
        total += std::sqrt(dot(delta, delta));
    }
    return total / 4.0;
}

static Point from_uv(double u, double v, const Point& axis, const Point& normal) {
    return add(scale(axis, u), scale(normal, v));
}

static void fill_quad(Point quad[4], const Border& lower, const Border& upper,
                      const double ends[2], double center,
                      const Point& axis, const Point& normal) {
    const double us[4] = {ends[0], ends[1], ends[1], ends[0]};
    const Border borders[4] = {lower, lower, upper, upper};
    for (int i = 0; i < 4; ++i)
        quad[i] = from_uv(us[i], borders[i].value + borders[i].slope * (us[i] - center), axis, normal);
}

}  // namespace

extern "C" int neck_border_candidates(const float* segments, int n, int width, int height,
                                      int limit, int require_frets, float* output) {
    if (!segments || !output || n < 0 || n > 4096 || width < 2 || height < 2 ||
        limit < 1 || limit > 6 || (require_frets != 0 && require_frets != 1))
        return 0;

    std::vector<Segment> source;
    source.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        const float* input = segments + 4 * i;
        if (!std::isfinite(input[0]) || !std::isfinite(input[1]) ||
            !std::isfinite(input[2]) || !std::isfinite(input[3]))
            return 0;
        Segment segment = {{ {input[0], input[1]}, {input[2], input[3]} }, {0, 0}, 0};
        segment.vector = subtract(segment.p[1], segment.p[0]);
        segment.length = std::sqrt(dot(segment.vector, segment.vector));
        source.push_back(segment);
    }
    if (source.empty()) return 0;

    std::vector<int> longest(source.size());
    for (size_t i = 0; i < source.size(); ++i) longest[i] = static_cast<int>(i);
    std::stable_sort(longest.begin(), longest.end(), [&source](int left, int right) {
        return source[left].length > source[right].length;
    });
    if (longest.size() > 12) longest.resize(12);

    bool have_best = false;
    Proposal best;
    std::vector<Proposal> proposals;
    for (size_t rank_i = 0; rank_i < longest.size(); ++rank_i) {
        const int i = longest[rank_i];
        if (source[i].length < 110.0) continue;
        Point axis = scale(source[i].vector, 1.0 / source[i].length);
        if (axis.x < 0) axis = scale(axis, -1.0);
        const Point normal = {-axis.y, axis.x};

        std::vector<Point> uv0(source.size()), uv1(source.size()), middle(source.size());
        std::vector<double> slopes(source.size());
        for (size_t k = 0; k < source.size(); ++k) {
            uv0[k] = {dot(source[k].p[0], axis), dot(source[k].p[0], normal)};
            uv1[k] = {dot(source[k].p[1], axis), dot(source[k].p[1], normal)};
            middle[k] = scale(add(uv0[k], uv1[k]), 0.5);
            slopes[k] = dot(source[k].vector, normal) /
                        std::max(std::fabs(dot(source[k].vector, axis)), 1.0);
        }

        for (size_t rank_j = 0; rank_j < longest.size(); ++rank_j) {
            const int j = longest[rank_j];
            if (j <= i || source[j].length < 110.0 || std::fabs(slopes[j]) > 0.12) continue;
            const double lo = std::min(std::min(uv0[i].x, uv1[i].x), std::min(uv0[j].x, uv1[j].x));
            const double hi = std::max(std::max(uv0[i].x, uv1[i].x), std::max(uv0[j].x, uv1[j].x));
            if (hi - lo < 130.0) continue;
            const double center = (lo + hi) / 2.0;
            Border borders[2];
            const int pair[2] = {i, j};
            for (int side = 0; side < 2; ++side) {
                const Point a = uv0[pair[side]], b = uv1[pair[side]];
                borders[side] = {(b.y - a.y) / (b.x - a.x), a.y + (b.y - a.y) / (b.x - a.x) * (center - a.x)};
            }
            if (borders[0].value > borders[1].value) std::swap(borders[0], borders[1]);
            const double board_width = borders[1].value - borders[0].value;
            if (board_width < 18.0 || board_width > 110.0 || (hi - lo) / board_width < 3.0) continue;

            std::vector<int> available_cross;
            std::vector<double> xs;
            for (size_t k = 0; k < source.size(); ++k) {
                bool cross = std::fabs(dot(source[k].vector, axis)) <
                             std::fabs(dot(source[k].vector, normal)) * 0.5 &&
                             source[k].length >= board_width * 0.45;
                const double lower = borders[0].value + borders[0].slope * (middle[k].x - center);
                const double upper = borders[1].value + borders[1].slope * (middle[k].x - center);
                cross = cross && std::min(uv0[k].y, uv1[k].y) <= lower + board_width * 0.3 &&
                                std::max(uv0[k].y, uv1[k].y) >= upper - board_width * 0.3;
                if (!cross) continue;
                available_cross.push_back(static_cast<int>(k));
                if (middle[k].x >= lo - 4.0 && middle[k].x <= hi + 4.0) xs.push_back(middle[k].x);
            }
            unique_positions(&xs);
            if (require_frets && xs.size() < 3) continue;
            if (!xs.empty() && xs.back() - xs.front() < std::max(110.0, board_width * 2.5)) continue;

            bool lattice_confirmed = false;
            if (xs.size() >= 5) {
                std::vector<double> gaps;
                double sum_gaps = 0, sum_scaled = 0;
                for (size_t k = 1; k < xs.size(); ++k) {
                    const double gap = xs[k] - xs[k - 1];
                    gaps.push_back(gap);
                    sum_gaps += gap;
                    sum_scaled += std::pow(2.0, static_cast<double>(k - 1) / 12.0);
                }
                const double mean_gaps = sum_gaps / gaps.size();
                const double mean_scaled = sum_scaled / gaps.size();
                double variance_gaps = 0, variance_scaled = 0;
                for (size_t k = 0; k < gaps.size(); ++k) {
                    const double gap_delta = gaps[k] - mean_gaps;
                    const double scale_delta = std::pow(2.0, static_cast<double>(k) / 12.0) - mean_scaled;
                    variance_gaps += gap_delta * gap_delta;
                    variance_scaled += scale_delta * scale_delta;
                }
                const double gap_std = std::sqrt(variance_gaps / gaps.size());
                const double progression = std::sqrt(variance_scaled / gaps.size()) / mean_scaled;
                if (gap_std / mean_gaps < progression * 0.45 ||
                    *std::max_element(gaps.begin(), gaps.end()) > median_gaps(xs) * 3.0)
                    continue;
                lattice_confirmed = true;
            }

            std::vector<double> all_xs;
            for (size_t k = 0; k < available_cross.size(); ++k)
                all_xs.push_back(middle[available_cross[k]].x);
            unique_positions(&all_xs);
            double first = xs.empty() ? lo : xs.front();
            double last = xs.empty() ? hi : xs.back();
            double gap_limit = xs.size() > 1 ? median_gaps(xs) * 1.8 : hi - lo;
            for (size_t k = 0; k < all_xs.size(); ++k) {
                const double x = all_xs[k];
                if (x <= last + 5.0) continue;
                if (x - last > gap_limit) break;
                gap_limit = (x - last) * 1.5;
                last = x;
            }
            gap_limit = xs.size() > 1 ? median_gaps(xs) * 1.8 : hi - lo;
            for (size_t k = all_xs.size(); k-- > 0;) {
                const double x = all_xs[k];
                if (x >= first - 5.0) continue;
                if (first - x > gap_limit) break;
                gap_limit = (first - x) * 1.5;
                first = x;
            }
            const double ends[2] = {std::min(lo, first - 4.0), std::max(hi, last + 4.0)};
            const double slope = (borders[0].slope + borders[1].slope) * 0.5;
            std::vector<double> transverse_min, transverse_max;
            for (size_t k = 0; k < available_cross.size(); ++k) {
                const int index = available_cross[k];
                const double a = uv0[index].y - slope * (uv0[index].x - center);
                const double b = uv1[index].y - slope * (uv1[index].x - center);
                transverse_min.push_back(std::min(a, b));
                transverse_max.push_back(std::max(a, b));
            }
            double low = transverse_min.empty() ? borders[0].value : linear_quantile(transverse_min, 0.20) - 2.0;
            double high = transverse_max.empty() ? borders[1].value : linear_quantile(transverse_max, 0.80) + 2.0;
            const double tight_low = std::min(borders[0].value - 2.0, low);
            const double tight_high = std::max(borders[1].value + 2.0, high);
            low = std::min(borders[0].value - board_width * 0.6, low);
            high = std::max(borders[1].value + board_width * 0.6, high);
            if (high - low > std::min(180.0, board_width * 5.0)) continue;

            const Border wide_lower = {borders[0].slope, low};
            const Border wide_upper = {borders[1].slope, high};
            Proposal proposal;
            proposal.score = std::max(1.0, static_cast<double>(xs.size())) * (ends[1] - ends[0]) * std::min(board_width, 60.0);
            if (!lattice_confirmed) proposal.score *= 0.35;
            fill_quad(proposal.quad, wide_lower, wide_upper, ends, center, axis, normal);
            if (!in_frame(proposal.quad, width, height)) continue;
            if (!have_best || proposal.score > best.score) {
                best = proposal;
                have_best = true;
            }
            if (limit == 1) continue;
            proposals.push_back(proposal);
            const Border tight_lower = {borders[0].slope, tight_low};
            const Border tight_upper = {borders[1].slope, tight_high};
            Proposal tight = {proposal.score, {}};
            fill_quad(tight.quad, tight_lower, tight_upper, ends, center, axis, normal);
            if (in_frame(tight.quad, width, height)) proposals.push_back(tight);
        }
    }

    if (!have_best) return 0;
    std::vector<Proposal> selected;
    selected.push_back(best);
    std::stable_sort(proposals.begin(), proposals.end(), [](const Proposal& left, const Proposal& right) {
        return left.score > right.score;
    });
    for (size_t i = 0; i < proposals.size() && static_cast<int>(selected.size()) < limit; ++i) {
        bool distinct = true;
        for (size_t old = 0; old < selected.size(); ++old) {
            if (mean_corner_distance(proposals[i].quad, selected[old].quad) <= 7.0) {
                distinct = false;
                break;
            }
        }
        if (distinct) selected.push_back(proposals[i]);
    }
    for (size_t i = 0; i < selected.size(); ++i) {
        for (int corner = 0; corner < 4; ++corner) {
            output[i * 8 + corner * 2] = static_cast<float>(selected[i].quad[corner].x / (width - 1));
            output[i * 8 + corner * 2 + 1] = static_cast<float>(selected[i].quad[corner].y / (height - 1));
        }
    }
    return static_cast<int>(selected.size());
}
