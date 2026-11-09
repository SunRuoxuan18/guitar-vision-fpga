/* Extracted patch statistics only; no allocation, retained state or board IO.
 * gcc -O3 -shared -fPIC -std=c99 native/fingertip_patch_quality.c \
 *     -lm -o libfingertip_patch_quality.so
 */
#include <math.h>
#include <stddef.h>

#ifdef __FAST_MATH__
#error "Finite validation requires compilation without -ffast-math"
#endif

/* Caller supplies count*121 floats per patch buffer, count*2 per coordinate
 * buffer and count*5 doubles of disjoint output. Invalid input leaves output
 * untouched. All reductions use double, including coordinate subtraction. */
#ifdef _WIN32
__declspec(dllexport)
#endif
int fingertip_patch_quality(const float *previous, const float *current,
        const float *anchors, const float *original, const float *returned,
        int count, double *output) {
    if (!previous || !current || !anchors || !original || !returned || !output ||
            count < 1 || count > 4) return -1;
    for (int i = 0; i < count * 121; ++i)
        if (!isfinite(previous[i]) || !isfinite(current[i]) ||
                !isfinite(anchors[i])) return -1;
    for (int i = 0; i < count * 2; ++i)
        if (!isfinite(original[i]) || !isfinite(returned[i])) return -1;

    for (int n = 0; n < count; ++n) {
        int base = n * 121;
        double old_mean = 0., new_mean = 0., anchor_mean = 0.;
        for (int i = 0; i < 121; ++i) {
            old_mean += previous[base + i];
            new_mean += current[base + i];
            anchor_mean += anchors[base + i];
        }
        old_mean /= 121.; new_mean /= 121.; anchor_mean /= 121.;
        double old_ss = 0., new_ss = 0., anchor_ss = 0.;
        double old_new = 0., anchor_new = 0.;
        for (int i = 0; i < 121; ++i) {
            double a = (double)previous[base + i] - old_mean;
            double b = (double)current[base + i] - new_mean;
            double c = (double)anchors[base + i] - anchor_mean;
            old_ss += a * a; new_ss += b * b; anchor_ss += c * c;
            old_new += a * b; anchor_new += c * b;
        }
        double dx = (double)returned[2 * n] - original[2 * n];
        double dy = (double)returned[2 * n + 1] - original[2 * n + 1];
        output[5 * n] = sqrt(dx * dx + dy * dy);
        output[5 * n + 1] = old_ss > 0. && new_ss > 0. ?
            old_new / sqrt(old_ss * new_ss) : -1.;
        output[5 * n + 2] = anchor_ss > 0. && new_ss > 0. ?
            anchor_new / sqrt(anchor_ss * new_ss) : -1.;
        output[5 * n + 3] = sqrt(new_ss / 121.);
        output[5 * n + 4] = sqrt(old_ss / 121.);
    }
    return 0;
}
