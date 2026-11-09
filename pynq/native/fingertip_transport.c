/* PS transport only. All learned multiply-accumulates remain in hand_conv_q16.
 * ARMv7 build (from pynq/):
 * gcc -O3 -shared -fPIC -std=c99 -mfpu=neon -mfloat-abi=hard \
 *     -fno-fast-math -ffp-contract=off native/fingertip_transport.c \
 *     -pthread -lm -o libfingertip_transport.so
 */
#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stddef.h>
#include <float.h>
#include <math.h>
#include <fenv.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <errno.h>
#ifdef __FAST_MATH__
#error "FP32 quantization requires -fno-fast-math and -ffp-contract=off"
#endif
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

/* ctypes releases the GIL for the whole wait instead of reacquiring it on
 * every MMIO poll. The caller retains core ownership and quarantines on error. */
int fingertip_wait_done(volatile uint32_t *control, double timeout_seconds) {
    if (!control || !isfinite(timeout_seconds) || timeout_seconds <= 0.) return -1;
    struct timespec started, now, pause = {0, 50000};
    if (clock_gettime(CLOCK_MONOTONIC, &started)) return -1;
    for (;;) {
        if (*control & 2u) return 0;
        if (clock_gettime(CLOCK_MONOTONIC, &now)) return -1;
        double elapsed = (double)(now.tv_sec-started.tv_sec) +
                         (double)(now.tv_nsec-started.tv_nsec)*1e-9;
        if (elapsed > timeout_seconds) return 1;
        if (nanosleep(&pause, NULL) && errno != EINTR) return -1;
    }
}

static float converted(int64_t sum, float factor, float bias) {
    float value = (float)sum;
    value = value * factor;
    value = value + bias;
    return value;
}

static float clipped(float value) {
    return fminf(6.f, fmaxf(0.f, value));
}

static int16_t rounded(float value) {
    value = fminf(32767.f, fmaxf(0.f, value));
    int32_t integer = (int32_t)value;
    float fraction = value - (float)integer;
    return (int16_t)(integer + (fraction > .5f ||
                               (fraction == .5f && (integer & 1))));
}

#ifdef __ARM_NEON
static float32x4_t converted_four(const int64_t *src, float factor, float bias) {
    /* Match libhand_q16's vector int64 conversion, including its rounding
     * for values outside int32. Scalar pixel tails use the C cast above. */
    int32x4x2_t words = vld2q_s32((const int32_t *)src);
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    int32x4_t low = words.val[1], high = words.val[0];
#else
    int32x4_t low = words.val[0], high = words.val[1];
#endif
    uint32x4_t fits = vceqq_s32(high, vshrq_n_s32(low, 31));
    uint32x2_t pair = vpmin_u32(vget_low_u32(fits), vget_high_u32(fits));
    float32x4_t value;
    if ((vget_lane_u32(pair, 0) & vget_lane_u32(pair, 1)) == UINT32_MAX) {
        value = vcvtq_f32_s32(low);
    } else {
        uint32x4_t sign = vreinterpretq_u32_s32(vshrq_n_s32(high, 31));
        uint32x4_t lower = vreinterpretq_u32_s32(low);
        uint32x4_t ml = vsubq_u32(veorq_u32(lower, sign), sign);
        uint32x4_t mh = vsubq_u32(veorq_u32(vreinterpretq_u32_s32(high), sign),
            vandq_u32(sign, vceqq_u32(lower, vdupq_n_u32(0))));
        value = vaddq_f32(vmulq_n_f32(vcvtq_f32_u32(mh), 4294967296.f),
                         vcvtq_f32_u32(ml));
        value = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(value),
            vandq_u32(sign, vdupq_n_u32(0x80000000u))));
    }
    return vaddq_f32(vmulq_n_f32(value, factor), vdupq_n_f32(bias));
}

static int16x4_t quantized_four(float32x4_t value, float32x4_t reciprocal) {
    float32x4_t quotient = vminq_f32(vmulq_f32(value, reciprocal),
                                    vdupq_n_f32(32767.f));
    int32x4_t integer = vcvtq_s32_f32(quotient);
    float32x4_t fraction = vsubq_f32(quotient, vcvtq_f32_s32(integer));
    uint32x4_t increment = vorrq_u32(vcgtq_f32(fraction, vdupq_n_f32(.5f)),
        vandq_u32(vceqq_f32(fraction, vdupq_n_f32(.5f)),
                  vtstq_s32(integer, vdupq_n_s32(1))));
    /* Preserve hand_q16_pack's reciprocal multiplication and ties-to-even. */
    return vmovn_s32(vsubq_s32(integer, vreinterpretq_s32_u32(increment)));
}

static void store_four(int16_t *dst, int padded, int16x4_t q[8]) {
    int16x4x2_t a = vzip_s16(q[0], q[1]), b = vzip_s16(q[2], q[3]);
    int16x4x2_t d = vzip_s16(q[4], q[5]), e = vzip_s16(q[6], q[7]);
    int32x2x2_t ab0 = vzip_s32(vreinterpret_s32_s16(a.val[0]), vreinterpret_s32_s16(b.val[0]));
    int32x2x2_t ab1 = vzip_s32(vreinterpret_s32_s16(a.val[1]), vreinterpret_s32_s16(b.val[1]));
    int32x2x2_t de0 = vzip_s32(vreinterpret_s32_s16(d.val[0]), vreinterpret_s32_s16(e.val[0]));
    int32x2x2_t de1 = vzip_s32(vreinterpret_s32_s16(d.val[1]), vreinterpret_s32_s16(e.val[1]));
    vst1q_s16(dst, vcombine_s16(vreinterpret_s16_s32(ab0.val[0]), vreinterpret_s16_s32(de0.val[0])));
    vst1q_s16(dst+padded, vcombine_s16(vreinterpret_s16_s32(ab0.val[1]), vreinterpret_s16_s32(de0.val[1])));
    vst1q_s16(dst+2*padded, vcombine_s16(vreinterpret_s16_s32(ab1.val[0]), vreinterpret_s16_s32(de1.val[0])));
    vst1q_s16(dst+3*padded, vcombine_s16(vreinterpret_s16_s32(ab1.val[1]), vreinterpret_s16_s32(de1.val[1])));
}
#endif

typedef struct {
    const int64_t *src;
    float *scratch;
    const float *factors, *bias;
    int first, last, pixels;
    float maximum;
    int status;
} convert_job;

static void *convert_worker(void *argument) {
    convert_job *job = (convert_job *)argument;
    const int64_t *src = job->src;
    const float *factors = job->factors, *bias = job->bias;
    int pixels = job->pixels;
    float maximum = 0.f;
    job->status = -1;
    for (int c = job->first; c < job->last; ++c) {
        const int64_t *plane = src+(size_t)c*pixels;
        float *output = job->scratch+(size_t)c*pixels;
        int p = 0;
#ifdef __ARM_NEON
        float32x4_t vmax = vdupq_n_f32(0.f);
        uint32x4_t finite = vdupq_n_u32(UINT32_MAX);
        for (; p+4 <= pixels; p += 4) {
            float32x4_t value = converted_four(plane+p, factors[c], bias[c]);
            finite = vandq_u32(finite, vcltq_f32(vabsq_f32(value), vdupq_n_f32(INFINITY)));
            value = vminq_f32(vmaxq_f32(value, vdupq_n_f32(0.f)), vdupq_n_f32(6.f));
            vst1q_f32(output+p, value);
            vmax = vmaxq_f32(vmax, value);
        }
        float maxima[4]; uint32_t checked[4];
        vst1q_f32(maxima, vmax); vst1q_u32(checked, finite);
        for (int i = 0; i < 4; ++i) {
            if (!checked[i]) return NULL;
            maximum = fmaxf(maximum, maxima[i]);
        }
#endif
        for (; p < pixels; ++p) {
            float value = converted(plane[p], factors[c], bias[c]);
            if (!isfinite(value)) return NULL;
            output[p] = clipped(value);
            maximum = fmaxf(maximum, output[p]);
        }
    }
    job->maximum = maximum;
    job->status = 0;
    return NULL;
}

typedef struct {
    const float *src;
    int16_t *dst;
    int channels, pixels, first, last;
    float scale;
} pack_job;

typedef struct {
    const int64_t *src;
    int16_t *dst;
    const float *factors, *bias;
    int channels, pixels, first, last;
    float scale;
} direct_job;

static void *direct_worker(void *argument) {
    direct_job *job = (direct_job *)argument;
    int padded = (job->channels+7)/8*8;
#ifdef __ARM_NEON
    float32x4_t divisor = vdupq_n_f32(job->scale);
    float32x4_t reciprocal = vrecpeq_f32(divisor);
    reciprocal = vmulq_f32(vrecpsq_f32(divisor, reciprocal), reciprocal);
    reciprocal = vmulq_f32(vrecpsq_f32(divisor, reciprocal), reciprocal);
#endif
    for (int block = job->first; block < job->last; block += 16) {
        int end = block+16 < job->last ? block+16 : job->last;
        for (int c = 0; c < padded; c += 8) {
            int p = block;
#ifdef __ARM_NEON
            for (; p+4 <= end; p += 4) {
                int16x4_t q[8];
                for (int lane = 0; lane < 8; ++lane) {
                    int ch = c+lane;
                    q[lane] = vdup_n_s16(0);
                    if (ch < job->channels) {
                        float32x4_t value = converted_four(job->src+(size_t)ch*job->pixels+p,
                                                          job->factors[ch], job->bias[ch]);
                        value = vminq_f32(vmaxq_f32(value, vdupq_n_f32(0.f)), vdupq_n_f32(6.f));
                        q[lane] = quantized_four(value, reciprocal);
                    }
                }
                store_four(job->dst+(size_t)p*padded+c, padded, q);
            }
#endif
            for (; p < end; ++p)
                for (int lane = 0; lane < 8; ++lane) {
                    int ch = c+lane;
                    job->dst[(size_t)p*padded+ch] = ch < job->channels ? rounded(clipped(
                        converted(job->src[(size_t)ch*job->pixels+p], job->factors[ch],
                                  job->bias[ch]))/job->scale) : 0;
                }
        }
    }
    return NULL;
}

static int saturated_scale(const int64_t *src, const float *factors,
                           const float *bias, int channels, int pixels) {
    /* A loose bound proves every unvisited conversion finite. Only then may
     * finding one saturated value replace the full maximum/validation pass. */
    for (int c = 0; c < channels; ++c)
        if ((double)factors[c]*9223372036854775808.0+fabs((double)bias[c]) >
                (double)FLT_MAX*.5) return 0;
    int remaining = 2048;
    for (int c = 0; c < channels && remaining > 0; ++c) {
        int end = pixels < 64 ? pixels : 64;
        if (end > remaining) end = remaining;
        remaining -= end;
        const int64_t *plane = src+(size_t)c*pixels;
        int p = 0;
#ifdef __ARM_NEON
        for (; p+4 <= end; p += 4) {
            uint32x4_t saturated = vcgeq_f32(converted_four(plane+p, factors[c], bias[c]),
                                            vdupq_n_f32(6.f));
            uint32x2_t pair = vpmax_u32(vget_low_u32(saturated), vget_high_u32(saturated));
            if (vget_lane_u32(pair, 0) | vget_lane_u32(pair, 1)) return 1;
        }
        /* A scan limit can split a vector; leave it for the exact full path. */
        if (end < pixels) continue;
#endif
        for (; p < end; ++p)
            if (converted(plane[p], factors[c], bias[c]) >= 6.f) return 1;
    }
    return 0;
}

static void *pack_worker(void *argument) {
    pack_job *job = (pack_job *)argument;
    const float *src = job->src;
    int16_t *dst = job->dst;
    int channels = job->channels, pixels = job->pixels;
    int padded = (channels+7)/8*8;
    float scale = job->scale;
#ifdef __ARM_NEON
    float32x4_t divisor = vdupq_n_f32(scale);
    float32x4_t reciprocal = vrecpeq_f32(divisor);
    reciprocal = vmulq_f32(vrecpsq_f32(divisor, reciprocal), reciprocal);
    reciprocal = vmulq_f32(vrecpsq_f32(divisor, reciprocal), reciprocal);
#endif
    for (int block = job->first; block < job->last; block += 16) {
        int end = block+16 < job->last ? block+16 : job->last;
        for (int c = 0; c < padded; c += 8) {
            int p = block;
#ifdef __ARM_NEON
            for (; p+4 <= end; p += 4) {
                int16x4_t q[8];
                for (int lane = 0; lane < 8; ++lane) {
                    int ch = c+lane;
                    q[lane] = vdup_n_s16(0);
                    if (ch < channels) {
                        float32x4_t value = vld1q_f32(src+(size_t)ch*pixels+p);
                        q[lane] = quantized_four(value, reciprocal);
                    }
                }
                store_four(dst+(size_t)p*padded+c, padded, q);
            }
#endif
            for (; p < end; ++p)
                for (int lane = 0; lane < 8; ++lane) {
                    int ch = c+lane;
                    dst[(size_t)p*padded+ch] = ch < channels ?
                        rounded(src[(size_t)ch*pixels+p]/scale) : 0;
                }
        }
    }
    return NULL;
}

/* Caller owns scratch (channels*pixels floats). No retained pointers or
 * process-wide workers: at most one extra thread, joined in each phase. */
int fingertip_q16_clip_pack_workspace(const int64_t *src, int16_t *dst,
        const float *factors, const float *bias, int channels, int pixels,
        float *scale_out, float *scratch) {
    if (!src || !dst || !factors || !bias || !scale_out || !scratch || channels < 1 ||
            channels > 1024 || pixels < 1 || pixels > 1048576 ||
            (size_t)((channels+7)/8*8)*pixels > 1048576 ||
            fegetround() != FE_TONEAREST) return -1;
    for (int c = 0; c < channels; ++c)
        if (!isfinite(factors[c]) || factors[c] < 0.f || !isfinite(bias[c])) return -1;
    size_t count = (size_t)channels*pixels;
    const char *serial_enabled = getenv("FRETLENS_NEURAL_SERIAL");
    int serial = serial_enabled && serial_enabled[0] == '1' && serial_enabled[1] == '\0';
    const char *direct_enabled = getenv("FRETLENS_FINGERTIP_DIRECT");
    if (direct_enabled && direct_enabled[0] == '1' &&
            saturated_scale(src, factors, bias, channels, pixels)) {
        float scale = 6.f/32767.f;
        direct_job first = {src, dst, factors, bias, channels, pixels, 0, pixels, scale};
        pthread_t direct_thread;
        if (!serial && count >= 262144 && pixels >= 32) {
            first.last = (pixels/2)/16*16;
            direct_job second = {src, dst, factors, bias, channels, pixels, first.last, pixels, scale};
            if (pthread_create(&direct_thread, NULL, direct_worker, &first) == 0) {
                direct_worker(&second);
                pthread_join(direct_thread, NULL);
            } else {
                first.last = pixels;
                direct_worker(&first);
            }
        } else {
            direct_worker(&first);
        }
        *scale_out = scale;
        return 0;
    }
    convert_job first = {src, scratch, factors, bias, 0, channels, pixels, 0.f, 0};
    pthread_t thread;
    if (!serial && channels >= 2 && count >= 131072) {
        first.last = channels/2;
        convert_job second = {src, scratch, factors, bias, first.last, channels, pixels, 0.f, 0};
        if (pthread_create(&thread, NULL, convert_worker, &first) == 0) {
            convert_worker(&second);
            pthread_join(thread, NULL);
            if (first.status || second.status) return -1;
            first.maximum = fmaxf(first.maximum, second.maximum);
        } else {
            first.last = channels;
            convert_worker(&first);
        }
    } else {
        convert_worker(&first);
    }
    if (first.status) return -1;
    float scale = fmaxf(first.maximum/32767.f, FLT_MIN);
    pack_job pack = {scratch, dst, channels, pixels, 0, pixels, scale};
    if (!serial && count >= 262144 && pixels >= 32) {
        pack.last = (pixels/2)/16*16;
        pack_job second = {scratch, dst, channels, pixels, pack.last, pixels, scale};
        if (pthread_create(&thread, NULL, pack_worker, &pack) == 0) {
            pack_worker(&second);
            pthread_join(thread, NULL);
        } else {
            pack.last = pixels;
            pack_worker(&pack);
        }
    } else {
        pack_worker(&pack);
    }
    *scale_out = scale;
    return 0;
}

/* Compatibility ABI; the adapter uses reusable caller-owned scratch. */
int fingertip_q16_clip_pack(const int64_t *src, int16_t *dst,
        const float *factors, const float *bias, int channels, int pixels,
        float *scale_out) {
    if (channels < 1 || channels > 1024 || pixels < 1 || pixels > 1048576 ||
            (size_t)((channels+7)/8*8)*pixels > 1048576) return -1;
    float *scratch = malloc((size_t)channels*pixels*sizeof(float));
    if (!scratch) return -1;
    int status = fingertip_q16_clip_pack_workspace(src, dst, factors, bias,
                                                   channels, pixels, scale_out, scratch);
    free(scratch);
    return status;
}
