/* PS tensor transport only. Neural MACs execute in the PL kernel. */
#include <stdint.h>
#include <stddef.h>
#include <float.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <pthread.h>
#ifdef __ARM_NEON
#include <arm_neon.h>

static inline int16x4_t quantize_four(float32x4_t input, float32x4_t reciprocal) {
    float32x4_t value = vminq_f32(vmulq_f32(vabsq_f32(input),reciprocal),vdupq_n_f32(32767.f));
    int32x4_t integral = vcvtq_s32_f32(value);
    float32x4_t fraction = vsubq_f32(value,vcvtq_f32_s32(integral));
    uint32x4_t increment = vorrq_u32(vcgtq_f32(fraction,vdupq_n_f32(.5f)),
        vandq_u32(vceqq_f32(fraction,vdupq_n_f32(.5f)),vtstq_s32(integral,vdupq_n_s32(1))));
    integral = vsubq_s32(integral,vreinterpretq_s32_u32(increment));
    return vmovn_s32(vbslq_s32(vcltq_f32(input,vdupq_n_f32(0.f)),vnegq_s32(integral),integral));
}
#endif

/* Byte transport only: RGBA capture to the existing B,G,R,0 AXI contract. */
void vision_rgba_pack(const uint8_t *src, uint8_t *dst, int pixels) {
    int p = 0;
#ifdef __ARM_NEON
    for (; p+16 <= pixels; p += 16) {
        uint8x16x4_t rgba = vld4q_u8(src+4*p), bgra;
        bgra.val[0] = rgba.val[2];
        bgra.val[1] = rgba.val[1];
        bgra.val[2] = rgba.val[0];
        bgra.val[3] = vdupq_n_u8(0);
        vst4q_u8(dst+4*p,bgra);
    }
#endif
    for (; p < pixels; ++p) {
        dst[4*p] = src[4*p+2];
        dst[4*p+1] = src[4*p+1];
        dst[4*p+2] = src[4*p];
        dst[4*p+3] = 0;
    }
}

static void pack_pixels(const float *src, int16_t *dst, int channels, int pixels,
                        float scale, int first, int last) {
    int padded = (channels + 7) / 8 * 8;
#ifdef __ARM_NEON
    float32x4_t divisor = vdupq_n_f32(scale);
    float32x4_t reciprocal = vrecpeq_f32(divisor);
    reciprocal = vmulq_f32(vrecpsq_f32(divisor,reciprocal),reciprocal);
    reciprocal = vmulq_f32(vrecpsq_f32(divisor,reciprocal),reciprocal);
    /* Bound the NHWC destination working set instead of revisiting every
     * output cache line once for each channel group. */
    for (int block = first; block < last; block += 16) {
      int end = block+16 < last ? block+16 : last;
      for (int c = 0; c < padded; c += 8) {
        int p = block;
        for (; p + 4 <= end; p += 4) {
            /* Transpose four pixels by eight channels in registers, then store
             * contiguous NHWC8 rows instead of scattering each channel. */
            int16x4_t q[8];
            for (int lane = 0; lane < 8; ++lane)
                q[lane] = c+lane < channels ? quantize_four(vld1q_f32(src+(size_t)(c+lane)*pixels+p),reciprocal) : vdup_n_s16(0);
            int16x4x2_t a = vzip_s16(q[0],q[1]), b = vzip_s16(q[2],q[3]);
            int16x4x2_t d = vzip_s16(q[4],q[5]), e = vzip_s16(q[6],q[7]);
            int32x2x2_t ab0 = vzip_s32(vreinterpret_s32_s16(a.val[0]),vreinterpret_s32_s16(b.val[0]));
            int32x2x2_t ab1 = vzip_s32(vreinterpret_s32_s16(a.val[1]),vreinterpret_s32_s16(b.val[1]));
            int32x2x2_t de0 = vzip_s32(vreinterpret_s32_s16(d.val[0]),vreinterpret_s32_s16(e.val[0]));
            int32x2x2_t de1 = vzip_s32(vreinterpret_s32_s16(d.val[1]),vreinterpret_s32_s16(e.val[1]));
            vst1q_s16(dst+(size_t)p*padded+c,vcombine_s16(vreinterpret_s16_s32(ab0.val[0]),vreinterpret_s16_s32(de0.val[0])));
            vst1q_s16(dst+(size_t)(p+1)*padded+c,vcombine_s16(vreinterpret_s16_s32(ab0.val[1]),vreinterpret_s16_s32(de0.val[1])));
            vst1q_s16(dst+(size_t)(p+2)*padded+c,vcombine_s16(vreinterpret_s16_s32(ab1.val[0]),vreinterpret_s16_s32(de1.val[0])));
            vst1q_s16(dst+(size_t)(p+3)*padded+c,vcombine_s16(vreinterpret_s16_s32(ab1.val[1]),vreinterpret_s16_s32(de1.val[1])));
        }
        for (; p < end; ++p)
            for (int lane = 0; lane < 8; ++lane) {
                float value = c+lane < channels ? src[(size_t)(c+lane)*pixels+p]/scale : 0.f;
                dst[(size_t)p*padded+c+lane] = (int16_t)nearbyintf(fminf(32767.f,fmaxf(-32767.f,value)));
            }
      }
    }
#else
    if (padded != channels) memset(dst+(size_t)first*padded, 0,
                                  (size_t)padded * (last-first) * sizeof(int16_t));
    for (int c = 0; c < channels; ++c)
        for (int p = first; p < last; ++p) {
            float value = fminf(32767.f,fmaxf(-32767.f,src[(size_t)c*pixels+p]/scale));
            dst[(size_t)p*padded+c] = (int16_t)nearbyintf(value);
        }
#endif
}

typedef struct {
    const float *src;
    int16_t *dst;
    int channels, pixels, last;
    float scale;
} pack_job;

static void *pack_worker(void *argument) {
    pack_job *job = (pack_job *)argument;
    pack_pixels(job->src,job->dst,job->channels,job->pixels,job->scale,0,job->last);
    return NULL;
}

int hand_q16_pack(const float *src, int16_t *dst, int channels, int pixels, float *scale_out) {
    const char *serial_enabled = getenv("FRETLENS_NEURAL_SERIAL");
    int serial = serial_enabled && serial_enabled[0] == '1' && serial_enabled[1] == '\0';
    float maximum = 0.f;
    size_t count = (size_t)channels * pixels;
    size_t i = 0;
#ifdef __ARM_NEON
    float32x4_t vmax = vdupq_n_f32(0.f);
    uint32x4_t finite = vdupq_n_u32(~0u);
    for (; i+4 <= count; i += 4) {
        float32x4_t value = vabsq_f32(vld1q_f32(src+i));
        finite = vandq_u32(finite,vcltq_f32(value,vdupq_n_f32(INFINITY)));
        vmax = vmaxq_f32(vmax,value);
    }
    uint32_t checked[4]; float maxima[4];
    vst1q_u32(checked,finite); vst1q_f32(maxima,vmax);
    for (int j = 0; j < 4; ++j) {
        if (!checked[j]) return -1;
        if (maxima[j] > maximum) maximum = maxima[j];
    }
#endif
    for (; i < count; ++i) {
        if (!isfinite(src[i])) return -1;
        float value = fabsf(src[i]);
        if (value > maximum) maximum = value;
    }
    float scale = fmaxf(maximum / 32767.f, FLT_MIN);
    *scale_out = scale;
    /* Split on the existing 16-pixel block boundary so vector rounding and
     * tail handling are identical, with disjoint destination cache lines. */
    if (!serial && count >= 262144 && pixels >= 32) {
        int split = (pixels/2)/16*16;
        pack_job job = {src,dst,channels,pixels,split,scale};
        pthread_t worker;
        if (pthread_create(&worker,NULL,pack_worker,&job) == 0) {
            pack_pixels(src,dst,channels,pixels,scale,split,pixels);
            pthread_join(worker,NULL);
            return 0;
        }
    }
    pack_pixels(src,dst,channels,pixels,scale,0,pixels);
    return 0;
}

static void hand_q16_unpack_serial(const int64_t *src, float *dst, const float *scales,
                     const float *bias, float input_scale, int channels, int pixels) {
    for (int c = 0; c < channels; ++c) {
        float scale = scales[c]*input_scale;
        size_t start = (size_t)c*pixels;
        int p = 0;
#ifdef __ARM_NEON
        for (; p + 4 <= pixels; p += 4) {
            /* Deinterleave words directly, keeping the original unsigned
             * magnitude conversion and its rounding at every int64 boundary. */
            int32x4x2_t words = vld2q_s32((const int32_t *)(src+start+p));
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
            int32x4_t low = words.val[1], high = words.val[0];
#else
            int32x4_t low = words.val[0], high = words.val[1];
#endif
            uint32x4_t fits = vceqq_s32(high,vshrq_n_s32(low,31));
            uint32x2_t paired_fits = vpmin_u32(vget_low_u32(fits),vget_high_u32(fits));
            float32x4_t value;
            if ((vget_lane_u32(paired_fits,0) & vget_lane_u32(paired_fits,1)) == UINT32_MAX) {
                value = vcvtq_f32_s32(low);
            } else {
            uint32x4_t sign = vreinterpretq_u32_s32(vshrq_n_s32(high,31));
            uint32x4_t lower = vreinterpretq_u32_s32(low);
            uint32x4_t magnitude_low = vsubq_u32(veorq_u32(lower,sign),sign);
            uint32x4_t magnitude_high = vsubq_u32(veorq_u32(vreinterpretq_u32_s32(high),sign),
                vandq_u32(sign,vceqq_u32(lower,vdupq_n_u32(0))));
            value = vaddq_f32(vmulq_n_f32(vcvtq_f32_u32(magnitude_high),4294967296.f),
                vcvtq_f32_u32(magnitude_low));
            value = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(value),
                vandq_u32(sign,vdupq_n_u32(0x80000000u))));
            }
            value = vaddq_f32(vmulq_n_f32(value,scale),vdupq_n_f32(bias[c]));
            vst1q_f32(dst+start+p,value);
        }
#endif
        for (; p < pixels; ++p) dst[start+p] = (float)src[start+p]*scale+bias[c];
    }
}

typedef struct {
    const int64_t *src;
    float *dst;
    const float *scales, *bias;
    float input_scale;
    int channels, pixels;
} unpack_job;

static void *unpack_worker(void *argument) {
    unpack_job *job = (unpack_job *)argument;
    hand_q16_unpack_serial(job->src,job->dst,job->scales,job->bias,
        job->input_scale,job->channels,job->pixels);
    return NULL;
}

void hand_q16_unpack(const int64_t *src, float *dst, const float *scales,
                     const float *bias, float input_scale, int channels, int pixels) {
    const char *serial_enabled = getenv("FRETLENS_NEURAL_SERIAL");
    int serial = serial_enabled && serial_enabled[0] == '1' && serial_enabled[1] == '\0';
    /* Keep small tensors serial; both workers own disjoint channel planes.
     * Join before returning so no worker retains a DMA-buffer pointer. */
    if (!serial && channels >= 2 && (size_t)channels*pixels >= 131072) {
        int split = channels/2;
        size_t offset = (size_t)split*pixels;
        unpack_job job = {src,dst,scales,bias,input_scale,split,pixels};
        pthread_t worker;
        if (pthread_create(&worker,NULL,unpack_worker,&job) == 0) {
            hand_q16_unpack_serial(src+offset,dst+offset,scales+split,bias+split,
                input_scale,channels-split,pixels);
            pthread_join(worker,NULL);
            return;
        }
    }
    hand_q16_unpack_serial(src,dst,scales,bias,input_scale,channels,pixels);
}

static int compare_float(const void *left, const void *right) {
    float a = *(const float *)left, b = *(const float *)right;
    return (a > b)-(a < b);
}

/* PS appearance validation on already extracted uint8 patches. */
int tracking_patch_correlations(const uint8_t *first, const uint8_t *second,
        float *output, int count, int pixels, float minimum_energy, int flat_template) {
    if (!first || !second || !output || count < 1 || count > 240 ||
            pixels < 1 || pixels > 256 || minimum_energy < 0) return -1;
    for (int row = 0; row < count; ++row) {
        int64_t a = 0, b = 0, aa = 0, bb = 0, ab = 0;
        for (int p = 0; p < pixels; ++p) {
            int x = first[(size_t)row*pixels+p], y = second[(size_t)row*pixels+p];
            a += x; b += y; aa += x*x; bb += y*y; ab += x*y;
        }
        int64_t ea = pixels*aa-a*a, eb = pixels*bb-b*b;
        double denominator = sqrt((double)ea*(double)eb);
        double correlation = denominator > (double)minimum_energy*pixels ?
            (double)(pixels*ab-a*b)/denominator : 0.;
        if (flat_template && eb == 0) correlation = 1.;
        output[row] = (float)fmax(-1.,fmin(1.,correlation));
    }
    return 0;
}

/* PS ridge measurements on the PL-rectified image, not neural arithmetic. */
int fret_ridge_profiles(const uint8_t *image, float *profiles, uint8_t *valid,
                        int width, int height) {
    if (!image || !profiles || !valid || width < 16 || width > 320 ||
            height < 16 || height > 128) return -1;
    float changes[320], ordered[320];
    int columns[320], count = 0;
    for (int x = 0; x < width-1; ++x) {
        int change = 0;
        for (int y = 0; y < height; ++y)
            change += abs((int)image[y*width+x+1]-image[y*width+x]);
        changes[x] = (float)change/height;
    }
    changes[width-1] = changes[width-2];
    memcpy(ordered,changes,width*sizeof(float));
    qsort(ordered,width,sizeof(float),compare_float);
    float index = .6f*(width-1);
    int lower = (int)index;
    float threshold = ordered[lower]+(index-lower)*(ordered[lower+1]-ordered[lower]);
    for (int x = 0; x < width; ++x)
        if (changes[x] <= threshold) columns[count++] = x;
    memset(profiles,0,2*height*sizeof(float));
    memset(valid,0,2*height);
    if (count < width/4) return 0;
    for (int sign_index = 0; sign_index < 2; ++sign_index) {
        int sign = sign_index == 0 ? 1 : -1;
        int16_t ridge[128][320];
        memset(ridge,0,sizeof(ridge));
        for (int y = 3; y < height-3; ++y) {
            int total = 0;
            for (int c = 0; c < count; ++c) {
                int x = columns[c], center = image[y*width+x];
                int a = sign*(center-image[(y-3)*width+x]);
                int b = sign*(center-image[(y+3)*width+x]);
                int value = a < b ? a : b;
                ridge[y][c] = value;
                if (value > 0) total += value;
            }
            if (y >= 4 && y < height-4) profiles[sign_index*height+y] = (float)total/count;
        }
        for (int y = 0; y < height; ++y) {
            int supports[4] = {0,0,0,0}, sizes[4];
            int c = 0, total = 0;
            for (int part = 0; part < 4; ++part) {
                sizes[part] = count/4+(part < count%4);
                for (int end = c+sizes[part]; c < end; ++c) {
                    int support = 0;
                    for (int dy = -2; dy <= 2; ++dy)
                        if (y+dy >= 0 && y+dy < height && ridge[y+dy][c] > 3) support = 1;
                    supports[part] += support;
                }
                total += supports[part];
            }
            int supported = total*4 >= count;
            for (int part = 0; part < 4; ++part)
                if (supports[part]*10 < sizes[part]) supported = 0;
            valid[sign_index*height+y] = supported;
        }
    }
    return 0;
}
