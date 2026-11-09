/* Byte transport only; grayscale and edge algorithms execute in PL. */
#include <stdint.h>
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

void vision_unpack_planes(const uint8_t *src, uint8_t *gray, uint8_t *edge, int pixels) {
    int p = 0;
#ifdef __ARM_NEON
    for (; p + 16 <= pixels; p += 16) {
        uint8x16x4_t lanes = vld4q_u8(src + 4*p);
        vst1q_u8(gray+p, lanes.val[0]);
        vst1q_u8(edge+p, lanes.val[2]);
    }
#endif
    for (; p < pixels; ++p) {
        gray[p] = src[4*p];
        edge[p] = src[4*p+2];
    }
}
