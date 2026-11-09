/* ARM acceleration of the existing sparse dots and 60-step NNLS loop.
 * gcc -O3 -ffp-contract=off -shared -fPIC -o libaudio_fit.so native/audio_fit.c -lm
 * Do not use fast-math: recognition thresholds depend on finite arithmetic.
 */
#include <math.h>
#include <stddef.h>
#include <stdint.h>

void audio_mmio_words(const volatile uint32_t *source, uint32_t *target, size_t count) {
    /* Volatile aligned words are required for ARM device memory, never memcpy. */
    for (size_t i = 0; i < count; ++i) target[i] = source[i];
}

static float pairwise_sum(const float *values, size_t count) {
    if (count < 8) {
        float result = -0.0f;
        for (size_t i = 0; i < count; ++i) result += values[i];
        return result;
    }
    if (count > 128) {
        size_t half = (count / 2) & ~(size_t)7;
        return pairwise_sum(values, half) + pairwise_sum(values + half, count - half);
    }
    float lane[8];
    for (size_t j = 0; j < 8; ++j) lane[j] = values[j];
    size_t i = 8;
    for (; i + 7 < count; i += 8)
        for (size_t j = 0; j < 8; ++j) lane[j] += values[i+j];
    float sum = ((lane[0]+lane[1])+(lane[2]+lane[3]))
              + ((lane[4]+lane[5])+(lane[6]+lane[7]));
    for (; i < count; ++i) sum += values[i];
    return sum;
}

void audio_sparse_dot32(const float *spectrum, const ptrdiff_t *bins,
                        const float *values, size_t rows, size_t width,
                        float *output) {
    float products[2048];
    if (width > 2048) return;
    for (size_t row = 0; row < rows; ++row) {
        for (size_t column = 0; column < width; ++column) {
            size_t offset = row * width + column;
            products[column] = spectrum[bins[offset]] * values[offset];
        }
        output[row] = pairwise_sum(products, width);
    }
}

void audio_sparse_dot(const double *spectrum, const ptrdiff_t *bins,
                      const float *values, size_t rows, size_t width,
                      double *output) {
    for (size_t row = 0; row < rows; ++row) {
        double sum = 0;
        for (size_t column = 0; column < width; ++column) {
            size_t offset = row * width + column;
            sum += spectrum[bins[offset]] * values[offset];
        }
        output[row] = sum;
    }
}

void audio_nnls_weights(const float *gram, const double *target,
                        size_t count, float *weights) {
    for (size_t i = 0; i < count; ++i) weights[i] = 0;
    for (int iteration = 0; iteration < 60; ++iteration) {
        double change = 0;
        for (size_t i = 0; i < count; ++i) {
            float dot = 0;
            for (size_t j = 0; j < count; ++j)
                dot += gram[i * count + j] * weights[j];
            double denominator = fmax(gram[i * count + i], 1e-12);
            float next = (float)fmax(0, weights[i] + (target[i] - dot) / denominator);
            change = fmax(change, fabs((double)(next - weights[i])));
            weights[i] = next;
        }
        if (change < 1e-5) break;
    }
}
