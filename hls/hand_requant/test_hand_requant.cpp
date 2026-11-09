#include "hand_requant.h"

#include <cmath>
#include <cfloat>
#include <cstdint>
#include <iostream>
#include <vector>

static int16_t reference_round(float value, float scale) {
    // The datapath uses one reciprocal per tensor; tolerance below permits the
    // last-bit difference from PS division/NEON reciprocal refinement.
    float clamped = value * (1.0f / scale);
    clamped = clamped < -32767.0f ? -32767.0f : (clamped > 32767.0f ? 32767.0f : clamped);
    return (int16_t)nearbyintf(clamped);
}

static bool run_case(const char *name, int channels, int pixels, float clip_min, float clip_max) {
    const int padded = ((channels + 7) / 8) * 8;
    // The HLS interface depth is 4096 for co-simulation instrumentation.
    std::vector<long long> input(4096, 0);
    std::vector<float> params(4096, 0.0f);
    std::vector<ap_uint<128> > output(4096, 0);
    std::vector<float> metadata(4096, 0.0f);
    metadata[1] = 99.0f;
    std::vector<float> values(channels * pixels);
    float maximum = 0.0f;
    for (int channel = 0; channel < channels; ++channel) {
        params[channel] = (channel % 5 == 0 ? -0.03125f : 0.017578125f) * (channel + 1);
        params[HAND_REQUANT_MAX_CHANNELS + channel] = (channel % 7 - 3) * 0.125f;
        for (int pixel = 0; pixel < pixels; ++pixel) {
            long long base = ((long long)(channel + 11) * 917519 + (long long)(pixel + 3) * 131071) % 2097153 - 1048576;
            input[(long long)channel * pixels + pixel] = base;
            float value = (float)base * params[channel] + params[HAND_REQUANT_MAX_CHANNELS + channel];
            value = value < clip_min ? clip_min : (value > clip_max ? clip_max : value);
            values[(long long)channel * pixels + pixel] = value;
            float magnitude = fabsf(value);
            if (magnitude > maximum) maximum = magnitude;
        }
    }
    hand_requant(input.data(), params.data(), output.data(), metadata.data(), channels, pixels, clip_min, clip_max);
    float expected_scale = maximum / 32767.0f;
    if (expected_scale < FLT_MIN) expected_scale = FLT_MIN;
    if (metadata[1] != 0.0f || metadata[0] != expected_scale) {
        std::cerr << name << " metadata mismatch scale=" << metadata[0] << " expected=" << expected_scale
                  << " status=" << metadata[1] << '\n';
        return false;
    }
    for (int pixel = 0; pixel < pixels; ++pixel) {
        for (int channel = 0; channel < padded; ++channel) {
            int16_t got = (int16_t)output[(long long)pixel * padded / 8 + channel / 8]
                .range((channel % 8) * 16 + 15, (channel % 8) * 16).to_int();
            int16_t expected = channel < channels ? reference_round(values[(long long)channel * pixels + pixel], expected_scale) : 0;
            if (std::abs((int)got - (int)expected) > 1) {
                std::cerr << name << " mismatch pixel=" << pixel << " channel=" << channel
                          << " got=" << got << " expected=" << expected << '\n';
                return false;
            }
        }
    }
    return true;
}

int main() {
    bool ok = true;
    ok &= run_case("unclipped", 13, 29, -FLT_MAX, FLT_MAX);
    ok &= run_case("clipped", 9, 17, -11.5f, 7.25f);
    ok &= run_case("relu6", 8, 23, 0.0f, 6.0f);
    ok &= run_case("single-lane", 1, 5, -FLT_MAX, FLT_MAX);
    std::vector<long long> input(4096, 0);
    std::vector<float> params(4096, 0.0f);
    std::vector<ap_uint<128> > output(4096, 0);
    std::vector<float> metadata(4096, 0.0f);
    hand_requant(input.data(), params.data(), output.data(), metadata.data(), 0, 1, -FLT_MAX, FLT_MAX);
    ok &= metadata[1] == -1.0f;
    params[0] = INFINITY;
    hand_requant(input.data(), params.data(), output.data(), metadata.data(), 1, 1, -FLT_MAX, FLT_MAX);
    ok &= metadata[1] == -2.0f;
    params[0] = 1.0f;
    params[HAND_REQUANT_MAX_CHANNELS] = 0.0f;
    input[0] = 1;
    input[1] = 2;
    hand_requant(input.data(), params.data(), output.data(), metadata.data(), 1, 2, -FLT_MAX, FLT_MAX);
    ok &= (int16_t)output[0].range(15, 0).to_int() == 16384;
    ok &= (int16_t)output[1].range(15, 0).to_int() == 32767;
    input[0] = (1LL << 47) - 1;
    params[0] = 0.25f;
    hand_requant(input.data(), params.data(), output.data(), metadata.data(), 1, 1, -FLT_MAX, FLT_MAX);
    ok &= metadata[1] == 0.0f && (int16_t)output[0].range(15, 0).to_int() == 32767;
    if (!ok) return 1;
    std::cout << "HAND_REQUANT_TEST_PASS" << std::endl;
    return 0;
}
