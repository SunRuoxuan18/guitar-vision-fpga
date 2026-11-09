#include "hand_requant.h"

#include <float.h>

static float convert_sum(long long sum, float scale, float bias,
                         float clip_min, float clip_max) {
#pragma HLS INLINE off
    float product = (float)sum * scale;
#pragma HLS BIND_OP variable=product op=fmul impl=fabric
    float value = product + bias;
#pragma HLS BIND_OP variable=value op=fadd impl=fabric
    return value < clip_min ? clip_min : (value > clip_max ? clip_max : value);
}

static ap_int<16> round_to_even_int16(float value, float reciprocal) {
#pragma HLS INLINE off
    float scaled = value * reciprocal;
#pragma HLS BIND_OP variable=scaled op=fmul impl=fabric
    float clamped = scaled < -32767.0f ? -32767.0f :
                    (scaled > 32767.0f ? 32767.0f : scaled);
    int integral = (int)clamped;
    float fraction = clamped - (float)integral;
    float magnitude_fraction = fraction < 0.0f ? -fraction : fraction;
    bool increment = magnitude_fraction > 0.5f ||
        (magnitude_fraction == 0.5f && (integral & 1));
    if (increment)
        integral += clamped < 0.0f ? -1 : 1;
    return (ap_int<16>)integral;
}

void hand_requant(const long long *input, const float *params,
                  ap_uint<128> *output, float *metadata,
                  int channels, int pixels, float clip_min, float clip_max) {
#pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem max_read_burst_length=16 max_write_burst_length=16 num_read_outstanding=1 num_write_outstanding=1 depth=4096
#pragma HLS INTERFACE m_axi port=params offset=slave bundle=gmem max_read_burst_length=16 max_write_burst_length=16 num_read_outstanding=1 num_write_outstanding=1 depth=2048
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=packed max_write_burst_length=16 num_write_outstanding=1 depth=4096
#pragma HLS INTERFACE m_axi port=metadata offset=slave bundle=gmem max_read_burst_length=16 max_write_burst_length=16 num_read_outstanding=1 num_write_outstanding=1 depth=2
#pragma HLS INTERFACE s_axilite port=input bundle=control
#pragma HLS INTERFACE s_axilite port=params bundle=control
#pragma HLS INTERFACE s_axilite port=output bundle=control
#pragma HLS INTERFACE s_axilite port=metadata bundle=control
#pragma HLS INTERFACE s_axilite port=channels bundle=control
#pragma HLS INTERFACE s_axilite port=pixels bundle=control
#pragma HLS INTERFACE s_axilite port=clip_min bundle=control
#pragma HLS INTERFACE s_axilite port=clip_max bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
    const long long padded_channel_count = (((long long)channels + 7) / 8) * 8;
    if (channels <= 0 || channels > HAND_REQUANT_MAX_CHANNELS || pixels <= 0 ||
        pixels > HAND_REQUANT_MAX_ELEMENTS || (long long)channels * pixels > HAND_REQUANT_MAX_ELEMENTS ||
        padded_channel_count > HAND_REQUANT_MAX_ELEMENTS / pixels ||
        !(clip_min <= clip_max) || !(clip_min >= -FLT_MAX && clip_max <= FLT_MAX)) {
        metadata[HAND_REQUANT_METADATA_SCALE] = 0.0f;
        metadata[HAND_REQUANT_METADATA_STATUS] = -1.0f;
        return;
    }

    const int padded_channels = (int)padded_channel_count;
    float maximum = 0.0f;
    long long maximum_input_offset = 0;
find_dynamic_maximum:
    for (int channel = 0; channel < channels; ++channel) {
        float scale = params[channel];
        float bias = params[HAND_REQUANT_MAX_CHANNELS + channel];
        if (!(scale >= -FLT_MAX && scale <= FLT_MAX && bias >= -FLT_MAX && bias <= FLT_MAX)) {
            metadata[HAND_REQUANT_METADATA_SCALE] = 0.0f;
            metadata[HAND_REQUANT_METADATA_STATUS] = -2.0f;
            return;
        }
        for (int pixel = 0; pixel < pixels; ++pixel) {
            float value = convert_sum(input[maximum_input_offset + pixel], scale, bias,
                                      clip_min, clip_max);
            float magnitude = value < 0.0f ? -value : value;
            if (magnitude > maximum)
                maximum = magnitude;
        }
        maximum_input_offset += pixels;
    }
    float dynamic_scale = maximum / 32767.0f;
    if (dynamic_scale < FLT_MIN)
        dynamic_scale = FLT_MIN;
    metadata[HAND_REQUANT_METADATA_SCALE] = dynamic_scale;
    metadata[HAND_REQUANT_METADATA_STATUS] = 0.0f;
    float reciprocal = 1.0f / dynamic_scale;

    long long output_index = 0;
pack_nhwc8:
    for (int pixel = 0; pixel < pixels; ++pixel) {
        long long base_input_offset = 0;
        for (int channel_base = 0; channel_base < padded_channels; channel_base += 8) {
            ap_uint<128> word = 0;
            long long lane_input_offset = base_input_offset;
pack_lanes:
            for (int lane = 0; lane < 8; ++lane) {
#pragma HLS PIPELINE II=1
                int channel = channel_base + lane;
                ap_int<16> quantized = 0;
                if (channel < channels) {
                    float value = convert_sum(input[lane_input_offset + pixel], params[channel],
                                              params[HAND_REQUANT_MAX_CHANNELS + channel], clip_min, clip_max);
                    quantized = round_to_even_int16(value, reciprocal);
                }
                word.range(lane * 16 + 15, lane * 16) = quantized;
                lane_input_offset += pixels;
            }
            output[output_index++] = word;
            base_input_offset += (pixels << 3);
        }
    }
}
