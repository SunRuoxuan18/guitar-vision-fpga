#ifndef CPU_LK_PYRAMID_H
#define CPU_LK_PYRAMID_H

#include <cstdint>

extern "C" void cpu_lk_pyramid(const uint8_t* previous, const uint8_t* current,
                                const float* points, float* result,
                                unsigned width, unsigned height, unsigned count);

#endif
