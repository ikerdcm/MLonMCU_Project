#ifndef KWS20_MODEL_IO_H
#define KWS20_MODEL_IO_H

#include <stdint.h>

#include "network.h"

#if AI_NETWORK_IN_1_FORMAT == AI_BUFFER_FORMAT_S8
#define KWS20_MODEL_INPUT_IS_INT8 1
typedef int8_t kws20_input_elem_t;

static inline kws20_input_elem_t kws20_input_from_i32(int32_t v)
{
    if (v > 127) {
        v = 127;
    }
    if (v < -128) {
        v = -128;
    }
    return (int8_t)v;
}

static inline kws20_input_elem_t kws20_input_from_float(float v)
{
    int32_t q = (v >= 0.0f) ? (int32_t)(v + 0.5f) : (int32_t)(v - 0.5f);
    return kws20_input_from_i32(q);
}
#elif AI_NETWORK_IN_1_FORMAT == AI_BUFFER_FORMAT_FLOAT
#define KWS20_MODEL_INPUT_IS_INT8 0
typedef ai_float kws20_input_elem_t;

static inline kws20_input_elem_t kws20_input_from_i32(int32_t v)
{
    return (ai_float)v;
}

static inline kws20_input_elem_t kws20_input_from_float(float v)
{
    return (ai_float)v;
}
#else
#error Unsupported AI_NETWORK_IN_1_FORMAT for kws20_model_io.h
#endif

#if AI_NETWORK_OUT_1_FORMAT == AI_BUFFER_FORMAT_S8
#define KWS20_MODEL_OUTPUT_IS_INT8 1
typedef int8_t kws20_output_elem_t;

static inline float kws20_output_score(kws20_output_elem_t v)
{
    return (float)((int32_t)v);
}
#elif AI_NETWORK_OUT_1_FORMAT == AI_BUFFER_FORMAT_FLOAT
#define KWS20_MODEL_OUTPUT_IS_INT8 0
typedef ai_float kws20_output_elem_t;

static inline float kws20_output_score(kws20_output_elem_t v)
{
    return (float)v;
}
#else
#error Unsupported AI_NETWORK_OUT_1_FORMAT for kws20_model_io.h
#endif

#endif
