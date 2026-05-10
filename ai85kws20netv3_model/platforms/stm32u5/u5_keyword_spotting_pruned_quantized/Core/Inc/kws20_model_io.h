#ifndef KWS20_MODEL_IO_H
#define KWS20_MODEL_IO_H

#include <stdint.h>

#include "ai_platform.h"
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

static inline float kws20_output_score(const ai_buffer *buffer, uint32_t index, kws20_output_elem_t v)
{
    ai_float scale = 1.0f;
    int zero_point = 0;

    if (buffer != NULL && AI_BUFFER_META_INFO_INTQ(buffer->meta_info)) {
        uint32_t q_index = 0;
        ai_u16 q_size = AI_BUFFER_META_INFO_INTQ_GET_SIZE(buffer->meta_info);

        if (q_size > 1u && index < (uint32_t)q_size) {
            q_index = index;
        }

        scale = AI_BUFFER_META_INFO_INTQ_GET_SCALE(buffer->meta_info, q_index);
        zero_point = AI_BUFFER_META_INFO_INTQ_GET_ZEROPOINT(buffer->meta_info, q_index);
    }

    return ((float)((int32_t)v - zero_point)) * (float)scale;
}
#elif AI_NETWORK_OUT_1_FORMAT == AI_BUFFER_FORMAT_FLOAT
#define KWS20_MODEL_OUTPUT_IS_INT8 0
typedef ai_float kws20_output_elem_t;

static inline float kws20_output_score(const ai_buffer *buffer, uint32_t index, kws20_output_elem_t v)
{
    (void)buffer;
    (void)index;
    return (float)v;
}
#else
#error Unsupported AI_NETWORK_OUT_1_FORMAT for kws20_model_io.h
#endif

#endif
