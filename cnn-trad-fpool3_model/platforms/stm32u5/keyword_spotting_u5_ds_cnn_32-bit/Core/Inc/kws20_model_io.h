#ifndef KWS20_MODEL_IO_H
#define KWS20_MODEL_IO_H

#include <stdint.h>

#include "ai_platform.h"
#include "network.h"

/*
 * AI_BUFFER_FORMAT_FLOAT / AI_BUFFER_FORMAT_S8 are enum values, not
 * preprocessor macros, so they cannot be compared with #if.
 * This model is float32: hardcode accordingly.
 */
#define KWS20_MODEL_INPUT_IS_INT8  0
#define KWS20_MODEL_OUTPUT_IS_INT8 0

typedef ai_float kws20_input_elem_t;

static inline kws20_input_elem_t kws20_input_from_i32(int32_t v)
{
    return (ai_float)v;
}

static inline kws20_input_elem_t kws20_input_from_float(float v)
{
    return (ai_float)v;
}

typedef ai_float kws20_output_elem_t;

static inline float kws20_output_score(const ai_buffer *buffer, uint32_t index,
                                       kws20_output_elem_t v)
{
    (void)buffer;
    (void)index;
    return (float)v;
}

#endif
