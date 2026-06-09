#ifndef KWS20_MODEL_IO_H
#define KWS20_MODEL_IO_H

#include <stdint.h>

#include "ai_platform.h"
#include "network.h"

/*
 * AI_BUFFER_FORMAT_FLOAT / AI_BUFFER_FORMAT_S8 are enum values, not
 * preprocessor macros, so they cannot be compared with #if.
 * This model is INT8: hardcode accordingly.
 */
#define KWS20_MODEL_INPUT_IS_INT8  1
#define KWS20_MODEL_OUTPUT_IS_INT8 1

/*
 * Input quantization params from X-CUBE-AI analysis of ds_cnn_l.tflite:
 *   QLinear(0.584702671, 83, int8)
 */
#define KWS20_INPUT_SCALE       (0.584702671f)
#define KWS20_INPUT_ZERO_POINT  (83)

typedef ai_i8 kws20_input_elem_t;

static inline kws20_input_elem_t kws20_input_from_i32(int32_t v)
{
    float q = (float)v / KWS20_INPUT_SCALE + (float)KWS20_INPUT_ZERO_POINT;
    if (q < -128.0f) q = -128.0f;
    if (q > 127.0f) q = 127.0f;
    return (ai_i8)(int8_t)q;
}

static inline kws20_input_elem_t kws20_input_from_float(float v)
{
    float q = v / KWS20_INPUT_SCALE + (float)KWS20_INPUT_ZERO_POINT;
    if (q < -128.0f) q = -128.0f;
    if (q > 127.0f) q = 127.0f;
    return (ai_i8)(int8_t)q;
}

typedef ai_i8 kws20_output_elem_t;

/*
 * Dequantize INT8 output score to float using per-tensor quantization params
 * embedded in the ai_buffer metadata (ds_cnn_l.tflite: scale=0.00390625, zp=-128).
 */
static inline float kws20_output_score(const ai_buffer *buffer, uint32_t index,
                                       kws20_output_elem_t v)
{
    (void)index;
    float scale = AI_BUFFER_META_INFO_INTQ_GET_SCALE(AI_BUFFER_META_INFO(buffer), 0);
    ai_i8 zp    = (ai_i8)AI_BUFFER_META_INFO_INTQ_GET_ZEROPOINT(AI_BUFFER_META_INFO(buffer), 0);
    return ((float)v - (float)zp) * scale;
}

#endif
