#ifndef KWS20_32BIT_INFERENCE_H
#define KWS20_32BIT_INFERENCE_H

#include <stdint.h>

/* Network constants */
#define KWS32_INPUT_CHANNELS  128
#define KWS32_INPUT_TIME      128
#define KWS32_NUM_CLASSES     21

/* Return codes */
#define KWS32_OK   1
#define KWS32_FAIL 0

/*
 * Run one float32 inference on the Cortex-M4F CPU.
 *
 * input   : float[128 * 128]  — row-major (channels, time), range [0, 1)
 * scores  : float[21]          — raw logits (apply softmax externally)
 *
 * Returns KWS32_OK on success.
 * Uses ~60 KB of stack — call from a task with sufficient stack.
 */
int kws20_32bit_infer(const float *input, float *scores);

/*
 * Argmax over the logit vector.
 * Returns class index 0..20.
 */
int kws20_32bit_argmax(const float *scores, int n);

#endif /* KWS20_32BIT_INFERENCE_H */
