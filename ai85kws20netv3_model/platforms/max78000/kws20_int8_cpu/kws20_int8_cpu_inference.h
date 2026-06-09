#ifndef KWS20_INT8_CPU_INFERENCE_H
#define KWS20_INT8_CPU_INFERENCE_H

#define KWS_INT8_INPUT_CHANNELS  128
#define KWS_INT8_INPUT_TIME      128
#define KWS_INT8_NUM_CLASSES     21

#define KWS_INT8_OK   0
#define KWS_INT8_ERR -1

#include <stdint.h>
int kws20_int8_cpu_infer(const int8_t *input, float *scores);
int kws20_int8_cpu_argmax(const float *scores, int n);

#endif
