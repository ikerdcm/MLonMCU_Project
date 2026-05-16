/*
 * CNN accelerator inference wrapper for KWS12 MFCC model.
 *
 * Quantizes float MFCC features to INT8, loads them into the CNN
 * accelerator memory, runs inference, and unloads the result.
 *
 * Input memory layout (HWC, 10 channels, 49 time steps):
 *   0x50400000: channels 0-3, 49 × uint32  (4 int8 values packed per word)
 *   0x50408000: channels 4-7, 49 × uint32
 *   0x50410000: channels 8-9, 49 × uint32  (2 int8 values, upper bytes = 0)
 */

#include "cnn_inference.h"
#include "cnn.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "mxc_device.h"
#include "lp.h"

#define N_FRAMES    49
#define N_MFCC      10
#define SCALE       127.0f   /* input must be in [-1, 1]; int8 = round(val * 127) */

static int32_t ml_data[CNN_NUM_OUTPUTS];
static q15_t   ml_softmax[CNN_NUM_OUTPUTS];

static void load_mfcc(const float *mfcc_in)
{
    volatile uint32_t *mem0 = (volatile uint32_t *)0x50400000;
    volatile uint32_t *mem4 = (volatile uint32_t *)0x50408000;
    volatile uint32_t *mem8 = (volatile uint32_t *)0x50410000;

    for (int t = 0; t < N_FRAMES; t++) {
        const float *row = mfcc_in + t * N_MFCC;
        uint8_t c[N_MFCC];

        for (int ch = 0; ch < N_MFCC; ch++) {
            int32_t q = (int32_t)roundf(row[ch] * SCALE);
            if (q >  127) q =  127;
            if (q < -128) q = -128;
            c[ch] = (uint8_t)(int8_t)q;
        }

        mem0[t] = (uint32_t)c[0]
                | ((uint32_t)c[1] <<  8)
                | ((uint32_t)c[2] << 16)
                | ((uint32_t)c[3] << 24);

        mem4[t] = (uint32_t)c[4]
                | ((uint32_t)c[5] <<  8)
                | ((uint32_t)c[6] << 16)
                | ((uint32_t)c[7] << 24);

        mem8[t] = (uint32_t)c[8]
                | ((uint32_t)c[9] <<  8);
    }
}

int cnn_infer(const float *mfcc_in, float *scores)
{
    /* Load quantized MFCC into accelerator input memory */
    load_mfcc(mfcc_in);

    /* Run accelerator */
    cnn_time = 0;
    cnn_start();
    while (cnn_time == 0)
        MXC_LP_EnterSleepMode();

    /* Unload raw INT32 logits */
    cnn_unload((uint32_t *)ml_data);

    /* Optional softmax */
    if (scores != NULL) {
        softmax_q17p14_q15((const q31_t *)ml_data, CNN_NUM_OUTPUTS, ml_softmax);
        for (int i = 0; i < CNN_NUM_OUTPUTS; i++)
            scores[i] = (float)ml_softmax[i] / 32768.0f;
    }

    /* Argmax */
    int best = 0;
    for (int i = 1; i < CNN_NUM_OUTPUTS; i++)
        if (ml_data[i] > ml_data[best])
            best = i;

    return best;
}
