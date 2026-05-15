/**
 * DS-CNN-L offline benchmark for MAX78000 FTHR.
 * Measures inference latency using the DWT cycle counter.
 */

#include "kws20_measure.h"
#include "kws20_mode_config.h"
#include "ds_cnn_inference.h"

#include "mxc_device.h"

/* Cortex-M4 DWT / CoreDebug registers — from CMSIS core_cm4.h */
#include "core_cm4.h"

#include <stdio.h>
#include <string.h>

#ifndef KWS20_CFG_MEASURE_RUNS
#define KWS20_CFG_MEASURE_RUNS 50
#endif

#define DS_CNN_INPUT_ELEMS   490u
#define DS_CNN_OUTPUT_ELEMS  12u
#define DS_CNN_SAMPLE_RATE   16000u
#define DS_CNN_WINDOW_SAMPLES 16000u

static float measure_input[DS_CNN_INPUT_ELEMS];
static float measure_scores[DS_CNN_OUTPUT_ELEMS];

static void dwt_init(void)
{
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL  |= DWT_CTRL_CYCCNTENA_Msk;
}

void kws20_measure_run_once(void)
{
    uint32_t hclk = SystemCoreClock;

    printf("\r\n==============================\r\n");
    printf("DS-CNN MEASUREMENT MODE\r\n");
    printf("==============================\r\n");

    memset(measure_input, 0, sizeof(measure_input));
    dwt_init();

    printf("BENCH,event=model_info,mode=offline"
           ",hclk_hz=%lu"
           ",input_elems=%u"
           ",output_elems=%u"
           ",sample_rate_hz=%u"
           ",full_window_samples=%u"
           ",runs=%u\r\n",
           (unsigned long)hclk,
           (unsigned int)DS_CNN_INPUT_ELEMS,
           (unsigned int)DS_CNN_OUTPUT_ELEMS,
           (unsigned int)DS_CNN_SAMPLE_RATE,
           (unsigned int)DS_CNN_WINDOW_SAMPLES,
           (unsigned int)KWS20_CFG_MEASURE_RUNS);

    printf("BENCH,event=acquisition,mode=offline"
           ",sample_rate_hz=%u"
           ",sample_count=%u"
           ",audio_window_ms=%u"
           ",feature_frames=49"
           ",feature_bins=10\r\n",
           (unsigned int)DS_CNN_SAMPLE_RATE,
           (unsigned int)DS_CNN_WINDOW_SAMPLES,
           (unsigned int)((DS_CNN_WINDOW_SAMPLES * 1000u) / DS_CNN_SAMPLE_RATE));

    for (uint32_t run = 0; run < KWS20_CFG_MEASURE_RUNS; run++) {
        uint32_t start_cycles;
        uint32_t end_cycles;
        uint32_t cycles;
        uint32_t time_us;
        int pred;

        memset(measure_scores, 0, sizeof(measure_scores));
        DWT->CYCCNT = 0;
        start_cycles = DWT->CYCCNT;
        pred = ds_cnn_infer(measure_input, measure_scores);
        end_cycles = DWT->CYCCNT;

        cycles  = end_cycles - start_cycles;
        time_us = (hclk > 0u)
                  ? (uint32_t)(((uint64_t)cycles * 1000000ULL) / (uint64_t)hclk)
                  : 0u;

        printf("BENCH,event=inference,run=%lu,mode=offline"
               ",cnn_us=%lu,cycles=%lu,pred_idx=%d\r\n",
               (unsigned long)run,
               (unsigned long)time_us,
               (unsigned long)cycles,
               pred);
    }

    printf("BENCH,event=done\r\n");
}
