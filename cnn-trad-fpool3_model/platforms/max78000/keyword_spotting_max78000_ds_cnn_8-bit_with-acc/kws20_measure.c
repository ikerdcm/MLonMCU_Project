/*
 * DS-CNN-L v1 offline benchmark — CNN accelerator, INT8.
 * When RUNS=1: sleep → infer → sleep for clean power-trace isolation.
 * When RUNS>1: back-to-back inferences (latency benchmark).
 */

#include "kws20_measure.h"
#include "kws20_mode_config.h"
#include "cnn_inference.h"
#include "cnn.h"
#include "ds_cnn_test_input_left.h"

#include "mxc_device.h"
#include "tmr.h"
#include "lp.h"
#include "uart.h"
#include "board.h"

#include <stdio.h>
#include <string.h>

#ifndef KWS20_CFG_MEASURE_RUNS
#define KWS20_CFG_MEASURE_RUNS 50
#endif

#define POWER_SLEEP_PRE_MS  500u
#define POWER_SLEEP_POST_MS 500u

#define CNN_INPUT_ELEMS   490u
#define CNN_OUTPUT_ELEMS  12u
#define SAMPLE_RATE       16000u
#define WINDOW_SAMPLES    16000u

static float measure_input[CNN_INPUT_ELEMS];
static float measure_scores[CNN_OUTPUT_ELEMS];

#if KWS20_CFG_MEASURE_RUNS == 1
static volatile int tmr1_fired = 0;

void TMR1_IRQHandler(void)
{
    MXC_TMR_ClearFlags(MXC_TMR1);
    tmr1_fired = 1;
}

static void power_sleep_ms(uint32_t ms)
{
    mxc_tmr_cfg_t cfg;

    MXC_UART_ClearTXFIFO(MXC_UART_GET_UART(CONSOLE_UART));

    tmr1_fired = 0;
    MXC_TMR_Shutdown(MXC_TMR1);

    cfg.pres    = MXC_TMR_PRES_1;
    cfg.mode    = MXC_TMR_MODE_ONESHOT;
    cfg.bitMode = MXC_TMR_BIT_MODE_32;
    cfg.clock   = MXC_TMR_APB_CLK;
    cfg.cmp_cnt = (SystemCoreClock / 2u) / 1000u * ms;
    cfg.pol     = 0;

    MXC_TMR_Init(MXC_TMR1, &cfg, false);
    MXC_TMR_EnableInt(MXC_TMR1);
    NVIC_EnableIRQ(TMR1_IRQn);
    MXC_TMR_Start(MXC_TMR1);

    while (!tmr1_fired)
        MXC_LP_EnterSleepMode();

    NVIC_DisableIRQ(TMR1_IRQn);
    MXC_TMR_Shutdown(MXC_TMR1);
}
#endif /* KWS20_CFG_MEASURE_RUNS == 1 */

void kws20_measure_run_once(void)
{
    uint32_t hclk = SystemCoreClock;

    printf("\r\n==============================\r\n");
    printf("DS-CNN v1 MEASUREMENT MODE\r\n");
    printf("CNN accelerator, INT8\r\n");
    printf("==============================\r\n");

    /* Feed the real "left" reference MFCC (expected idx 2) instead of zeros, so
       the offline bench exercises a true keyword prediction — like the other
       boards' benches — not just latency on a null input. Latency is
       value-independent on the accelerator, so the timing is unchanged. */
    memcpy(measure_input, ds_cnn_test_input_left, sizeof(measure_input));

    printf("BENCH,event=model_info,mode=offline"
           ",hclk_hz=%lu"
           ",input_elems=%u"
           ",output_elems=%u"
           ",sample_rate_hz=%u"
           ",full_window_samples=%u"
           ",runs=%u"
           ",version=v1_cnn_int8\r\n",
           (unsigned long)hclk,
           (unsigned int)CNN_INPUT_ELEMS,
           (unsigned int)CNN_OUTPUT_ELEMS,
           (unsigned int)SAMPLE_RATE,
           (unsigned int)WINDOW_SAMPLES,
           (unsigned int)KWS20_CFG_MEASURE_RUNS);

    printf("BENCH,event=acquisition,mode=offline"
           ",sample_rate_hz=%u"
           ",sample_count=%u"
           ",audio_window_ms=%u"
           ",feature_frames=49"
           ",feature_bins=10\r\n",
           (unsigned int)SAMPLE_RATE,
           (unsigned int)WINDOW_SAMPLES,
           (unsigned int)((WINDOW_SAMPLES * 1000u) / SAMPLE_RATE));

    for (uint32_t run = 0; run < KWS20_CFG_MEASURE_RUNS; run++) {
        int pred;

        memset(measure_scores, 0, sizeof(measure_scores));

#if KWS20_CFG_MEASURE_RUNS == 1
        power_sleep_ms(POWER_SLEEP_PRE_MS);
#endif

        pred = cnn_infer(measure_input, measure_scores);
        /* cnn_time is set by the CNN ISR (µs) */
        uint32_t time_us = cnn_time;
        uint32_t cycles  = (uint32_t)(((uint64_t)time_us * (uint64_t)hclk) / 1000000ULL);

#if KWS20_CFG_MEASURE_RUNS == 1
        power_sleep_ms(POWER_SLEEP_POST_MS);
#endif

        printf("BENCH,event=inference,run=%lu,mode=offline"
               ",cnn_us=%lu,cycles=%lu,pred_idx=%d\r\n",
               (unsigned long)run,
               (unsigned long)time_us,
               (unsigned long)cycles,
               pred);
    }

    printf("BENCH,event=done\r\n");
}
