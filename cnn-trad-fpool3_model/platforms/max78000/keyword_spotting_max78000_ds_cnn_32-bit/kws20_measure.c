/**
 * DS-CNN-L offline benchmark for MAX78000 FTHR.
 * When RUNS=1: sleep → infer → sleep pattern for clean power-trace isolation.
 * When RUNS>1: back-to-back inferences (latency benchmark).
 */

#include "kws20_measure.h"
#include "kws20_mode_config.h"
#include "ds_cnn_inference.h"

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

/* Duration of sleep window before and after inference (power isolation) */
#define POWER_SLEEP_PRE_MS  500u
#define POWER_SLEEP_POST_MS 500u

#define DS_CNN_INPUT_ELEMS    490u
#define DS_CNN_OUTPUT_ELEMS   12u
#define DS_CNN_SAMPLE_RATE    16000u
#define DS_CNN_WINDOW_SAMPLES 16000u

static float measure_input[DS_CNN_INPUT_ELEMS];
static float measure_scores[DS_CNN_OUTPUT_ELEMS];

#if KWS20_CFG_MEASURE_RUNS == 1
/* ── TMR1 one-shot sleep (power-measurement mode only) ───────────────────── */
static volatile int tmr1_fired = 0;

void TMR1_IRQHandler(void)
{
    MXC_TMR_ClearFlags(MXC_TMR1);
    tmr1_fired = 1;
}

/* Put CPU to sleep for `ms` milliseconds using TMR1 + WFI.
 * PCLK = HCLK/2 = 50 MHz → cmp_cnt = 50000 * ms */
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

    while (!tmr1_fired) {
        MXC_LP_EnterSleepMode();
    }

    NVIC_DisableIRQ(TMR1_IRQn);
    MXC_TMR_Shutdown(MXC_TMR1);
}
#endif /* KWS20_CFG_MEASURE_RUNS == 1 */

/* ── Benchmark ───────────────────────────────────────────────────────────── */

void kws20_measure_run_once(void)
{
    uint32_t hclk = SystemCoreClock;

    printf("\r\n==============================\r\n");
    printf("DS-CNN MEASUREMENT MODE\r\n");
    printf("==============================\r\n");

    memset(measure_input, 0, sizeof(measure_input));

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
        uint32_t time_us;
        uint32_t cycles;
        int pred;

        memset(measure_scores, 0, sizeof(measure_scores));

#if KWS20_CFG_MEASURE_RUNS == 1
        /* Power-measurement mode: sleep before inference so the active
         * inference window is clearly isolated in the power trace. */
        power_sleep_ms(POWER_SLEEP_PRE_MS);
#endif

        MXC_TMR_SW_Start(MXC_TMR0);
        pred = ds_cnn_infer(measure_input, measure_scores);
        time_us = MXC_TMR_SW_Stop(MXC_TMR0);

        cycles = (uint32_t)(((uint64_t)time_us * (uint64_t)hclk) / 1000000ULL);

#if KWS20_CFG_MEASURE_RUNS == 1
        /* Sleep after inference to mark the end of the active window. */
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
