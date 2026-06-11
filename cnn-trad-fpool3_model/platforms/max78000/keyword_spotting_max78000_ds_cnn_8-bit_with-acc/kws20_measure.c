/*
 * DS-CNN-L v1 offline benchmark — CNN accelerator, INT8.
 * When RUNS=1: periodic LPM sleep → infer → repeat forever for power traces.
 *   The idle window enters Low-Power Mode (CM4 retained, IPO/peripherals
 *   gated) and wakes from the RTC alarm — the APB timers stop in LPM, so the
 *   wake source must be the RTC. The CNN is left enabled/configured (state is
 *   SRAM-retained across LPM), so the inference path is unchanged. This drops
 *   the between-inference baseline well below the ~15 mA of SLEEP mode,
 *   sharpening the per-inference delta for power traces.
 * When RUNS>1: back-to-back inferences (latency benchmark).
 */

#include "kws20_measure.h"
#include "kws20_mode_config.h"
#include "cnn_inference.h"
#include "cnn.h"
#include "ds_cnn_test_input_left.h"

#include "mxc_device.h"
#include "rtc.h"
#include "lp.h"
#include "uart.h"
#include "board.h"

#include <stdio.h>
#include <string.h>

#ifndef KWS20_CFG_MEASURE_RUNS
#define KWS20_CFG_MEASURE_RUNS 50
#endif

#define POWER_SLEEP_PRE_MS   500u
#define POWER_SLEEP_POST_MS  500u
#define POWER_PERIOD_MS      5000u

#define CNN_INPUT_ELEMS   490u
#define CNN_OUTPUT_ELEMS  12u
#define SAMPLE_RATE       16000u
#define WINDOW_SAMPLES    16000u

static float measure_input[CNN_INPUT_ELEMS];
static float measure_scores[CNN_OUTPUT_ELEMS];

#if KWS20_CFG_MEASURE_RUNS == 1
#ifndef KWS20_CFG_DEEP_IDLE
#define KWS20_CFG_DEEP_IDLE 0
#endif

#if KWS20_CFG_DEEP_IDLE
/* Re-power and reconfigure the CNN after a deep-idle power-down. Mirrors the
   one-time init sequence in main(); weights/bias/config are lost when
   cnn_disable() cuts CNN power, so they must be reloaded each wake. */
static void cnn_reinit(void)
{
    cnn_enable(MXC_S_GCR_PCLKDIV_CNNCLKSEL_PCLK, MXC_S_GCR_PCLKDIV_CNNCLKDIV_DIV1);
    cnn_init();
    cnn_load_weights();
    cnn_load_bias();
    cnn_configure();
}
#endif

static volatile int rtc_alarmed = 0;

void RTC_IRQHandler(void)
{
    int flags = MXC_RTC_GetFlags();

    if (flags & MXC_RTC_INT_FL_LONG) {
        MXC_RTC_ClearFlags(MXC_RTC_INT_FL_LONG);
        rtc_alarmed = 1;
    }
}

/* Sleep in LPM for ~ms, waking on the RTC time-of-day alarm. The TOD alarm
   has 1 s granularity, so ms is rounded up to whole seconds (the only caller
   passes a 5000 ms period). RTC is re-initialised to 0 each call so the alarm
   fires `sec` seconds later. */
static void power_sleep_ms(uint32_t ms)
{
    uint32_t sec = (ms + 999u) / 1000u;
    if (sec == 0u)
        sec = 1u;

    /* Don't enter LPM mid-transmit — let the console UART drain first. */
    while (MXC_UART_ReadyForSleep(MXC_UART_GET_UART(CONSOLE_UART)) != E_NO_ERROR) {}

    rtc_alarmed = 0;

    while (MXC_RTC_Init(0, 0) == E_BUSY) {}
    while (MXC_RTC_DisableInt(MXC_RTC_INT_EN_LONG) == E_BUSY) {}
    while (MXC_RTC_SetTimeofdayAlarm(sec) == E_BUSY) {}
    while (MXC_RTC_EnableInt(MXC_RTC_INT_EN_LONG) == E_BUSY) {}
    while (MXC_RTC_Start() == E_BUSY) {}

    NVIC_EnableIRQ(RTC_IRQn);
    MXC_LP_EnableRTCAlarmWakeup();

    while (!rtc_alarmed)
        MXC_LP_EnterLowPowerMode();

    MXC_LP_DisableRTCAlarmWakeup();
    MXC_RTC_Stop();
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

#if KWS20_CFG_MEASURE_RUNS == 1
    for (uint32_t run = 0;; run++) {
        int pred;

#if KWS20_CFG_DEEP_IDLE
        cnn_disable();                 /* power CNN array down for the idle window */
        power_sleep_ms(POWER_PERIOD_MS);
        cnn_reinit();                  /* re-power + reload weights before inference */
#else
        power_sleep_ms(POWER_PERIOD_MS);
#endif
        memset(measure_scores, 0, sizeof(measure_scores));

        pred = cnn_infer(measure_input, measure_scores);
        /* cnn_time is set by the CNN ISR (µs) */
        uint32_t time_us = cnn_time;
        uint32_t cycles  = (uint32_t)(((uint64_t)time_us * (uint64_t)hclk) / 1000000ULL);

        printf("BENCH,event=inference,run=%lu,mode=offline"
               ",cnn_us=%lu,cycles=%lu,pred_idx=%d,period_ms=%u\r\n",
               (unsigned long)run,
               (unsigned long)time_us,
               (unsigned long)cycles,
               pred,
               (unsigned int)POWER_PERIOD_MS);
    }
#else
    for (uint32_t run = 0; run < KWS20_CFG_MEASURE_RUNS; run++) {
        int pred;

        memset(measure_scores, 0, sizeof(measure_scores));

        pred = cnn_infer(measure_input, measure_scores);
        /* cnn_time is set by the CNN ISR (µs) */
        uint32_t time_us = cnn_time;
        uint32_t cycles  = (uint32_t)(((uint64_t)time_us * (uint64_t)hclk) / 1000000ULL);

        printf("BENCH,event=inference,run=%lu,mode=offline"
               ",cnn_us=%lu,cycles=%lu,pred_idx=%d\r\n",
               (unsigned long)run,
               (unsigned long)time_us,
               (unsigned long)cycles,
               pred);
    }

    printf("BENCH,event=done\r\n");
#endif
}
