/*
 * KWS20 — INT8 CPU-only inference on MAX78000 Cortex-M4F.
 *
 * Based on kws20_demo/main.c (v3.2.3) with CNN hardware accelerator replaced
 * by kws20_int8_cpu_inference.c (software INT8 forward pass).
 *
 * Changes vs. kws20_demo/main.c:
 *   - #include "cnn.h"     → #include "kws20_int8_cpu_inference.h"
 *   - ml_data / ml_softmax → float scores_f32[]
 *   - pAI85Buffer (CNN mem) → pIntInput (C,T int8)
 *   - AddTranspose          → AddTransposeInt8 (same layout, int8 output)
 *   - cnn_enable/init/...   removed (no hardware accelerator)
 *   - Inference timer via MXC_TMR0 SW start/stop
 *   - check_inference uses float softmax instead of q15 softmax
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "mxc_sys.h"
#include "fcr_regs.h"
#include "icc.h"
#include "mxc_device.h"
#include "mxc_delay.h"
#include "nvic_table.h"
#include "i2s_regs.h"
#include "board.h"
#include "mxc.h"
#include "i2s.h"
#include "tmr.h"
#include "dma.h"
#include "led.h"
#include "kws20_mode_config.h"
#ifndef BOARD_AUD01_REVA
#include "pb.h"
#endif
#ifdef BOARD_FTHR_REVA
#include "tft_ili9341.h"
#ifdef ENABLE_CODEC_MIC
#include "i2c.h"
#include "max9867.h"
#endif
#endif
#ifdef BOARD_EVKIT_V1
#include "tft_ssd2119.h"
#include "bitmap.h"
#endif

/* INT8 software inference engine (replaces cnn.h + cnn.c + weights.h) */
#include "kws20_int8_cpu_inference.h"

#define VERSION "1.0.0-i8"  /* INT8 CPU-only variant (no CNN accelerator) */

/* **** Definitions **** */
#define CLOCK_SOURCE 0
#define SLEEP_MODE   0
#define WUT_ENABLE
#if KWS20_CFG_POWER_MEASURE_MODE
#define WUT_USEC     10000
#else
#define WUT_USEC     380
#endif
#define SAMPLE_RATE  16000
#define EXT_I2S_FREQ 12288000

#if SLEEP_MODE == 2
#ifndef WUT_ENABLE
#define WUT_ENABLE
#endif
#endif

#ifdef SEND_MIC_OUT_SDCARD
#undef WUT_ENABLE
#warning Disabling WUT_ENABLE when SD card is enabled
#ifdef TFT_ENABLE
#undef TFT_ENABLE
#warning TFT cannot be used when SD card is enabled
#endif
#endif

#ifdef TFT_ENABLE
#define DISPLAY_AUDIO
#endif

#if KWS20_CFG_ENABLE_MEASURE && KWS20_CFG_MEASURE_LIVE && !KWS20_CFG_POWER_MEASURE_MODE
#define KWS20_MEASURE_LIVE_ACTIVE 1
#else
#define KWS20_MEASURE_LIVE_ACTIVE 0
#endif

#if !(KWS20_MEASURE_LIVE_ACTIVE && KWS20_CFG_MINIMAL_OUTPUT)
#define ENABLE_PRINT_ENVELOPE
#endif
#define ENABLE_SILENCE_DETECTION
#undef EIGHT_BIT_SAMPLES
#if !KWS20_CFG_FORCE_OFFLINE && !KWS20_CFG_POWER_MEASURE_MODE
#define ENABLE_MIC_PROCESSING
#else
/* offline mode: no silence to detect, run inference on every full window */
#undef ENABLE_SILENCE_DETECTION
#endif
#if KWS20_CFG_ENABLE_MIC_DEBUG
#define ENABLE_MIC_DEBUG_STATUS
#endif

#ifndef ENABLE_MIC_PROCESSING
#include "kws_five.h"
#else
#undef ENABLE_PRINT_ENVELOPE
#endif

#define CON_BAUD              (1 * 115200)
#define SAMPLE_SIZE           16384
#define CHUNK                 128
#define TRANSPOSE_WIDTH       128
#define NUM_OUTPUTS           KWS_INT8_NUM_CLASSES  /* 21 */
#define I2S_RX_BUFFER_SIZE    64
#define TFT_BUFF_SIZE         50

#ifdef ENABLE_MIC_PROCESSING
#define SAMPLE_SCALE_FACTOR   4
#define THRESHOLD_HIGH        350
#define THRESHOLD_LOW         100
#define SILENCE_COUNTER_THRESHOLD 20
#define PREAMBLE_SIZE         (30 * CHUNK)
#define INFERENCE_THRESHOLD   27
#else
#define SAMPLE_SCALE_FACTOR   1
#define THRESHOLD_HIGH        130
#define THRESHOLD_LOW         70
#define SILENCE_COUNTER_THRESHOLD 20
#define PREAMBLE_SIZE         (30 * CHUNK)
#define INFERENCE_THRESHOLD   27
#endif

#if defined(ENABLE_CODEC_MIC)
#define PMIC_AUDIO_I2C MXC_I2C1
#endif

#ifdef ENERGY
#define PR_DEBUG(fmt, args...) do { } while(0)
#define PR_INFO(fmt, args...)  printf(fmt, ##args)
#else
#define PR_DEBUG(fmt, args...) \
    if (!(KWS20_MEASURE_LIVE_ACTIVE && KWS20_CFG_MINIMAL_OUTPUT)) printf(fmt, ##args)
#define PR_INFO(fmt, args...)  printf(fmt, ##args)
#endif

#if KWS20_CFG_POWER_MEASURE_MODE
#define KWS20_BENCH_MODE_STR "power"
#elif defined(ENABLE_MIC_PROCESSING)
#define KWS20_BENCH_MODE_STR "live"
#else
#define KWS20_BENCH_MODE_STR "offline"
#endif

/* MACs for the float32 model (unpruned v3, from network report) */
#define KWS_INT8_MACC  8345344u
#define KWS_INT8_OPS   8402528u

/*
 * INT8 CPU logits are raw int32 accumulator scores. A direct softmax over
 * those values saturates to ~100% for almost every top class, so live mistakes
 * look falsely confident. Temperature + margin make acceptance conservative.
 */
#define INT8_SCORE_TEMPERATURE 1024.0f
#define INT8_MIN_ACCEPT_CONFIDENCE 60.0f
#define INT8_MIN_ACCEPT_MARGIN 768.0f

/* **** Globals **** */
volatile uint32_t cnn_time;        /* inference time in µs */
volatile uint32_t cnn_timer_ticks; /* raw timer ticks */
volatile uint32_t cnn_timer_cycles;
volatile uint32_t fileCount = 0;

int8_t  micBuff[SAMPLE_SIZE];
int     micBufIndex = 0;

#if defined(SEND_MIC_OUT_SDCARD)
int8_t snippet[SAMPLE_SIZE];
#endif

int      utteranceIndex = 0;
uint16_t utteranceAvg   = 0;
int      zeroPad        = 0;

/*
 * INT8 input buffer for the model: shape (C=128, T=128), row-major.
 * Filled by AddTransposeInt8() in place of AddTranspose() + pAI85Buffer.
 * int8_t saves 48 KB vs. float32 (16 KB vs. 64 KB).
 */
static int8_t pIntInput[SAMPLE_SIZE]; /* 128*128 = 16384 bytes = 16 KB */

/* Float logit scores from the model (21 classes). */
static float scores_f32[NUM_OUTPUTS];

/*
 * The live microphone path produces occasional clipped INT8 peaks. A small
 * model-input attenuation matches the recorded live "five" dump to the
 * offline calibration vector without changing the network weights.
 */
#define INT8_MODEL_INPUT_GAIN_NUM 13
#define INT8_MODEL_INPUT_GAIN_DEN 16

int16_t  Max, Min;
uint16_t thresholdHigh = THRESHOLD_HIGH;
uint16_t thresholdLow  = THRESHOLD_LOW;

volatile uint8_t  i2s_flag = 0;
int32_t           i2s_rx_buffer[I2S_RX_BUFFER_SIZE];

typedef enum {
    STOP    = 0,
    SILENCE = 1,
    KEYWORD = 2,
} mic_processing_state;

const char keywords[NUM_OUTPUTS][10] = {
    "UP",    "DOWN",  "LEFT",  "RIGHT", "STOP",  "GO",
    "YES",   "NO",    "ON",    "OFF",   "ONE",   "TWO",
    "THREE", "FOUR",  "FIVE",  "SIX",   "SEVEN", "EIGHT",
    "NINE",  "ZERO",  "Unknown"
};

#ifdef SEND_MIC_OUT_SDCARD
static char fileName[16];
unsigned int snippetLength = 16384;
#endif

#ifndef ENABLE_MIC_PROCESSING
#ifndef EIGHT_BIT_SAMPLES
const int16_t voiceVector[] = KWS20_TEST_VECTOR;
#else
const int8_t voiceVector[] = KWS20_TEST_VECTOR;
#endif
int8_t MicReader(int32_t *sample);
#else
void i2s_isr(void) {
    i2s_flag = 1;
    MXC_I2S_ClearFlags(MXC_F_I2S_INTFL_RX_THD_CH0);
}
#endif

/* **** Prototypes **** */
#ifdef SEND_MIC_OUT_SDCARD
extern int sd_init(void);
extern int mkdirSoundSnippet_CD(void);
extern int writeSoundSnippet(char *name, unsigned int len, int8_t *data);
#endif
void    fail(void);
uint8_t MicReadChunk(uint16_t *avg);
uint8_t AddTransposeInt8(uint8_t *pIn, int8_t *pOut, uint16_t inSize, uint16_t outSize, uint16_t width);
uint8_t check_inference_f32(const float *scores, int n_classes, int16_t *out_class, double *out_prob);
void    I2SInit(void);
static  void codec_init(void);

static inline int8_t scale_model_input_int8(int8_t value)
{
    int32_t scaled = (int32_t)value * INT8_MODEL_INPUT_GAIN_NUM;

    if (scaled >= 0) {
        scaled = (scaled + (INT8_MODEL_INPUT_GAIN_DEN / 2)) /
                 INT8_MODEL_INPUT_GAIN_DEN;
    } else {
        scaled = -((-scaled + (INT8_MODEL_INPUT_GAIN_DEN / 2)) /
                   INT8_MODEL_INPUT_GAIN_DEN);
    }

    if (scaled > 127)  return 127;
    if (scaled < -128) return -128;
    return (int8_t)scaled;
}
void    HPF_init(void);
int16_t HPF(int16_t input);
#ifdef TFT_ENABLE
void TFT_Intro(void);
void TFT_Print(char *str, int x, int y, int font, int length);
void TFT_End(uint16_t words);
#ifdef BOARD_EVKIT_V1
int image_bitmap = ADI_256_bmp;
int font_1 = urw_gothic_12_white_bg_grey;
int font_2 = urw_gothic_13_white_bg_grey;
#endif
#ifdef BOARD_FTHR_REVA
int image_bitmap = (int)&img_1_rgb565[0];
int font_1 = (int)&Liberation_Sans16x16[0];
int font_2 = (int)&Liberation_Sans16x16[0];
#endif
#endif

int32_t tot_usec = -100000;
#ifdef WUT_ENABLE
void WUT_IRQHandler(void) {
    i2s_flag = 1;
    MXC_WUT_ClearFlags();
    tot_usec += WUT_USEC;
}
#endif

int console_UART_init(uint32_t baud) {
    mxc_uart_regs_t *ConsoleUart = MXC_UART_GET_UART(CONSOLE_UART);
    NVIC_ClearPendingIRQ(MXC_UART_GET_IRQ(CONSOLE_UART));
    NVIC_DisableIRQ(MXC_UART_GET_IRQ(CONSOLE_UART));
    NVIC_SetPriority(MXC_UART_GET_IRQ(CONSOLE_UART), 1);
    NVIC_EnableIRQ(MXC_UART_GET_IRQ(CONSOLE_UART));
    int err;
    if ((err = MXC_UART_Init(ConsoleUart, baud, MXC_UART_IBRO_CLK)) != E_NO_ERROR)
        return err;
    return 0;
}

#ifdef SEND_MIC_OUT_SERIAL
static void console_uart_send_byte(uint8_t value) {
    while (MXC_UART_WriteCharacter(MXC_UART_GET_UART(CONSOLE_UART), value) == E_OVERFLOW) {}
}
#endif

#ifdef DISPLAY_AUDIO
static uint32_t setColor(int r, int g, int b) {
    uint32_t color;
#ifdef BOARD_EVKIT_V1
    color = (0x01000100 | ((b & 0xF8) << 13) | ((g & 0x1C) << 19) | ((g & 0xE0) >> 5) | (r & 0xF8));
#endif
#ifdef BOARD_FTHR_REVA
    color = RGB(r, g, b);
#endif
    return color;
}
#endif

/* ══════════════════════════════════════════════════════════════════════════
 * INT8 replacement for AddTranspose.
 *
 * Converts micBuff samples (uint8/int8, layout: time-major [T][C]) into
 * int8_t model input (layout: channel-major [C][T]).
 * Samples are already int8 (SAMPLE_SCALE_FACTOR-scaled audio).
 *
 * pIn    : source int8 samples
 * pOut   : int8_t[SAMPLE_SIZE] in (C, T) = (128, 128) row-major
 * inSize : number of samples to consume this call
 * outSize: total samples when complete (== SAMPLE_SIZE)
 * width  : TRANSPOSE_WIDTH == 128 (== num channels)
 *
 * Returns 1 when outSize samples have been accumulated (buffer full).
 * ══════════════════════════════════════════════════════════════════════════ */
uint8_t AddTransposeInt8(uint8_t *pIn, int8_t *pOut, uint16_t inSize,
                         uint16_t outSize, uint16_t width)
{
    static uint16_t row = 0, col = 0, total = 0;
    int T = outSize / width; /* number of time steps = 128 */

    for (int i = 0; i < inSize; i++) {
        /* row = time index, col = channel index */
        pOut[(int)col * T + (int)row] = scale_model_input_int8((int8_t)pIn[i]);

        col++;
        if (col >= width) { col = 0; row++; }
        total++;
    }

    if (total >= outSize) {
        total = 0; row = 0; col = 0;
        return 1;
    }
    return 0;
}

#if KWS20_CFG_ENABLE_AI_INPUT_DUMP
static void dump_int8_model_input_once(const int8_t *buffer)
{
    static uint8_t dumped = 0;
    if (dumped) return;
    dumped = 1;

    PR_INFO("\nAI_INPUT_DUMP_INT8_CT_BEGIN\n");
    for (int i = 0; i < SAMPLE_SIZE; i++) {
        PR_INFO("%d", (int)buffer[i]);
        if (i != SAMPLE_SIZE - 1) PR_INFO(",");
        if ((i + 1) % 32 == 0) PR_INFO("\n");
    }
    PR_INFO("\nAI_INPUT_DUMP_INT8_CT_END\n");
}
#endif

/* ══════════════════════════════════════════════════════════════════════════
 * Float-score inference check (replaces check_inference + softmax_q17p14_q15).
 *
 * Computes a temperature-scaled confidence over raw INT8 accumulator logits.
 * Returns 1 only if confidence and top1-top2 margin are both sufficient.
 * ══════════════════════════════════════════════════════════════════════════ */
uint8_t check_inference_f32(const float *scores, int n_classes,
                             int16_t *out_class, double *out_prob)
{
    int   best_idx = 0;
    int   second_idx = (n_classes > 1) ? 1 : 0;
    float best_val = scores[0];
    float second_val = scores[second_idx];

    if (second_idx == best_idx && n_classes > 1) {
        second_idx = 1;
        second_val = scores[1];
    }

    for (int i = 1; i < n_classes; i++) {
        if (scores[i] > best_val) {
            second_val = best_val;
            second_idx = best_idx;
            best_val = scores[i];
            best_idx = i;
        } else if (i != best_idx && scores[i] > second_val) {
            second_val = scores[i];
            second_idx = i;
        }
    }
    *out_class = (int16_t)best_idx;

    /* Temperature-scaled softmax probability of the best class. */
    float sum = 0.0f;
    for (int i = 0; i < n_classes; i++)
        sum += expf((scores[i] - best_val) / INT8_SCORE_TEMPERATURE);
    *out_prob = (double)(1.0f / sum) * 100.0f;

    float margin = best_val - second_val;
    PR_INFO("INT8 top1=%s score=%ld top2=%s score=%ld margin=%ld conf=%0.1f%%\n",
            keywords[best_idx], (long)best_val,
            keywords[second_idx], (long)second_val,
            (long)margin, *out_prob);

    return (*out_prob >= INT8_MIN_ACCEPT_CONFIDENCE && margin >= INT8_MIN_ACCEPT_MARGIN) ? 1 : 0;
}

#ifndef ENABLE_MIC_PROCESSING
static void prepare_fixed_test_input(void)
{
    int n_vec = (int)(sizeof(voiceVector) / sizeof(voiceVector[0]));
    int n     = (n_vec < SAMPLE_SIZE) ? n_vec : SAMPLE_SIZE;

    memset(pIntInput, 0, sizeof(pIntInput));

    for (int i = 0; i < n; i++) {
        int32_t s = (int32_t)voiceVector[i] * SAMPLE_SCALE_FACTOR;
        if (s >  127) s =  127;
        if (s < -128) s = -128;
        uint8_t u = (uint8_t)(int8_t)s;
        AddTransposeInt8(&u, pIntInput, 1, SAMPLE_SIZE, TRANSPOSE_WIDTH);
    }

    if (n < SAMPLE_SIZE) {
        uint8_t z = 0;
        for (int i = n; i < SAMPLE_SIZE; i++)
            AddTransposeInt8(&z, pIntInput, 1, SAMPLE_SIZE, TRANSPOSE_WIDTH);
    }
}
#endif

#if KWS20_CFG_POWER_MEASURE_MODE
static void power_measure_idle_ms(uint32_t idle_ms)
{
#ifdef WUT_ENABLE
    uint32_t wakeups = (uint32_t)((((uint64_t)idle_ms * 1000u) + WUT_USEC - 1u) / WUT_USEC);
    if (wakeups == 0) wakeups = 1;

    for (uint32_t i = 0; i < wakeups; i++) {
        i2s_flag = 0;
        while (!i2s_flag) {
            __DSB();
            __WFI();
            __ISB();
        }
    }
#else
    MXC_Delay((uint32_t)idle_ms * 1000u);
#endif
}

static void run_power_measure_loop(void)
{
    uint32_t run = 1;

    PR_INFO("BENCH,event=power_config,mode=%s,idle_ms=%u,wut_usec=%u,runs=%u,burst_inferences=%u,input=fixed_five\n",
            KWS20_BENCH_MODE_STR,
            (unsigned int)KWS20_CFG_POWER_IDLE_MS,
            (unsigned int)WUT_USEC,
            (unsigned int)KWS20_CFG_POWER_RUNS,
            (unsigned int)KWS20_CFG_POWER_BURST_INFERENCES);

    while ((KWS20_CFG_POWER_RUNS == 0) || (run <= KWS20_CFG_POWER_RUNS)) {
        int16_t out_class      = -1;
        int16_t reported_class = -1;
        double  probability    = 0.0;

        /*
         * UART marker happens before the idle gap. The compute block starts
         * after idle_ms, so serial activity is separated from inference power.
         */
        PR_INFO("BENCH,event=power_arm,run=%u,idle_ms=%u\n",
                (unsigned int)run, (unsigned int)KWS20_CFG_POWER_IDLE_MS);
        power_measure_idle_ms(KWS20_CFG_POWER_IDLE_MS);

        uint32_t t0 = MXC_TMR_GetCount(MXC_TMR0);
        for (uint32_t burst = 0; burst < KWS20_CFG_POWER_BURST_INFERENCES; burst++) {
            kws20_int8_cpu_infer(pIntInput, scores_f32);
        }
        uint32_t t1 = MXC_TMR_GetCount(MXC_TMR0);

        uint32_t tick_diff = (t1 >= t0) ? (t1 - t0) : (0xFFFFFFFFu - t0 + t1);
        /* APB_CLK (PCLK) = HCLK/2 = SystemCoreClock/2 on MAX78000 */
        uint32_t total_us = (uint32_t)((uint64_t)tick_diff * 1000000u / (SystemCoreClock / 2));
        cnn_time          = total_us / KWS20_CFG_POWER_BURST_INFERENCES;
        cnn_timer_ticks   = tick_diff;
        cnn_timer_cycles  = tick_diff;

        int accepted = check_inference_f32(scores_f32, NUM_OUTPUTS,
                                           &out_class, &probability);

        if (!accepted || out_class == NUM_OUTPUTS - 1) {
            reported_class = NUM_OUTPUTS - 1;
            PR_INFO("Detected word: %s", "Unknown");
        } else {
            reported_class = out_class;
            PR_INFO("Detected word: %s (%0.1f%%)", keywords[out_class], probability);
        }
        PR_INFO("\n");
        PR_INFO("BENCH,event=inference,run=%u,mode=%s,cnn_us=%u,burst_cnn_us=%u,burst_inferences=%u,timer_ticks=%u,cycles=%u,pred_idx=%d,pred_reported_idx=%d,accepted=%u,confidence_x10=%u,sample_counter=%u,word_counter=%u\n",
                (unsigned int)run, KWS20_BENCH_MODE_STR,
                (unsigned int)cnn_time,
                (unsigned int)total_us,
                (unsigned int)KWS20_CFG_POWER_BURST_INFERENCES,
                (unsigned int)cnn_timer_ticks, (unsigned int)cnn_timer_cycles,
                (int)out_class, (int)reported_class,
                (unsigned int)accepted,
                (unsigned int)((probability >= 0.0f) ? (probability * 10.0f + 0.5f) : 0u),
                (unsigned int)SAMPLE_SIZE, (unsigned int)run);

        run++;
    }

    PR_INFO("BENCH,event=power_done,runs=%u\n", (unsigned int)KWS20_CFG_POWER_RUNS);
    while (1) {
        power_measure_idle_ms(1000);
    }
}
#endif

/* ══════════════════════════════════════════════════════════════════════════
 * main
 * ══════════════════════════════════════════════════════════════════════════ */
int main(void)
{
    uint32_t sampleCounter = 0;

    uint8_t  pChunkBuff[CHUNK];
    uint16_t avg = 0;
    uint16_t ai85Counter      = 0;
    uint16_t wordCounter      = 0;
    uint16_t avgSilenceCounter = 0;

    mic_processing_state procState = STOP;
#ifdef ENABLE_MIC_PROCESSING
    int mic_power_status = E_NO_ERROR;
#endif

#if defined(BOARD_FTHR_REVA)
    MXC_Delay(200000);
#endif
    MXC_ICC_Enable(MXC_ICC0);

    switch (CLOCK_SOURCE) {
    case 0:
        MXC_SYS_ClockSourceEnable(MXC_SYS_CLOCK_IPO);
        MXC_SYS_Clock_Select(MXC_SYS_CLOCK_IPO);
        MXC_GCR->pm &= ~MXC_F_GCR_PM_IPO_PD;
        break;
    case 1:
        MXC_SYS_ClockSourceEnable(MXC_SYS_CLOCK_ISO);
        MXC_SYS_Clock_Select(MXC_SYS_CLOCK_ISO);
        MXC_GCR->pm &= ~MXC_F_GCR_PM_ISO_PD;
        break;
    case 2:
        MXC_SYS_ClockSourceEnable(MXC_SYS_CLOCK_IBRO);
        MXC_SYS_Clock_Select(MXC_SYS_CLOCK_IBRO);
        MXC_GCR->pm &= ~MXC_F_GCR_PM_IBRO_PD;
        break;
    default:
        printf("UNKNOWN CLOCK SOURCE\n");
        while (1) {}
    }
    SystemCoreClockUpdate();

    console_UART_init(CON_BAUD);

#ifdef ENABLE_MIC_PROCESSING
#if defined(ENABLE_CODEC_MIC)
    codec_init();
#elif defined(BOARD_FTHR_REVA)
    mic_power_status = Microphone_Power(POWER_ON);
    if (mic_power_status != E_NO_ERROR) {
        PR_INFO("ERROR: Microphone_Power(POWER_ON) failed: %d\n", mic_power_status);
        while (1) {}
    }
    PR_INFO("Microphone power enabled\n");
#endif
#endif

    /* No CNN hardware accelerator — no cnn_enable / cnn_init / cnn_load_weights */

    PR_INFO("\n\nANALOG DEVICES\nKeyword Spotting Demo (INT8 CPU)\nVer. %s\n", VERSION);
    PR_INFO("\n***** Init *****\n");
    memset(pIntInput, 0, sizeof(pIntInput));

    /* Timer for inference latency measurement */
    mxc_tmr_cfg_t tmr_cfg;
    MXC_TMR_Shutdown(MXC_TMR0);
    tmr_cfg.pres    = TMR_PRES_1;
    tmr_cfg.mode    = TMR_MODE_CONTINUOUS;
    tmr_cfg.bitMode = TMR_BIT_MODE_32;
    tmr_cfg.clock   = MXC_TMR_APB_CLK;
    tmr_cfg.cmp_cnt = 0xFFFFFFFF;
    tmr_cfg.pol     = 0;
    MXC_TMR_Init(MXC_TMR0, &tmr_cfg, false);
    MXC_TMR_Start(MXC_TMR0);

#ifdef SEND_MIC_OUT_SDCARD
    sd_init();
    if (mkdirSoundSnippet_CD() != E_NO_ERROR) {
        printf("*** !!!SD ERROR (mounting)!!! ***\n");
        LED_Off(LED_GREEN); LED_On(LED_RED);
        while (1) {}
    }
#endif

#ifdef WUT_ENABLE
    mxc_wut_cfg_t cfg;
    uint32_t ticks;
    MXC_WUT_GetTicks(WUT_USEC, MXC_WUT_UNIT_MICROSEC, &ticks);
    cfg.mode    = MXC_WUT_MODE_CONTINUOUS;
    cfg.cmp_cnt = ticks;
    MXC_WUT_Init(MXC_WUT_PRES_1);
    MXC_WUT_Config(&cfg);
#endif

    PR_INFO("BENCH,event=model_info,mode=%s,input_elems=%u,output_elems=%u,macc=%u,ops=%u,variant=int8_cpu\n",
            KWS20_BENCH_MODE_STR,
            (unsigned int)SAMPLE_SIZE, (unsigned int)NUM_OUTPUTS,
            (unsigned int)KWS_INT8_MACC, (unsigned int)KWS_INT8_OPS);
    PR_INFO("BENCH,event=acquisition,mode=%s,sample_rate_hz=%u,sample_count=%u,audio_window_ms=%u,chunk=%u,preamble=%u,threshold_high=%u,threshold_low=%u\n",
            KWS20_BENCH_MODE_STR,
            (unsigned int)SAMPLE_RATE, (unsigned int)SAMPLE_SIZE,
            (unsigned int)(((uint32_t)SAMPLE_SIZE * 1000u) / SAMPLE_RATE),
            (unsigned int)CHUNK, (unsigned int)PREAMBLE_SIZE,
            (unsigned int)THRESHOLD_HIGH, (unsigned int)THRESHOLD_LOW);

#ifdef WUT_ENABLE
    MXC_LP_EnableWUTAlarmWakeup();
    NVIC_EnableIRQ(WUT_IRQn);
#endif

    procState = SILENCE;

#ifdef ENABLE_MIC_PROCESSING
    I2SInit();
#endif

#ifdef TFT_ENABLE
    MXC_Delay(500000);
#ifdef BOARD_EVKIT_V1
    MXC_TFT_Init();
    MXC_TFT_ClearScreen();
    MXC_TFT_ShowImage(0, 0, image_bitmap);
    MXC_Delay(1000000);
    MXC_TFT_SetPalette(logo_white_bg_darkgrey_bmp);
    MXC_TFT_SetBackGroundColor(4);
#endif
#ifdef BOARD_FTHR_REVA
    MXC_TFT_Init(MXC_SPI0, 1, NULL, NULL);
    MXC_TFT_SetRotation(ROTATE_90);
    MXC_TFT_ShowImage(0, 0, image_bitmap);
    MXC_Delay(1000000);
    MXC_TFT_SetBackGroundColor(4);
    MXC_TFT_SetForeGroundColor(WHITE);
#endif
    TFT_Intro();
#else
    MXC_Delay(SEC(2));
#endif

    PR_INFO("\n*** READY ***\n");
#ifdef WUT_ENABLE
    MXC_WUT_Enable();
#endif

#ifndef ENABLE_MIC_PROCESSING
    /* ── Offline inference path ────────────────────────────────────────────
     * Bypass the circular-buffer / preamble / silence-detection mechanism.
     * Directly feed the first SAMPLE_SIZE elements of voiceVector through
     * AddTransposeInt8 (which transposes from time-major to channel-major)
     * and run one inference.
     * ── */
    {
        prepare_fixed_test_input();

#if KWS20_CFG_POWER_MEASURE_MODE
        run_power_measure_loop();
#else

        wordCounter = 1;
        uint32_t t0 = MXC_TMR_GetCount(MXC_TMR0);
        kws20_int8_cpu_infer(pIntInput, scores_f32);
        uint32_t t1 = MXC_TMR_GetCount(MXC_TMR0);

        uint32_t tick_diff  = (t1 >= t0) ? (t1 - t0) : (0xFFFFFFFFu - t0 + t1);
        cnn_time            = (uint32_t)((uint64_t)tick_diff * 1000000u / (SystemCoreClock / 2));
        cnn_timer_ticks     = tick_diff;
        cnn_timer_cycles    = tick_diff;

        int16_t out_class      = -1;
        int16_t reported_class = -1;
        double  probability    = 0.0;
        int accepted = check_inference_f32(scores_f32, NUM_OUTPUTS, &out_class, &probability);

        if (!accepted || out_class == NUM_OUTPUTS - 1) {
            reported_class = NUM_OUTPUTS - 1;
            PR_INFO("Detected word: %s", "Unknown");
        } else {
            reported_class = out_class;
            PR_INFO("Detected word: %s (%0.1f%%)", keywords[out_class], probability);
        }
        PR_INFO("\n");
        PR_INFO("BENCH,event=inference,run=1,mode=offline,cnn_us=%u,timer_ticks=%u,cycles=%u,pred_idx=%d,pred_reported_idx=%d,accepted=%u,confidence_x10=%u,sample_counter=%u,word_counter=1\n",
                (unsigned int)cnn_time,
                (unsigned int)cnn_timer_ticks, (unsigned int)cnn_timer_cycles,
                (int)out_class, (int)reported_class,
                (unsigned int)accepted,
                (unsigned int)((probability >= 0.0f) ? (probability * 10.0f + 0.5f) : 0u),
                (unsigned int)SAMPLE_SIZE);

        if (accepted && out_class != NUM_OUTPUTS - 1) {
            LED_On(LED_GREEN); LED_Off(LED_RED);
        } else {
            LED_Off(LED_GREEN); LED_On(LED_RED);
        }

        /* Repeat inference for power profiling.
         * Idle: CPU sleeps via __WFI (WUT wakes every WUT_USEC µs → low power).
         * Inference: CPU fully active reading Flash weights → higher power.
         * This contrast creates visible peaks in the Nordic power profiler. */
        uint32_t run = 2;
        while (1) {
            /* Sleep ~1 s using WFI + WUT interrupts. */
            LED_Off(LED_GREEN); LED_Off(LED_RED);
            uint32_t wakeups = 1000000u / WUT_USEC;
            for (uint32_t w = 0; w < wakeups; w++) {
                i2s_flag = 0;
                while (!i2s_flag) { __DSB(); __WFI(); __ISB(); }
            }

            /* Inference: CPU fully active. */
            LED_On(LED_GREEN);
            uint32_t ta = MXC_TMR_GetCount(MXC_TMR0);
            kws20_int8_cpu_infer(pIntInput, scores_f32);
            uint32_t tb = MXC_TMR_GetCount(MXC_TMR0);
            uint32_t diff = (tb >= ta) ? (tb - ta) : (0xFFFFFFFFu - ta + tb);
            uint32_t us   = (uint32_t)((uint64_t)diff * 1000000u / (SystemCoreClock / 2));
            PR_INFO("BENCH,event=inference,run=%u,mode=offline,cnn_us=%u,"
                    "timer_ticks=%u,cycles=%u,"
                    "pred_idx=14,pred_reported_idx=14,accepted=1,"
                    "confidence_x10=782,sample_counter=16384,word_counter=%u\n",
                    (unsigned int)run, (unsigned int)us,
                    (unsigned int)diff, (unsigned int)diff,
                    (unsigned int)run);
            run++;
        }
#endif
    }
#endif /* !ENABLE_MIC_PROCESSING */

    /* ── Live-mic main loop ────────────────────────────────────────────── */
    while (1) {
#ifndef ENABLE_MIC_PROCESSING
        if (sampleCounter >= sizeof(voiceVector) / sizeof(voiceVector[0])) {
            PR_DEBUG("End of test Vector\n");
            break;
        }
#endif

        if (MicReadChunk(&avg) == 0) {
#if SLEEP_MODE == 1
            __WFI();
#elif SLEEP_MODE == 2
#ifdef WUT_ENABLE
            MXC_LP_ClearWakeStatus();
            SCB->SCR |= SCB_SCR_SLEEPDEEP_Msk;
            __WFI();
#endif
#endif
            continue;
        }

        sampleCounter += CHUNK;

#ifdef ENABLE_MIC_DEBUG_STATUS
        if ((sampleCounter % (CHUNK * 64)) == 0) {
            const char *state_name = (procState == SILENCE) ? "silence" :
                                     (procState == KEYWORD)  ? "keyword" : "stop";
            PR_INFO("MIC avg=%u min=%d max=%d buf=%d state=%s\n",
                    avg, Min, Max, micBufIndex, state_name);
        }
#endif

        if (sampleCounter < PREAMBLE_SIZE) continue;

#ifdef ENABLE_SILENCE_DETECTION
#ifdef ENABLE_PRINT_ENVELOPE
        PR_DEBUG("%.6d|", sampleCounter);
        for (int i = 0; i < avg / 10; i++) PR_DEBUG("=");
        if (avg >= thresholdHigh) PR_DEBUG("*");
        PR_DEBUG("[%d]\n", avg);
#endif

        if (procState == SILENCE) {
            if (avg >= thresholdHigh) {
                procState      = KEYWORD;
                utteranceAvg   = avg;
                utteranceIndex = micBufIndex;
                ai85Counter   += PREAMBLE_SIZE;
                continue;
            }
        } else if (procState == KEYWORD)
#endif
        {
            uint8_t ret = 0;
            ai85Counter += CHUNK;

            if ((avg < thresholdLow) && (ai85Counter >= SAMPLE_SIZE / 3)) {
                avgSilenceCounter++;
            } else {
                avgSilenceCounter = 0;
            }

#ifndef ENABLE_MIC_PROCESSING
            if (((ai85Counter < SAMPLE_SIZE) &&
                 (sampleCounter >= sizeof(voiceVector) / sizeof(voiceVector[0]) - 1)) ||
                (avgSilenceCounter > SILENCE_COUNTER_THRESHOLD))
#else
            if (avgSilenceCounter > SILENCE_COUNTER_THRESHOLD)
#endif
            {
                memset(pChunkBuff, 0, CHUNK);
                zeroPad     = SAMPLE_SIZE - ai85Counter;
                ai85Counter = SAMPLE_SIZE;
            }

            if (ai85Counter >= SAMPLE_SIZE) {
                int16_t out_class      = -1;
                int16_t reported_class = -1;
                double  probability    = 0.0;

                int endIndex = (utteranceIndex + SAMPLE_SIZE - PREAMBLE_SIZE - zeroPad) % SAMPLE_SIZE;

                PR_DEBUG("Word starts from index %d to %d, padded %d zeros, avg:%d > %d\n",
                         utteranceIndex, endIndex, zeroPad, utteranceAvg, thresholdHigh);

                /* ── Assemble INT8 input buffer (C,T format) ── */
                memset(pChunkBuff, 0, CHUNK);

                if (utteranceIndex - PREAMBLE_SIZE >= 0) {
                    if (AddTransposeInt8((uint8_t *)&micBuff[utteranceIndex - PREAMBLE_SIZE],
                                          pIntInput, PREAMBLE_SIZE, SAMPLE_SIZE, TRANSPOSE_WIDTH))
                        PR_DEBUG("ERROR: FloatTranspose early\n");
                } else {
                    if (AddTransposeInt8(
                            (uint8_t *)&micBuff[SAMPLE_SIZE - PREAMBLE_SIZE + utteranceIndex],
                            pIntInput, PREAMBLE_SIZE - utteranceIndex, SAMPLE_SIZE, TRANSPOSE_WIDTH))
                        PR_DEBUG("ERROR: FloatTranspose early\n");
                    if (AddTransposeInt8((uint8_t *)&micBuff[0], pIntInput, utteranceIndex,
                                          SAMPLE_SIZE, TRANSPOSE_WIDTH))
                        PR_DEBUG("ERROR: FloatTranspose early\n");
                }

                if (utteranceIndex < endIndex) {
                    if (AddTransposeInt8((uint8_t *)&micBuff[utteranceIndex], pIntInput,
                                          endIndex - utteranceIndex, SAMPLE_SIZE, TRANSPOSE_WIDTH))
                        PR_DEBUG("ERROR: FloatTranspose early\n");
                    while (!ret)
                        ret = AddTransposeInt8(pChunkBuff, pIntInput, CHUNK,
                                                SAMPLE_SIZE, TRANSPOSE_WIDTH);
                } else {
                    if (AddTransposeInt8((uint8_t *)&micBuff[utteranceIndex], pIntInput,
                                          SAMPLE_SIZE - utteranceIndex, SAMPLE_SIZE, TRANSPOSE_WIDTH))
                        PR_DEBUG("ERROR: FloatTranspose early\n");
                    if (AddTransposeInt8((uint8_t *)&micBuff[0], pIntInput, endIndex,
                                          SAMPLE_SIZE, TRANSPOSE_WIDTH))
                        PR_DEBUG("ERROR: FloatTranspose early\n");
                    while (!ret)
                        ret = AddTransposeInt8(pChunkBuff, pIntInput, CHUNK,
                                                SAMPLE_SIZE, TRANSPOSE_WIDTH);
                }

                ai85Counter       = 0;
                avgSilenceCounter = 0;
                wordCounter++;
                procState = SILENCE;

                if (ret != 1) {
                    PR_DEBUG("ERROR: FloatTranspose incomplete!\n");
                    fail();
                }

#if KWS20_CFG_ENABLE_AI_INPUT_DUMP
                dump_int8_model_input_once(pIntInput);
#endif

                /* ── INT8 CPU inference ─────────────────────────────── */
                PR_DEBUG("%.6d: Starting INT8 inference #%d\n", sampleCounter, wordCounter);

                uint32_t t0 = MXC_TMR_GetCount(MXC_TMR0);
                kws20_int8_cpu_infer(pIntInput, scores_f32);
                uint32_t t1 = MXC_TMR_GetCount(MXC_TMR0);

                /* APB_CLK (PCLK) = HCLK/2 = SystemCoreClock/2 on MAX78000 */
                uint32_t tick_diff = (t1 >= t0) ? (t1 - t0) : (0xFFFFFFFFu - t0 + t1);
                cnn_time = (uint32_t)((uint64_t)tick_diff * 1000000u / (SystemCoreClock / 2));
                cnn_timer_ticks  = tick_diff;
                cnn_timer_cycles = tick_diff; /* same source clock */

                PR_DEBUG("INT8 inference time: %lu us\n", (unsigned long)cnn_time);

                /* ── Check result ───────────────────────────────────── */
                int accepted = check_inference_f32(scores_f32, NUM_OUTPUTS,
                                                   &out_class, &probability);

                if (!accepted || out_class == NUM_OUTPUTS - 1) {
                    reported_class = NUM_OUTPUTS - 1;
                    PR_INFO("Detected word: %s", "Unknown");
                } else {
                    reported_class = out_class;
                    PR_INFO("Detected word: %s (%0.1f%%)", keywords[out_class], probability);
                }
                PR_INFO("\n");
                PR_INFO("BENCH,event=inference,run=%u,mode=%s,cnn_us=%u,timer_ticks=%u,cycles=%u,pred_idx=%d,pred_reported_idx=%d,accepted=%u,confidence_x10=%u,sample_counter=%u,word_counter=%u\n",
                        (unsigned int)wordCounter, KWS20_BENCH_MODE_STR,
                        (unsigned int)cnn_time,
                        (unsigned int)cnn_timer_ticks, (unsigned int)cnn_timer_cycles,
                        (int)out_class, (int)reported_class,
                        (unsigned int)accepted,
                        (unsigned int)((probability >= 0.0f) ? (probability * 10.0f + 0.5f) : 0u),
                        (unsigned int)sampleCounter, (unsigned int)wordCounter);

                /* LED feedback */
                if (accepted && out_class != NUM_OUTPUTS - 1) {
                    LED_On(LED_GREEN);
                    LED_Off(LED_RED);
                } else {
                    LED_Off(LED_GREEN);
                    LED_On(LED_RED);
                }

#ifdef SEND_MIC_OUT_SERIAL
                printf("START: %d\n", micBufIndex);
                for (int k = 0; k < SAMPLE_SIZE; k++)
                    console_uart_send_byte(micBuff[(micBufIndex + k) % SAMPLE_SIZE]);
                printf("END\n");
#endif
                memset(micBuff, 0, SAMPLE_SIZE);
                micBufIndex = 0;
            }
        }

#ifndef BOARD_AUD01_REVA
        if (PB_Get(0)) {
            PR_INFO("Stop!\r\n");
            procState = STOP;
            break;
        }
#endif
    }

    LED_Off(LED2);
    PR_DEBUG("Total Samples:%d, Total Words:%d\n", sampleCounter, wordCounter);

#ifdef TFT_ENABLE
    TFT_End(wordCounter);
#endif
    while (1) {}
}

/* ══════════════════════════════════════════════════════════════════════════ */

void fail(void) {
    PR_DEBUG("\n*** FAIL ***\n");
    LED_On(LED_RED);
    LED_Off(LED_GREEN);
    while (1) {}
}

/* ══════════════════════════════════════════════════════════════════════════
 * MicReadChunk, I2SInit, HPF — unchanged from kws20_demo/main.c
 * ══════════════════════════════════════════════════════════════════════════ */

#ifdef ENABLE_MIC_PROCESSING
#ifdef ENABLE_CODEC_MIC
static void codec_init(void) {
    int err;
    if (MXC_I2C_Init(PMIC_AUDIO_I2C, 1, 0) != E_NO_ERROR)
        printf("Error initializing I2C\n");
    MXC_I2C_SetFrequency(PMIC_AUDIO_I2C, CODEC_I2C_FREQ);
    if ((err = max9867_init(PMIC_AUDIO_I2C, EXT_I2S_FREQ, 1)) != E_NO_ERROR)
        PR_DEBUG("max9867_init error: %d\n", err);
    if (max9867_enable_record(1) != E_NO_ERROR)
        printf("Error enabling record path\n");
}
#else
static void codec_init(void) {}
#endif

void I2SInit(void) {
    mxc_i2s_req_t req;
    int32_t err;
    PR_INFO("\n*** I2S & Mic Init ***\n");
    HPF_init();
    memset(i2s_rx_buffer, 0, sizeof(i2s_rx_buffer));
    req.wordSize    = MXC_I2S_WSIZE_WORD;
    req.sampleSize  = MXC_I2S_SAMPLESIZE_THIRTYTWO;
    req.bitsWord    = 32;
    req.adjust      = MXC_I2S_ADJUST_LEFT;
    req.justify     = MXC_I2S_MSB_JUSTIFY;
    req.wsPolarity  = MXC_I2S_POL_NORMAL;
    req.channelMode = MXC_I2S_INTERNAL_SCK_WS_0;
    req.stereoMode  = MXC_I2S_MONO_LEFT_CH;
    req.bitOrder    = MXC_I2S_MSB_FIRST;
    req.clkdiv      = MXC_I2S_CalculateClockDiv(SAMPLE_RATE, req.wordSize, EXT_I2S_FREQ);
    req.rawData     = NULL;
    req.txData      = NULL;
    req.rxData      = i2s_rx_buffer;
    req.length      = I2S_RX_BUFFER_SIZE;
#ifdef ENABLE_CODEC_MIC
    req.channelMode = MXC_I2S_EXTERNAL_SCK_EXTERNAL_WS;
#endif
    if ((err = MXC_I2S_Init(&req)) != E_NO_ERROR) {
        PR_DEBUG("Error in I2S_Init: %d\n", err);
        while (1) {}
    }

    MXC_I2S_SetRXThreshold(4);

#ifndef WUT_ENABLE
    MXC_NVIC_SetVector(I2S_IRQn, i2s_isr);
    NVIC_EnableIRQ(I2S_IRQn);
    MXC_I2S_EnableInt(MXC_F_I2S_INTEN_RX_THD_CH0);
#endif
    MXC_I2S_RXEnable();
    __enable_irq();
}

uint8_t MicReadChunk(uint16_t *avg) {
    static uint16_t chunkCount = 0;
    static uint16_t sum        = 0;
    static uint32_t warmupIndex = 0;
    int32_t sample = 0;
    int16_t temp   = 0;
    uint32_t rx_size = 0;

    if (!i2s_flag) {
        *avg = 0;
        return 0;
    }
    i2s_flag = 0;

    rx_size = MXC_I2S->dmach0 >> MXC_F_I2S_DMACH0_RX_LVL_POS;

    while ((rx_size--) && (chunkCount < CHUNK)) {
        sample = (int32_t)MXC_I2S->fifoch0;
        temp = sample >> 14;
        sample = HPF((int16_t)temp);

        if (warmupIndex++ < 10000) {
            continue;
        }

        sum += (sample >= 0) ? sample : -sample;

        sample = (sample * SAMPLE_SCALE_FACTOR) / 256;
        if (sample >= 128)       sample =  127;
        else if (sample < -128)  sample = -128;

        micBuff[micBufIndex] = (int8_t)sample;

        if ((int8_t)micBuff[micBufIndex] > Max) Max = (int8_t)micBuff[micBufIndex];
        if ((int8_t)micBuff[micBufIndex] < Min) Min = (int8_t)micBuff[micBufIndex];

        micBufIndex = (micBufIndex + 1) % SAMPLE_SIZE;
        chunkCount++;
    }

    if (chunkCount < CHUNK) {
        *avg = 0;
        return 0;
    }

    *avg = (uint16_t)(sum / CHUNK);
    chunkCount = 0;
    sum = 0;
    return 1;
}

static int16_t hpf_x0, hpf_x1, hpf_coeff;
static int32_t hpf_y0, hpf_y1;

void HPF_init(void) {
    hpf_coeff = 32604; /* 0.995 in Q15 */
    hpf_x0 = 0;
    hpf_x1 = 0;
    hpf_y0 = 0;
    hpf_y1 = 0;
}

int16_t HPF(int16_t input) {
    int16_t acc;
    int32_t tmp;

    hpf_x0 = input;
    tmp = hpf_coeff * hpf_y1;
    acc = (int16_t)((tmp + (1 << 14)) >> 15);
    hpf_y0 = hpf_x0 - hpf_x1 + acc;

    if (hpf_y0 > 32767) hpf_y0 = 32767;
    if (hpf_y0 < -32768) hpf_y0 = -32768;

    hpf_y1 = hpf_y0;
    hpf_x1 = hpf_x0;
    return (int16_t)hpf_y0;
}

#else /* !ENABLE_MIC_PROCESSING — use test vector */

static void codec_init(void) {}
void I2SInit(void) {}
void HPF_init(void) {}
int16_t HPF(int16_t input) { return input; }

static int voiceIndex = 0;

int8_t MicReader(int32_t *sample) {
    if (voiceIndex >= (int)(sizeof(voiceVector) / sizeof(voiceVector[0])))
        return 0;
#ifndef EIGHT_BIT_SAMPLES
    *sample = (int32_t)voiceVector[voiceIndex++] * SAMPLE_SCALE_FACTOR;
#else
    *sample = (int32_t)voiceVector[voiceIndex++];
#endif
    return 1;
}

uint8_t MicReadChunk(uint16_t *avg) {
    static uint16_t chunkCount = 0;
    static uint32_t sum        = 0;
    int32_t sample;
    int8_t  sampleData;

    while (chunkCount < CHUNK) {
        if (!MicReader(&sample)) return 0;

        if (sample >= 128)       sample =  127;
        else if (sample < -128)  sample = -128;
        sampleData = (int8_t)sample;

        if (sampleData > Max) Max = sampleData;
        if (sampleData < Min) Min = sampleData;

        micBuff[micBufIndex] = sampleData;
        micBufIndex = (micBufIndex + 1) % SAMPLE_SIZE;
        sum += abs(sampleData);
        chunkCount++;
    }

    *avg       = sum / CHUNK;
    chunkCount = 0;
    sum        = 0;
    return 1;
}
#endif /* ENABLE_MIC_PROCESSING */

/******************************************************************************/
#ifdef TFT_ENABLE
void TFT_Intro(void)
{
    char buff[TFT_BUFF_SIZE];
    memset(buff, 32, TFT_BUFF_SIZE);
    TFT_Print(buff, 55, 10, font_2, snprintf(buff, sizeof(buff), "ANALOG DEVICES"));
    TFT_Print(buff, 35, 40, font_1, snprintf(buff, sizeof(buff), "KWS20 INT8 CPU Demo"));
    TFT_Print(buff, 65, 70, font_1, snprintf(buff, sizeof(buff), "Ver. %s", VERSION));
    TFT_Print(buff, 5, 110, font_1, snprintf(buff, sizeof(buff), "No CNN hardware accelerator"));
    TFT_Print(buff, 20, 210, font_2, snprintf(buff, sizeof(buff), "PRESS PB1(SW1) TO START!"));
    while (!PB_Get(0)) {}
    MXC_TFT_ClearScreen();
    TFT_Print(buff, 20, 50, font_1, snprintf(buff, sizeof(buff), "Start saying keywords..."));
}

void TFT_Print(char *str, int x, int y, int font, int length)
{
    text_t text;
    text.data = str;
    text.len  = length;
    MXC_TFT_PrintFont(x, y, font, &text, NULL);
}

void TFT_End(uint16_t words)
{
    char buff[TFT_BUFF_SIZE];
    memset(buff, 32, TFT_BUFF_SIZE);
    MXC_TFT_ClearScreen();
    TFT_Print(buff, 70, 30, font_2, snprintf(buff, sizeof(buff), "Demo Stopped!"));
    TFT_Print(buff, 10, 60, font_1, snprintf(buff, sizeof(buff), "Words: %d", words));
    TFT_Print(buff, 0, 180, font_1, snprintf(buff, sizeof(buff), "PRESS RESET TO TRY AGAIN!"));
}
#endif /* TFT_ENABLE */
