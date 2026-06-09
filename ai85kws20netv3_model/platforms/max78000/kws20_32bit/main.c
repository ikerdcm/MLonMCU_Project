/*
 * KWS20 — Float32 CPU-only inference on MAX78000 Cortex-M4F.
 *
 * Based on kws20_demo/main.c (v3.2.3) with CNN hardware accelerator replaced
 * by kws20_32bit_inference.c (software float32 forward pass).
 *
 * Changes vs. kws20_demo/main.c:
 *   - #include "cnn.h"     → #include "kws20_32bit_inference.h"
 *   - ml_data / ml_softmax → float scores_f32[]
 *   - pAI85Buffer (CNN mem) → pFloatInput (C,T float)
 *   - AddTranspose          → AddTransposeFloat (same logic, float output)
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

/* Float32 inference engine (replaces cnn.h + cnn.c + weights.h) */
#include "kws20_32bit_inference.h"

#define VERSION "1.0.0-f32" /* float32 CPU-only variant */

/* **** Definitions **** */
#define CLOCK_SOURCE 0
#define SLEEP_MODE   0
#define WUT_ENABLE
#define WUT_USEC     380
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

#if KWS20_CFG_ENABLE_MEASURE && KWS20_CFG_MEASURE_LIVE
#define KWS20_MEASURE_LIVE_ACTIVE 1
#else
#define KWS20_MEASURE_LIVE_ACTIVE 0
#endif

#if !(KWS20_MEASURE_LIVE_ACTIVE && KWS20_CFG_MINIMAL_OUTPUT)
#define ENABLE_PRINT_ENVELOPE
#endif
#define ENABLE_SILENCE_DETECTION
#undef EIGHT_BIT_SAMPLES
#if !KWS20_CFG_FORCE_OFFLINE
#define ENABLE_MIC_PROCESSING
#endif
#if !(KWS20_MEASURE_LIVE_ACTIVE && KWS20_CFG_MINIMAL_OUTPUT)
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
#define NUM_OUTPUTS           KWS32_NUM_CLASSES  /* 21 */
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

#ifdef ENABLE_MIC_PROCESSING
#define KWS20_BENCH_MODE_STR "live"
#else
#define KWS20_BENCH_MODE_STR "offline"
#endif

/* MACs for the float32 model (unpruned v3, from network report) */
#define KWS32_MACC  8345344u
#define KWS32_OPS   8402528u

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
 * Float input buffer for the model: shape (C=128, T=128), row-major.
 * Filled by AddTransposeFloat() in place of AddTranspose() + pAI85Buffer.
 */
static float pFloatInput[SAMPLE_SIZE]; /* 128*128 = 16384 floats = 64 KB */

/* Float logit scores from the model (21 classes). */
static float scores_f32[NUM_OUTPUTS];

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
uint8_t AddTransposeFloat(uint8_t *pIn, float *pOut, uint16_t inSize, uint16_t outSize, uint16_t width);
uint8_t check_inference_f32(const float *scores, int n_classes, int16_t *out_class, double *out_prob);
void    I2SInit(void);
static  void codec_init(void);
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
 * Float32 replacement for AddTranspose.
 *
 * Converts micBuff samples (uint8/int8, layout: time-major [T][C]) into
 * float model input (layout: channel-major [C][T]) while normalizing to
 * the range [-1, 1) via division by 128.
 *
 * pIn    : source int8 samples, SAMPLE_SCALE_FACTOR-scaled audio
 * pOut   : float[SAMPLE_SIZE] in (C, T) = (128, 128) row-major
 * inSize : number of samples to consume this call
 * outSize: total samples when complete (== SAMPLE_SIZE)
 * width  : TRANSPOSE_WIDTH == 128 (== num channels)
 *
 * Returns 1 when outSize samples have been accumulated (buffer full).
 * ══════════════════════════════════════════════════════════════════════════ */
uint8_t AddTransposeFloat(uint8_t *pIn, float *pOut, uint16_t inSize,
                          uint16_t outSize, uint16_t width)
{
    static uint16_t row = 0, col = 0, total = 0;
    int T = outSize / width; /* number of time steps = 128 */

    for (int i = 0; i < inSize; i++) {
        /* row = time index, col = channel index */
        pOut[(int)col * T + (int)row] = (float)(int8_t)pIn[i] / 128.0f;

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

/* ══════════════════════════════════════════════════════════════════════════
 * Float32 inference check (replaces check_inference + softmax_q17p14_q15).
 *
 * Computes softmax over raw logits, returns argmax class and probability.
 * Returns 1 if probability >= INFERENCE_THRESHOLD%, 0 otherwise.
 * ══════════════════════════════════════════════════════════════════════════ */
uint8_t check_inference_f32(const float *scores, int n_classes,
                             int16_t *out_class, double *out_prob)
{
    /* argmax */
    int   best_idx = 0;
    float best_val = scores[0];
    for (int i = 1; i < n_classes; i++) {
        if (scores[i] > best_val) { best_val = scores[i]; best_idx = i; }
    }
    *out_class = (int16_t)best_idx;

    /* softmax probability of the best class */
    float sum = 0.0f;
    for (int i = 0; i < n_classes; i++)
        sum += expf(scores[i] - best_val);
    *out_prob = (double)(1.0f / sum) * 100.0f;

    return (*out_prob >= INFERENCE_THRESHOLD) ? 1 : 0;
}

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

    PR_INFO("\n\nANALOG DEVICES\nKeyword Spotting Demo (Float32 CPU)\nVer. %s\n", VERSION);
    PR_INFO("*** Weights must be generated from extract_weights.py ***\n");
    PR_INFO("\n***** Init *****\n");
    memset(pFloatInput, 0, sizeof(pFloatInput));

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

    PR_INFO("BENCH,event=model_info,mode=%s,input_elems=%u,output_elems=%u,macc=%u,ops=%u,variant=float32_cpu\n",
            KWS20_BENCH_MODE_STR,
            (unsigned int)SAMPLE_SIZE, (unsigned int)NUM_OUTPUTS,
            (unsigned int)KWS32_MACC, (unsigned int)KWS32_OPS);
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

    /* ── Main loop ──────────────────────────────────────────────────────── */
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

                /* ── Assemble float input buffer (C,T format) ── */
                memset(pChunkBuff, 0, CHUNK);

                if (utteranceIndex - PREAMBLE_SIZE >= 0) {
                    if (AddTransposeFloat((uint8_t *)&micBuff[utteranceIndex - PREAMBLE_SIZE],
                                          pFloatInput, PREAMBLE_SIZE, SAMPLE_SIZE, TRANSPOSE_WIDTH))
                        PR_DEBUG("ERROR: FloatTranspose early\n");
                } else {
                    if (AddTransposeFloat(
                            (uint8_t *)&micBuff[SAMPLE_SIZE - PREAMBLE_SIZE + utteranceIndex],
                            pFloatInput, PREAMBLE_SIZE - utteranceIndex, SAMPLE_SIZE, TRANSPOSE_WIDTH))
                        PR_DEBUG("ERROR: FloatTranspose early\n");
                    if (AddTransposeFloat((uint8_t *)&micBuff[0], pFloatInput, utteranceIndex,
                                          SAMPLE_SIZE, TRANSPOSE_WIDTH))
                        PR_DEBUG("ERROR: FloatTranspose early\n");
                }

                if (utteranceIndex < endIndex) {
                    if (AddTransposeFloat((uint8_t *)&micBuff[utteranceIndex], pFloatInput,
                                          endIndex - utteranceIndex, SAMPLE_SIZE, TRANSPOSE_WIDTH))
                        PR_DEBUG("ERROR: FloatTranspose early\n");
                    while (!ret)
                        ret = AddTransposeFloat(pChunkBuff, pFloatInput, CHUNK,
                                                SAMPLE_SIZE, TRANSPOSE_WIDTH);
                } else {
                    if (AddTransposeFloat((uint8_t *)&micBuff[utteranceIndex], pFloatInput,
                                          SAMPLE_SIZE - utteranceIndex, SAMPLE_SIZE, TRANSPOSE_WIDTH))
                        PR_DEBUG("ERROR: FloatTranspose early\n");
                    if (AddTransposeFloat((uint8_t *)&micBuff[0], pFloatInput, endIndex,
                                          SAMPLE_SIZE, TRANSPOSE_WIDTH))
                        PR_DEBUG("ERROR: FloatTranspose early\n");
                    while (!ret)
                        ret = AddTransposeFloat(pChunkBuff, pFloatInput, CHUNK,
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

                /* ── Float32 CPU inference ──────────────────────────── */
                PR_DEBUG("%.6d: Starting float32 inference #%d\n", sampleCounter, wordCounter);

                uint32_t t0 = MXC_TMR_GetCount(MXC_TMR0);
                kws20_32bit_infer(pFloatInput, scores_f32);
                uint32_t t1 = MXC_TMR_GetCount(MXC_TMR0);

                /* Convert timer counts to microseconds */
                uint32_t tick_diff = (t1 >= t0) ? (t1 - t0) : (0xFFFFFFFFu - t0 + t1);
                cnn_time = (uint32_t)((uint64_t)tick_diff * 1000000u / SystemCoreClock);
                cnn_timer_ticks  = tick_diff;
                cnn_timer_cycles = tick_diff; /* same source clock */

                PR_DEBUG("Float32 inference time: %lu us\n", (unsigned long)cnn_time);

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
    MXC_NVIC_SetVector(I2S_IRQn, i2s_isr);
    NVIC_EnableIRQ(I2S_IRQn);
    MXC_I2S_EnableInt(MXC_F_I2S_INTEN_RX_THD_CH0);
    MXC_I2S_RXEnable();
    __enable_irq();
}

uint8_t MicReadChunk(uint16_t *avg) {
    static uint16_t chunkCount = 0;
    static uint16_t sum        = 0;
    static int      i          = 0;
    int32_t sample   = 0;
    int8_t  sampleData;
    uint16_t totalCount = 0;

    /* If there is not a complete sample, return 0 */
    if (!i2s_flag) return 0;
    i2s_flag = 0;

    /* Read number of samples in I2S RX FIFO */
    totalCount = (MXC_I2S->dmach0 >> MXC_F_I2S_DMACH0_RX_LVL_POS) & 0xFF;

    for (i = 0; i < totalCount; i++) {
        /* Read sample from FIFO */
        sample = i2s_rx_buffer[i];

        /* Clear I2S RX FIFO */
        i2s_rx_buffer[i] = 0;

        /* Scale and clip sample */
        sample = (sample >> 7) * SAMPLE_SCALE_FACTOR;
        if (sample >= 128)       sample =  127;
        else if (sample < -128)  sample = -128;

        /* High-pass filter */
        sample = HPF(sample);

        sampleData = sample;

        /* Track min/max */
        if (sampleData > Max) Max = sampleData;
        if (sampleData < Min) Min = sampleData;

        /* Copy to circular buffer */
        micBuff[micBufIndex] = sampleData;
        micBufIndex = (micBufIndex + 1) % SAMPLE_SIZE;

        /* Accumulate absolute value for envelope */
        sum += abs(sampleData);
        chunkCount++;
    }

    if (chunkCount >= CHUNK) {
        *avg       = sum / chunkCount;
        chunkCount = 0;
        sum        = 0;
        return 1;
    }
    return 0;
}

/* High-pass filter (unchanged from kws20_demo) */
#define COEF_A1   (-1.9334f)
#define COEF_A2   ( 0.9350f)
#define COEF_B0   ( 0.9672f)
#define COEF_B1   (-1.9344f)
#define COEF_B2   ( 0.9672f)
static float hpf_z1 = 0.0f, hpf_z2 = 0.0f;

void HPF_init(void) { hpf_z1 = 0.0f; hpf_z2 = 0.0f; }

int16_t HPF(int16_t input) {
    float in  = (float)input;
    float out = COEF_B0 * in + hpf_z1;
    hpf_z1    = COEF_B1 * in - COEF_A1 * out + hpf_z2;
    hpf_z2    = COEF_B2 * in - COEF_A2 * out;
    if (out >  127.0f) out =  127.0f;
    if (out < -128.0f) out = -128.0f;
    return (int16_t)out;
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
