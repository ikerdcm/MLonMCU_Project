#include "kws20_live.h"
#include "kws20_mode_config.h"
#include "ds_cnn_frontend.h"
#include "cnn_inference.h"
#include "cnn.h"

#include <stdio.h>
#include <string.h>
#include <stdint.h>

#include "mxc.h"
#include "tmr.h"
#include "i2s.h"
#include "i2s_regs.h"
#include "board.h"
#include "dma.h"

#define LIVE_SAMPLE_RATE        16000u
#define LIVE_WINDOW_SAMPLES     16000u
#define LIVE_CHUNK              128u
#define LIVE_PREAMBLE_SAMPLES   (30u * LIVE_CHUNK)
#define LIVE_POST_MAX_SAMPLES   (LIVE_WINDOW_SAMPLES - LIVE_PREAMBLE_SAMPLES)
#define LIVE_THRESHOLD_HIGH     700u
#define LIVE_THRESHOLD_LOW      200u
#define LIVE_SILENCE_THRESH     20u
#define LIVE_WARMUP_SAMPLES     10000u
#define LIVE_I2S_BUF_SIZE       64u
#define LIVE_EXT_I2S_FREQ       12288000u

#define LIVE_OUT_CLASSES        12u
#define LIVE_INPUT_ELEMS        490u

#if KWS20_CFG_ENABLE_MEASURE && KWS20_CFG_MEASURE_LIVE
#define LIVE_BENCH_ENABLED 1
#else
#define LIVE_BENCH_ENABLED 0
#endif

static volatile int32_t  i2s_rx_buf[LIVE_I2S_BUF_SIZE];
static volatile uint8_t  i2s_flag = 0;

static int16_t  mic_ring[LIVE_WINDOW_SAMPLES];
static int16_t  audio_win[LIVE_WINDOW_SAMPLES];
static float    mfcc_in[LIVE_INPUT_ELEMS];
static float    scores[LIVE_OUT_CLASSES];

static uint32_t ring_head   = 0;
static uint32_t ring_filled = 0;

static const char *labels[LIVE_OUT_CLASSES] = {
    "down", "go", "left", "no", "off", "on",
    "right", "stop", "up", "yes", "silence", "unknown"
};

/* HPF */
static int16_t hpf_x1 = 0;
static int32_t hpf_y1 = 0;

static void hpf_init(void) { hpf_x1 = 0; hpf_y1 = 0; }

static int16_t hpf(int16_t input)
{
    const int32_t coeff = 32604;
    int32_t acc = (coeff * hpf_y1 + (1 << 14)) >> 15;
    int32_t y   = (int32_t)input - (int32_t)hpf_x1 + acc;
    if (y >  32767) y =  32767;
    if (y < -32768) y = -32768;
    hpf_x1 = input;
    hpf_y1 = y;
    return (int16_t)y;
}

void I2S_IRQHandler(void)
{
    i2s_flag = 1;
    MXC_I2S_ClearFlags(MXC_F_I2S_INTFL_RX_THD_CH0);
}

static void i2s_init(void)
{
    mxc_i2s_req_t req;
    hpf_init();
    memset((void *)i2s_rx_buf, 0, sizeof(i2s_rx_buf));

    req.wordSize    = MXC_I2S_WSIZE_WORD;
    req.sampleSize  = MXC_I2S_SAMPLESIZE_THIRTYTWO;
    req.bitsWord    = 32;
    req.adjust      = MXC_I2S_ADJUST_LEFT;
    req.justify     = MXC_I2S_MSB_JUSTIFY;
    req.wsPolarity  = MXC_I2S_POL_NORMAL;
    req.channelMode = MXC_I2S_INTERNAL_SCK_WS_0;
    req.stereoMode  = MXC_I2S_MONO_LEFT_CH;
    req.bitOrder    = MXC_I2S_MSB_FIRST;
    req.clkdiv      = MXC_I2S_CalculateClockDiv(LIVE_SAMPLE_RATE, req.wordSize, LIVE_EXT_I2S_FREQ);
    req.rawData     = NULL;
    req.txData      = NULL;
    req.rxData      = (void *)i2s_rx_buf;
    req.length      = LIVE_I2S_BUF_SIZE;

    if (MXC_I2S_Init(&req) != E_NO_ERROR) {
        printf("I2S_Init failed\r\n");
        while (1) {}
    }

    MXC_I2S_SetRXThreshold(4);
    MXC_NVIC_SetVector(I2S_IRQn, I2S_IRQHandler);
    NVIC_EnableIRQ(I2S_IRQn);
    MXC_I2S_EnableInt(MXC_F_I2S_INTEN_RX_THD_CH0);
    MXC_I2S_RXEnable();
    __enable_irq();
}

static uint8_t mic_read_chunk(uint16_t *avg_out)
{
    static uint32_t chunk_count = 0, chunk_sum = 0;
    static uint32_t warmup_done = 0, warmup_idx = 0;
    uint32_t rx_level;

    if (!i2s_flag) { *avg_out = 0; return 0; }
    i2s_flag = 0;
    rx_level = MXC_I2S->dmach0 >> MXC_F_I2S_DMACH0_RX_LVL_POS;

    while (rx_level > 0 && chunk_count < LIVE_CHUNK) {
        int32_t raw  = (int32_t)MXC_I2S->fifoch0;
        int16_t s16  = (int16_t)(raw >> 14);
        int16_t filt = hpf(s16);
        rx_level--;

        if (!warmup_done) {
            if (++warmup_idx >= LIVE_WARMUP_SAMPLES) {
                warmup_done = 1;
                printf("Mic warmup done.\r\n");
            }
            continue;
        }

        mic_ring[ring_head] = filt;
        ring_head = (ring_head + 1u) % LIVE_WINDOW_SAMPLES;
        if (ring_filled < LIVE_WINDOW_SAMPLES) ring_filled++;

        uint16_t abs_s = (filt >= 0) ? (uint16_t)filt : (uint16_t)(-filt);
        chunk_sum += abs_s;
        chunk_count++;
    }

    if (chunk_count < LIVE_CHUNK) { *avg_out = 0; return 0; }

    *avg_out = (uint16_t)(chunk_sum / LIVE_CHUNK);
    chunk_count = 0;
    chunk_sum   = 0;
    return 1;
}

static void ring_extract(uint32_t offset_before_head, uint32_t n_samples)
{
    uint32_t start = (ring_head + LIVE_WINDOW_SAMPLES - offset_before_head) % LIVE_WINDOW_SAMPLES;
    for (uint32_t i = 0; i < n_samples && i < LIVE_WINDOW_SAMPLES; i++)
        audio_win[i] = mic_ring[(start + i) % LIVE_WINDOW_SAMPLES];
    for (uint32_t i = n_samples; i < LIVE_WINDOW_SAMPLES; i++)
        audio_win[i] = 0;
}

static void run_inference(uint32_t run_idx, uint32_t post_samples)
{
    int pred;

    memset(mfcc_in, 0, sizeof(mfcc_in));
    ds_cnn_frontend_compute(audio_win, LIVE_WINDOW_SAMPLES, mfcc_in, LIVE_INPUT_ELEMS);

    pred = cnn_infer(mfcc_in, scores);
    uint32_t time_us = cnn_time;

#if LIVE_BENCH_ENABLED
    {
        uint32_t cycles = (uint32_t)(((uint64_t)time_us * (uint64_t)SystemCoreClock) / 1000000ULL);
        printf("BENCH,event=inference,run=%lu,mode=live,version=v1_cnn_int8"
               ",cnn_us=%lu,cycles=%lu,pred_idx=%d,audio_window_ms=%lu\r\n",
               (unsigned long)run_idx,
               (unsigned long)time_us,
               (unsigned long)cycles,
               pred,
               (unsigned long)(((uint64_t)post_samples * 1000ULL) / LIVE_SAMPLE_RATE));
    }
#endif

    printf("DS-CNN v1: predicted=%s  cnn_us=%lu\r\n",
           (pred >= 0 && pred < (int)LIVE_OUT_CLASSES) ? labels[pred] : "?",
           (unsigned long)time_us);
}

void kws20_live_run_once(void)
{
    uint32_t run_idx = 0, ai_counter = 0, silence_run = 0;
    uint8_t  collecting = 0;

    if (Microphone_Power(POWER_ON) != E_NO_ERROR) {
        printf("Microphone_Power failed\r\n");
        return;
    }

    printf("\r\n==============================\r\n");
    printf("DS-CNN v1 LIVE MODE\r\n");
    printf("CNN accelerator, INT8\r\n");
    printf("==============================\r\n");
    printf("Speak a keyword (down/go/left/no/off/on/right/stop/up/yes)\r\n");

    i2s_init();

#if LIVE_BENCH_ENABLED
    printf("BENCH,event=model_info,mode=live,version=v1_cnn_int8"
           ",hclk_hz=%lu,input_elems=%u,output_elems=%u"
           ",sample_rate_hz=%u,full_window_samples=%u\r\n",
           (unsigned long)SystemCoreClock,
           (unsigned int)LIVE_INPUT_ELEMS, (unsigned int)LIVE_OUT_CLASSES,
           (unsigned int)LIVE_SAMPLE_RATE, (unsigned int)LIVE_WINDOW_SAMPLES);
#endif

    for (;;) {
        uint16_t avg = 0;
        if (!mic_read_chunk(&avg)) continue;

        /* Mic level for the dashboard plot — chunk rate is 125 Hz, emit ~25 Hz. */
        static uint32_t level_div = 0;
        if ((++level_div % 5u) == 0u) {
            printf("BENCH,event=level,rms=%u\r\n", avg);
        }

        if (ring_filled < LIVE_PREAMBLE_SAMPLES) continue;

        if (!collecting) {
            if (avg >= LIVE_THRESHOLD_HIGH) {
                collecting  = 1;
                ai_counter  = LIVE_PREAMBLE_SAMPLES;
                silence_run = 0;
                printf("Trigger (avg=%u)\r\n", avg);
            }
        } else {
            ai_counter += LIVE_CHUNK;
            if (avg < LIVE_THRESHOLD_LOW && ai_counter >= LIVE_WINDOW_SAMPLES / 3u)
                silence_run++;
            else
                silence_run = 0;

            if ((silence_run >= LIVE_SILENCE_THRESH) || (ai_counter >= LIVE_POST_MAX_SAMPLES)) {
                uint32_t total  = (ai_counter < LIVE_WINDOW_SAMPLES) ? ai_counter : LIVE_WINDOW_SAMPLES;
                ring_extract(total, LIVE_WINDOW_SAMPLES);
                run_inference(run_idx++, ai_counter - LIVE_PREAMBLE_SAMPLES);
                collecting = 0; ai_counter = 0; silence_run = 0;
                printf("Waiting for next keyword...\r\n");
            }
        }
    }
}
