#include "kws20_live.h"
#include "kws20_mode_config.h"
#include "ds_cnn_frontend.h"
#include "ds_cnn_inference.h"

#include <stdio.h>
#include <string.h>
#include <stdint.h>

#include "mxc.h"
#include "tmr.h"
#include "i2s.h"
#include "i2s_regs.h"
#include "board.h"
#include "dma.h"

/* -------------------------------------------------------------------------
 * Constants
 * ---------------------------------------------------------------------- */
#define LIVE_SAMPLE_RATE        16000u
#define LIVE_WINDOW_SAMPLES     16000u   /* 1 s audio window fed to MFCC */
#define LIVE_CHUNK              128u     /* samples processed per iteration */
#define LIVE_PREAMBLE_SAMPLES   (30u * LIVE_CHUNK)   /* 3840 samples pre-trigger */
#define LIVE_POST_MAX_SAMPLES   (LIVE_WINDOW_SAMPLES - LIVE_PREAMBLE_SAMPLES)
/* UNIFIED across all 3 boards: level is reported normalized to the noise floor
   (×100, baseline ~100 everywhere) and the threshold is the same line at
   K_HIGH×100. nf is floored at the startup-ambient seed so it never collapses.
   Adaptive nf VAD (ported from the 8-bit/v1 firmware) replaces the old fixed
   350/100 thresholds so v0 and v1 detect identically. */
#define LIVE_VAD_K_HIGH         4.0f   /* trigger when avg >= nf * K_HIGH */
#define LIVE_VAD_K_LOW          2.0f   /* end when avg <  nf * K_LOW */
#define LIVE_VAD_WARMUP_CHUNKS  60u    /* ~0.5 s @ 125 Hz chunk rate to seed nf */
#define LIVE_SILENCE_THRESH     20u      /* consecutive quiet chunks → end */
#define LIVE_WARMUP_SAMPLES     10000u   /* discard initial mic charge samples */
#define LIVE_I2S_BUF_SIZE       64u
#define LIVE_EXT_I2S_FREQ       12288000u

#define LIVE_OUT_CLASSES        12u
#define LIVE_INPUT_ELEMS        490u

#if KWS20_CFG_ENABLE_MEASURE && KWS20_CFG_MEASURE_LIVE
#define LIVE_BENCH_ENABLED 1
#else
#define LIVE_BENCH_ENABLED 0
#endif

/* -------------------------------------------------------------------------
 * Buffers
 * ---------------------------------------------------------------------- */
static volatile int32_t  i2s_rx_buf[LIVE_I2S_BUF_SIZE];
static volatile uint8_t  i2s_flag = 0;

static int16_t  mic_ring[LIVE_WINDOW_SAMPLES]; /* circular sample ring     */
static int16_t  audio_win[LIVE_WINDOW_SAMPLES]; /* extracted utterance      */
static float    mfcc_in[LIVE_INPUT_ELEMS];
static float    scores[LIVE_OUT_CLASSES];

static uint32_t ring_head   = 0;   /* next write position in mic_ring */
static uint32_t ring_filled = 0;

/* -------------------------------------------------------------------------
 * Labels
 * ---------------------------------------------------------------------- */
static const char *labels[LIVE_OUT_CLASSES] = {
    "down", "go", "left", "no", "off", "on",
    "right", "stop", "up", "yes", "silence", "unknown"
};

/* -------------------------------------------------------------------------
 * HPF (1st-order IIR, ~100 Hz cutoff)
 * ---------------------------------------------------------------------- */
static int16_t hpf_x1 = 0;
static int32_t hpf_y1 = 0;

static void hpf_init(void)
{
    hpf_x1 = 0;
    hpf_y1 = 0;
}

static int16_t hpf(int16_t input)
{
    const int32_t coeff = 32604; /* 0.995 * 2^15 */
    int32_t acc = (coeff * hpf_y1 + (1 << 14)) >> 15;
    int32_t y   = (int32_t)input - (int32_t)hpf_x1 + acc;

    if (y >  32767) y =  32767;
    if (y < -32768) y = -32768;

    hpf_x1 = input;
    hpf_y1 = y;
    return (int16_t)y;
}

/* -------------------------------------------------------------------------
 * I2S ISR
 * ---------------------------------------------------------------------- */
void I2S_IRQHandler(void)
{
    i2s_flag = 1;
    MXC_I2S_ClearFlags(MXC_F_I2S_INTFL_RX_THD_CH0);
}

/* -------------------------------------------------------------------------
 * I2S + Mic init
 * ---------------------------------------------------------------------- */
static void i2s_init(void)
{
    mxc_i2s_req_t req;
    int err;

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

    err = MXC_I2S_Init(&req);
    if (err != E_NO_ERROR) {
        printf("I2S_Init failed: %d\r\n", err);
        while (1) {}
    }

    MXC_I2S_SetRXThreshold(4);
    MXC_NVIC_SetVector(I2S_IRQn, I2S_IRQHandler);
    NVIC_EnableIRQ(I2S_IRQn);
    MXC_I2S_EnableInt(MXC_F_I2S_INTEN_RX_THD_CH0);
    MXC_I2S_RXEnable();
    __enable_irq();
}

/* -------------------------------------------------------------------------
 * Read one CHUNK of samples from I2S FIFO.
 * Returns 1 when a full chunk is ready, 0 otherwise.
 * avg_out = mean absolute value of the chunk (energy proxy).
 * Also appends samples into mic_ring.
 * ---------------------------------------------------------------------- */
static uint8_t mic_read_chunk(uint16_t *avg_out)
{
    static uint32_t chunk_count = 0;
    static uint32_t chunk_sum   = 0;
    static uint32_t warmup_done = 0;
    static uint32_t warmup_idx  = 0;

    uint32_t rx_level;

    if (!i2s_flag) {
        *avg_out = 0;
        return 0;
    }
    i2s_flag = 0;

    rx_level = MXC_I2S->dmach0 >> MXC_F_I2S_DMACH0_RX_LVL_POS;

    while (rx_level > 0 && chunk_count < LIVE_CHUNK) {
        int32_t raw  = (int32_t)MXC_I2S->fifoch0;
        int16_t s16  = (int16_t)(raw >> 14);   /* top 18 MSB → 16-bit */
        int16_t filt = hpf(s16);
        rx_level--;

        /* discard initial samples while mic cap charges */
        if (!warmup_done) {
            if (++warmup_idx >= LIVE_WARMUP_SAMPLES) {
                warmup_done = 1;
                printf("Mic warmup done.\r\n");
            }
            continue;
        }

        /* store in ring buffer */
        mic_ring[ring_head] = filt;
        ring_head = (ring_head + 1u) % LIVE_WINDOW_SAMPLES;
        if (ring_filled < LIVE_WINDOW_SAMPLES)
            ring_filled++;

        uint16_t abs_s = (filt >= 0) ? (uint16_t)filt : (uint16_t)(-filt);
        chunk_sum += abs_s;
        chunk_count++;
    }

    if (chunk_count < LIVE_CHUNK) {
        *avg_out = 0;
        return 0;
    }

    *avg_out    = (uint16_t)(chunk_sum / LIVE_CHUNK);
    chunk_count = 0;
    chunk_sum   = 0;
    return 1;
}

/* -------------------------------------------------------------------------
 * Copy 'n_samples' from ring buffer starting 'offset' samples before
 * ring_head into audio_win[0..n_samples-1].
 * ---------------------------------------------------------------------- */
static void ring_extract(uint32_t offset_before_head, uint32_t n_samples)
{
    uint32_t start = (ring_head + LIVE_WINDOW_SAMPLES - offset_before_head) % LIVE_WINDOW_SAMPLES;

    for (uint32_t i = 0; i < n_samples && i < LIVE_WINDOW_SAMPLES; i++) {
        audio_win[i] = mic_ring[(start + i) % LIVE_WINDOW_SAMPLES];
    }
    for (uint32_t i = n_samples; i < LIVE_WINDOW_SAMPLES; i++) {
        audio_win[i] = 0;
    }
}

/* -------------------------------------------------------------------------
 * Run inference on audio_win, print result.
 * ---------------------------------------------------------------------- */
static void run_inference(uint32_t run_idx, uint32_t post_samples)
{
    uint32_t time_us;
    int pred;

    /* MFCC */
    memset(mfcc_in, 0, sizeof(mfcc_in));
    ds_cnn_frontend_compute(audio_win, LIVE_WINDOW_SAMPLES, mfcc_in, LIVE_INPUT_ELEMS);

    /* Inference with hardware timer */
    MXC_TMR_SW_Start(MXC_TMR0);
    pred    = ds_cnn_infer(mfcc_in, scores);
    time_us = MXC_TMR_SW_Stop(MXC_TMR0);

#if LIVE_BENCH_ENABLED
    {
        uint32_t cycles = (uint32_t)(((uint64_t)time_us * (uint64_t)SystemCoreClock) / 1000000ULL);
        printf("BENCH,event=inference,run=%lu,mode=live,frontend=mfcc_tf_49x10"
               ",cnn_us=%lu,cycles=%lu,pred_idx=%d,audio_window_ms=%lu\r\n",
               (unsigned long)run_idx,
               (unsigned long)time_us,
               (unsigned long)cycles,
               pred,
               (unsigned long)(((uint64_t)post_samples * 1000ULL) / LIVE_SAMPLE_RATE));

        /* Decision view: per-class scores 0..100 for the dashboard bar chart
           (ds_cnn_infer applies softmax → scores are 0..1). */
        char sb[256];
        int n = snprintf(sb, sizeof(sb), "BENCH,event=scores,run=%lu,s=", (unsigned long)run_idx);
        for (uint32_t i = 0; i < LIVE_OUT_CLASSES; ++i) {
            int s = (int)(scores[i] * 100.0f + 0.5f);
            if (s < 0) s = 0; else if (s > 100) s = 100;
            n += snprintf(sb + n, sizeof(sb) - n, "%s%d", i ? ";" : "", s);
        }
        printf("%s\r\n", sb);
    }
#endif

    /* Human-readable result */
    printf("DS-CNN live: predicted=%s  cnn_us=%lu\r\n",
           (pred >= 0 && pred < (int)LIVE_OUT_CLASSES) ? labels[pred] : "?",
           (unsigned long)time_us);
}

/* -------------------------------------------------------------------------
 * kws20_live_run_once — main loop (never returns)
 * ---------------------------------------------------------------------- */
void kws20_live_run_once(void)
{
    uint32_t sample_count   = 0;
    uint32_t run_idx        = 0;
    uint32_t ai_counter     = 0;    /* samples collected after trigger       */
    uint32_t silence_run    = 0;    /* consecutive quiet chunks after trigger */
    uint8_t  collecting     = 0;
    float    nf = 0.0f, nf_seed = 0.0f;   /* noise floor + startup-ambient floor */
    uint32_t vad_warmup = 0;

    int mic_err = Microphone_Power(POWER_ON);
    if (mic_err != E_NO_ERROR) {
        printf("Microphone_Power failed: %d\r\n", mic_err);
        return;
    }

    printf("\r\n==============================\r\n");
    printf("DS-CNN LIVE MODE\r\n");
    printf("==============================\r\n");
    printf("Speak a keyword (down/go/left/no/off/on/right/stop/up/yes)\r\n");

    i2s_init();

#if LIVE_BENCH_ENABLED
    printf("BENCH,event=model_info,mode=live,hclk_hz=%lu"
           ",input_elems=%u,output_elems=%u,sample_rate_hz=%u,full_window_samples=%u\r\n",
           (unsigned long)SystemCoreClock,
           (unsigned int)LIVE_INPUT_ELEMS,
           (unsigned int)LIVE_OUT_CLASSES,
           (unsigned int)LIVE_SAMPLE_RATE,
           (unsigned int)LIVE_WINDOW_SAMPLES);
    printf("BENCH,event=acquisition,mode=live,sample_rate_hz=%u,sample_count=%u"
           ",audio_window_ms=%u,chunk=%u,preamble=%u\r\n",
           (unsigned int)LIVE_SAMPLE_RATE,
           (unsigned int)LIVE_WINDOW_SAMPLES,
           (unsigned int)(LIVE_WINDOW_SAMPLES * 1000u / LIVE_SAMPLE_RATE),
           (unsigned int)LIVE_CHUNK,
           (unsigned int)LIVE_PREAMBLE_SAMPLES);
#endif

    for (;;) {
        uint16_t avg = 0;

        if (!mic_read_chunk(&avg))
            continue;

        sample_count += LIVE_CHUNK;

        /* Adaptive noise floor (shared design, ported from the 8-bit VAD):
           running mean during warmup, asymmetric EMA while idle (rise slow,
           fall fast) so speech doesn't inflate it. threshold = nf * K
           self-normalizes to this board's scale (replaces the old fixed
           350/100 thresholds so v0 and v1 detect identically). */
        float L = (float)avg;
        if (vad_warmup < LIVE_VAD_WARMUP_CHUNKS) {
            nf += (L - nf) / (float)(vad_warmup + 1u);
            if (++vad_warmup == LIVE_VAD_WARMUP_CHUNKS) nf_seed = (nf < 1.0f) ? 1.0f : nf;
        } else if (!collecting) {
            nf += (L > nf ? (1.0f / 64.0f) : (1.0f / 8.0f)) * (L - nf);
        }
        if (nf < 1.0f) nf = 1.0f;
        if (nf < nf_seed) nf = nf_seed;   /* never collapse */
        uint32_t thr_high = (uint32_t)(nf * LIVE_VAD_K_HIGH);
        uint32_t thr_low  = (uint32_t)(nf * LIVE_VAD_K_LOW);

#if LIVE_BENCH_ENABLED
        /* Normalized mic level (×100 of noise floor) + unified threshold — ~25 Hz —
           for the dashboard's "Mic level" plot (matches the 8-bit firmware). */
        {
            static uint32_t level_div = 0;
            if ((++level_div % 5u) == 0u) {
                uint32_t norm = (uint32_t)(100.0f * L / nf);
                printf("BENCH,event=level,rms=%lu,thr=%lu\r\n",
                       (unsigned long)norm, (unsigned long)(uint32_t)(100.0f * LIVE_VAD_K_HIGH));
            }
        }
#endif

        /* need at least PREAMBLE_SAMPLES before we can trigger */
        if (ring_filled < LIVE_PREAMBLE_SAMPLES)
            continue;
        if (vad_warmup < LIVE_VAD_WARMUP_CHUNKS)
            continue;   /* wait for nf seed before detecting */

        if (!collecting) {
            if (avg >= thr_high) {
                collecting     = 1;
                ai_counter     = LIVE_PREAMBLE_SAMPLES;
                silence_run    = 0;
                printf("Trigger detected (avg=%u thr=%lu)\r\n", avg, (unsigned long)thr_high);
            }
        } else {
            ai_counter += LIVE_CHUNK;

            if (avg < thr_low && ai_counter >= LIVE_WINDOW_SAMPLES / 3u)
                silence_run++;
            else
                silence_run = 0;

            uint8_t end_of_word = (silence_run >= LIVE_SILENCE_THRESH)
                               || (ai_counter >= LIVE_POST_MAX_SAMPLES);

            if (end_of_word) {
                /* extract utterance from ring: preamble + collected post-trigger */
                uint32_t total     = (ai_counter < LIVE_WINDOW_SAMPLES)
                                     ? ai_counter : LIVE_WINDOW_SAMPLES;
                uint32_t offset    = total;   /* samples back from current head */
                ring_extract(offset, LIVE_WINDOW_SAMPLES);

                run_inference(run_idx++, ai_counter - LIVE_PREAMBLE_SAMPLES);

                /* reset state */
                collecting  = 0;
                ai_counter  = 0;
                silence_run = 0;

                printf("Waiting for next keyword...\r\n");
            }
        }
    }
}
