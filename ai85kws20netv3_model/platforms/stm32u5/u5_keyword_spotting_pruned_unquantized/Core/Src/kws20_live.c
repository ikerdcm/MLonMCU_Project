#include "kws20_live.h"
#include "kws20_mode_config.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "main.h"
#include "network.h"
#include "network_data.h"
#include "network_data_params.h"

#include "../../Drivers/BSP/B-U585I-IOT02A/b_u585i_iot02a_audio.h"

#define KWS_INPUT_SIZE              16384u
#define KWS_OUTPUT_SIZE             21u

#define LIVE_SAMPLE_RATE            16000u
#define LIVE_DMA_SAMPLES            2048u
#define LIVE_DMA_BYTES              (LIVE_DMA_SAMPLES * sizeof(int16_t))
#define LIVE_DMA_HALF_SAMPLES       (LIVE_DMA_SAMPLES / 2u)
#define LIVE_BLOCK_FIFO_DEPTH       8u
/* BSP mapping on B-U585I-IOT02A: MIC1 -> ADF1_Filter0, MIC2 -> MDF1_Filter0.
 * Default to MIC1/ADF1 because it is the only path that has produced a stable
 * continuous stream in this project so far. */
#define LIVE_AUDIO_DEVICE           AUDIO_IN_DEVICE_DIGITAL_MIC2

#define DEMO_CHUNK                  128u
#define DEMO_PREAMBLE_SIZE          (30u * DEMO_CHUNK)
#define DEMO_TRIGGER_CONSEC_HIGH    3u
/* Log shows premature triggers at avg=37-56 (background noise sustains 3 chunks
 * just above 35).  Real speech in the log reaches avg 200-450.  Setting 80
 * eliminates noise triggers while still firing well before the vowel peak. */
#define DEMO_THRESHOLD_HIGH         80u
#define DEMO_QUIET_THRESHOLD        20u
#define DEMO_ARM_QUIET_CHUNKS       0u
/* Scale PCM→int8.  The STM32U5 X-CUBE-AI float model expects the same int8-as-float
 * values that the kws20_test.c static test vector uses at scale=1.0.
 * kws_five_stm32.h (confirmed working) has speech peaks of ~±68–80 int8.
 * With MDF hardware gain=-16 dB, MIC2 raw HPF peaks are ~±1000 ADC units.
 * scale=20 maps those to int8 ≈ ±78, matching the training amplitude exactly.
 * scale=32 (previous) was ±123 (1.6× over-scaled). Do NOT go above 32 — it clips
 * fricatives (the "s" in "stop") and degrades recognition for consonant-heavy words. */
#define DEMO_SAMPLE_SCALE           20
#define DEMO_INITIAL_WARMUP_SAMPLES 32000u
#define DEMO_POST_TRIGGER_SAMPLES   (KWS_INPUT_SIZE - DEMO_PREAMBLE_SIZE)
/* Silence detection after keyword onset (mirrors MAX78000 demo behaviour).
 * Collection stops early when DEMO_SILENCE_COUNTER_THRESHOLD consecutive quiet
 * chunks are seen AND at least DEMO_MIN_KEYWORD_SAMPLES post-trigger samples
 * have been collected.  The remainder of live_i8 is zero-padded.
 *
 * MAX78000 increments ai85Counter += PREAMBLE_SIZE at trigger time, so its
 * silence gate (ai85Counter >= SAMPLE_SIZE/3 = 5461) fires after only
 * 5461 - 3840 = 1621 post-trigger samples.  We must match that offset:
 *   DEMO_MIN_KEYWORD_SAMPLES = KWS_INPUT_SIZE/3 - DEMO_PREAMBLE_SIZE = 1621
 * Using KWS_INPUT_SIZE/3 alone (5461) was a bug that kept 3.4× more ambient
 * noise in the window than MAX78000, since the silence cutoff was delayed. */
#define DEMO_SILENCE_COUNTER_THRESHOLD 20u
#define DEMO_MIN_KEYWORD_SAMPLES    (KWS_INPUT_SIZE / 3u - DEMO_PREAMBLE_SIZE)

typedef struct {
    uint32_t ring_index;
    uint32_t ring_filled;
    uint32_t global_samples;
    uint32_t chunk_count;
    int32_t chunk_sum;
    int16_t chunk_samples[DEMO_CHUNK];
    uint32_t state;
    uint32_t trigger_ring_index;
    uint32_t post_trigger_samples;
    uint32_t high_run;
    uint32_t quiet_run;
    uint32_t silence_run;
    uint32_t warmup_samples;
} live_capture_ctx_t;

static ai_u8 activations[AI_NETWORK_DATA_ACTIVATIONS_SIZE];

static ai_float ai_input_data[KWS_INPUT_SIZE];
static ai_float ai_output_data[KWS_OUTPUT_SIZE];

static int16_t mic_dma[LIVE_DMA_SAMPLES];
static int16_t live_block_fifo[LIVE_BLOCK_FIFO_DEPTH][LIVE_DMA_HALF_SAMPLES];
static int8_t live_i8[KWS_INPUT_SIZE];
static int8_t mic_ring[KWS_INPUT_SIZE];

static volatile uint32_t live_dma_half_count = 0;
static volatile uint32_t live_dma_full_count = 0;
static volatile uint32_t live_dma_error_count = 0;
static volatile uint32_t live_block_write_seq = 0;
static volatile uint32_t live_block_read_seq = 0;
static volatile uint32_t live_block_overrun_count = 0;
static volatile uint32_t live_capture_enabled = 0;
static volatile uint32_t live_dma_active = 0;
static uint32_t live_stream_primed = 0;
static uint32_t live_last_post_trigger_samples = 0;

static int live_capture_ok = 0;

static const char *labels[KWS_OUTPUT_SIZE] = {
    "up", "down", "left", "right", "stop", "go",
    "yes", "no", "on", "off",
    "one", "two", "three", "four", "five",
    "six", "seven", "eight", "nine", "zero",
    "unknown"
};

#if KWS20_CFG_ENABLE_MEASURE && KWS20_CFG_MEASURE_LIVE
#define KWS20_LIVE_BENCH_ENABLED 1
#else
#define KWS20_LIVE_BENCH_ENABLED 0
#endif

#if KWS20_LIVE_BENCH_ENABLED && KWS20_CFG_LIVE_MEASURE_MINIMAL_OUTPUT
#define KWS20_LIVE_MINIMAL_OUTPUT_ENABLED 1
#else
#define KWS20_LIVE_MINIMAL_OUTPUT_ENABLED 0
#endif

static void live_dwt_init(void)
{
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

static int32_t iabs32(int32_t x)
{
    return (x < 0) ? -x : x;
}

static int16_t live_hpf_x1 = 0;
static int32_t live_hpf_y1 = 0;

static const char *live_audio_device_desc(void)
{
    return (LIVE_AUDIO_DEVICE == AUDIO_IN_DEVICE_DIGITAL_MIC1)
               ? "DIGITAL_MIC1 / ADF1_Filter0"
               : "DIGITAL_MIC2 / MDF1_Filter0";
}

static uint32_t live_audio_callback_slot(void)
{
    return (LIVE_AUDIO_DEVICE == AUDIO_IN_DEVICE_DIGITAL_MIC1) ? 0u : 1u;
}

static void live_hpf_init(void)
{
    live_hpf_x1 = 0;
    live_hpf_y1 = 0;
}

static int16_t live_hpf(int16_t input)
{
    const int32_t coeff = 32604;
    int32_t acc = (coeff * live_hpf_y1 + (1 << 14)) >> 15;
    int32_t y = (int32_t)input - (int32_t)live_hpf_x1 + acc;

    if (y > 32767) y = 32767;
    if (y < -32768) y = -32768;

    live_hpf_x1 = input;
    live_hpf_y1 = y;

    return (int16_t)y;
}

/* Copy preamble + num_post_samples from ring into live_i8 and zero-pad the rest.
 * Passing DEMO_POST_TRIGGER_SAMPLES for num_post_samples copies the full window
 * without padding (original behaviour).  Passing a smaller value zero-pads the
 * tail, which mirrors the MAX78000 demo's silence-triggered early cutoff. */
static void copy_demo_ring_to_live_i8(uint32_t trigger_ring_index, uint32_t num_post_samples)
{
    uint32_t start = (trigger_ring_index + KWS_INPUT_SIZE - DEMO_PREAMBLE_SIZE) % KWS_INPUT_SIZE;
    uint32_t valid = DEMO_PREAMBLE_SIZE + num_post_samples;

    if (valid > KWS_INPUT_SIZE) {
        valid = KWS_INPUT_SIZE;
    }

    for (uint32_t i = 0; i < valid; i++) {
        live_i8[i] = mic_ring[(start + i) % KWS_INPUT_SIZE];
    }
    for (uint32_t i = valid; i < KWS_INPUT_SIZE; i++) {
        live_i8[i] = 0;
    }
}

static int audio_stream_start(void)
{
    BSP_AUDIO_Init_t AudioInit;
    int32_t ret;

    if (live_dma_active != 0u) {
        return 1;
    }

    memset(mic_dma, 0, sizeof(mic_dma));
    live_dma_half_count = 0;
    live_dma_full_count = 0;
    live_dma_error_count = 0;
    live_block_write_seq = 0;
    live_block_read_seq = 0;
    live_block_overrun_count = 0;
    live_capture_enabled = 0;
    live_stream_primed = 0;

    AudioInit.Device = LIVE_AUDIO_DEVICE;
    AudioInit.SampleRate = AUDIO_FREQUENCY_16K;
    AudioInit.BitsPerSample = AUDIO_RESOLUTION_16B;
    AudioInit.ChannelsNbr = 1;
    AudioInit.Volume = 100;

#if !KWS20_LIVE_MINIMAL_OUTPUT_ENABLED
    printf("Audio path: %s\r\n", live_audio_device_desc());
#endif
    ret = BSP_AUDIO_IN_Init(0, &AudioInit);
    if (ret != BSP_ERROR_NONE) {
        printf("BSP_AUDIO_IN_Init returned: %ld\r\n", (long)ret);
        return 0;
    }

#if !KWS20_LIVE_MINIMAL_OUTPUT_ENABLED
    printf("Starting DMA capture on %s...\r\n", live_audio_device_desc());
#endif
    ret = BSP_AUDIO_IN_Record(0, (uint8_t *)mic_dma, LIVE_DMA_BYTES);
    if (ret != BSP_ERROR_NONE) {
        printf("BSP_AUDIO_IN_Record returned: %ld\r\n", (long)ret);
        BSP_AUDIO_IN_DeInit(0);
        return 0;
    }

    live_dma_active = 1;
    return 1;
}

static int process_capture_sample(live_capture_ctx_t *ctx, int16_t raw)
{
    int16_t hpf = live_hpf(raw);
    int32_t v;

    ctx->global_samples++;
    if (ctx->global_samples <= ctx->warmup_samples) {
        if (ctx->global_samples == ctx->warmup_samples) {
            live_stream_primed = 1;
#if !KWS20_LIVE_MINIMAL_OUTPUT_ENABLED
            printf("audio warmup finished after %lu samples\r\n",
                   (unsigned long)ctx->warmup_samples);
#endif
        }
        return 0;
    }

    v = ((int32_t)hpf * DEMO_SAMPLE_SCALE) / 256;
    if (v > 127) v = 127;
    if (v < -128) v = -128;

    mic_ring[ctx->ring_index] = (int8_t)v;
    ctx->ring_index = (ctx->ring_index + 1u) % KWS_INPUT_SIZE;
    if (ctx->ring_filled < KWS_INPUT_SIZE) {
        ctx->ring_filled++;
    }

    if (ctx->state != 0u) {
        ctx->post_trigger_samples++;
    }

    ctx->chunk_samples[ctx->chunk_count] = hpf;
    ctx->chunk_sum += (int32_t)hpf;
    ctx->chunk_count++;

    if (ctx->chunk_count < DEMO_CHUNK) {
        return 0;
    }

    {
        int32_t chunk_mean = ctx->chunk_sum / (int32_t)DEMO_CHUNK;
        uint32_t chunk_sum_abs = 0;

        for (uint32_t i = 0; i < DEMO_CHUNK; i++) {
            chunk_sum_abs += iabs32((int32_t)ctx->chunk_samples[i] - chunk_mean);
        }

        ctx->chunk_count = 0;
        ctx->chunk_sum = 0;
        {
            uint32_t avg = chunk_sum_abs / DEMO_CHUNK;

            if (ctx->ring_filled < DEMO_PREAMBLE_SIZE) {
                return 0;
            }

            if (ctx->state == 0u) {
                if (avg <= DEMO_QUIET_THRESHOLD) {
                    if (ctx->quiet_run < DEMO_ARM_QUIET_CHUNKS) {
                        ctx->quiet_run++;
                    }
                } else {
                    ctx->quiet_run = 0;
                }

                if (ctx->quiet_run < DEMO_ARM_QUIET_CHUNKS) {
                    ctx->high_run = 0;
                    return 0;
                }

                if (avg >= DEMO_THRESHOLD_HIGH) {
                    ctx->high_run++;
                } else {
                    ctx->high_run = 0;
                }

                if (ctx->high_run >= DEMO_TRIGGER_CONSEC_HIGH) {
                    ctx->state = 1u;
                    ctx->trigger_ring_index = ctx->ring_index;
                    ctx->post_trigger_samples = 0;
                    ctx->silence_run = 0;
                    ctx->high_run = 0;

                }
            } else {
                /* --- silence detection after keyword onset (like MAX78000 demo) --- */
                if ((avg < DEMO_QUIET_THRESHOLD) &&
                    (ctx->post_trigger_samples >= DEMO_MIN_KEYWORD_SAMPLES)) {
                    ctx->silence_run++;
                } else {
                    ctx->silence_run = 0;
                }

                if (ctx->silence_run >= DEMO_SILENCE_COUNTER_THRESHOLD) {
                    copy_demo_ring_to_live_i8(ctx->trigger_ring_index,
                                              ctx->post_trigger_samples);
                    live_last_post_trigger_samples = ctx->post_trigger_samples;
                    return 1;
                }

                if (ctx->post_trigger_samples >= DEMO_POST_TRIGGER_SAMPLES) {
                    copy_demo_ring_to_live_i8(ctx->trigger_ring_index,
                                              DEMO_POST_TRIGGER_SAMPLES);
                    live_last_post_trigger_samples = DEMO_POST_TRIGGER_SAMPLES;
                    return 1;
                }
            }
        }
    }

    return 0;
}

static int process_capture_block(live_capture_ctx_t *ctx, const int16_t *samples, uint32_t sample_count)
{
    for (uint32_t i = 0; i < sample_count; i++) {
        if (process_capture_sample(ctx, samples[i]) != 0) {
            return 1;
        }
    }

    return 0;
}

static int capture_keyword_like_demo_to_i8(void)
{
    live_capture_ctx_t ctx;
    int capture_result = 0;

    memset(&ctx, 0, sizeof(ctx));
    memset(mic_ring, 0, sizeof(mic_ring));
    memset(live_i8, 0, sizeof(live_i8));
    live_last_post_trigger_samples = 0;

    live_hpf_init();
    ctx.warmup_samples = live_stream_primed ? 0u : DEMO_INITIAL_WARMUP_SAMPLES;
    live_block_read_seq = live_block_write_seq;
    live_block_overrun_count = 0;
    live_capture_enabled = 1;

    for (;;) {
        while (live_block_read_seq != live_block_write_seq) {
            uint32_t block_index = live_block_read_seq % LIVE_BLOCK_FIFO_DEPTH;
            const int16_t *block = live_block_fifo[block_index];
            live_block_read_seq++;

            if (process_capture_block(&ctx, block, LIVE_DMA_HALF_SAMPLES) != 0) {
                capture_result = 1;
                goto capture_done;
            }
        }

        if (live_block_overrun_count != 0u) {
            printf("Audio block FIFO overrun count: %lu\r\n",
                   (unsigned long)live_block_overrun_count);
            goto capture_done;
        }

        if (live_dma_error_count != 0u) {
            printf("BSP audio DMA error count: %lu\r\n",
                   (unsigned long)live_dma_error_count);
            goto capture_done;
        }

        HAL_Delay(1);
    }

capture_done:
    live_capture_enabled = 0;
    live_block_read_seq = live_block_write_seq;
    return capture_result;
}

static void record_audio_once(void)
{
    live_capture_ok = 0;

    if (!audio_stream_start()) {
        return;
    }

    live_capture_ok = capture_keyword_like_demo_to_i8();
}

static void load_ai_input_from_i8(int transpose, int invert)
{
    memset(ai_input_data, 0, sizeof(ai_input_data));

    for (uint32_t i = 0; i < KWS_INPUT_SIZE; i++) {
        uint32_t src_i = i;
        int32_t v;

        if (transpose != 0) {
            uint32_t row = i / 128u;
            uint32_t col = i % 128u;
            src_i = col * 128u + row;
        }

        v = (int32_t)live_i8[src_i];
        if (invert != 0) {
            v = -v;
            if (v > 127) v = 127;
            if (v < -128) v = -128;
        }

        ai_input_data[i] = (ai_float)v;
    }
}

static void print_top5(void)
{
    int used[KWS_OUTPUT_SIZE];
    memset(used, 0, sizeof(used));

    printf("top5:\r\n");

    for (uint32_t rank = 0; rank < 5u; rank++) {
        uint32_t best = 0;
        ai_float best_val = -3.4e38f;

        for (uint32_t i = 0; i < KWS_OUTPUT_SIZE; i++) {
            if (!used[i] && ai_output_data[i] > best_val) {
                best_val = ai_output_data[i];
                best = i;
            }
        }

        used[best] = 1;
        printf("  %02lu %-8s %ld\r\n",
               (unsigned long)best,
               labels[best],
               (long)(best_val * 1000.0f));
    }
}

static void run_inference(ai_handle network,
                          ai_buffer *ai_input,
                          ai_buffer *ai_output,
                          int transpose,
                          int invert,
                          uint32_t run_index,
                          uint32_t hclk_hz,
                          uint32_t post_trigger_samples)
{
    ai_i32 batch;
    uint32_t best = 0;
    ai_float best_val;
#if KWS20_LIVE_BENCH_ENABLED
    uint32_t start_cycles;
    uint32_t end_cycles;
    uint32_t cycles;
    uint32_t time_us;
#else
    (void)run_index;
    (void)hclk_hz;
    (void)post_trigger_samples;
#endif

    memset(ai_output_data, 0, sizeof(ai_output_data));
    load_ai_input_from_i8(transpose, invert);

#if KWS20_LIVE_BENCH_ENABLED
    DWT->CYCCNT = 0;
    start_cycles = DWT->CYCCNT;
#endif
    batch = ai_network_run(network, ai_input, ai_output);
#if KWS20_LIVE_BENCH_ENABLED
    end_cycles = DWT->CYCCNT;
#endif
    if (batch != 1) {
        ai_error err = ai_network_get_error(network);
        printf("ai_network_run failed: batch=%ld type=%d code=%d\r\n",
               (long)batch, err.type, err.code);
        return;
    }

    best_val = ai_output_data[0];
    for (uint32_t i = 1; i < KWS_OUTPUT_SIZE; i++) {
        if (ai_output_data[i] > best_val) {
            best_val = ai_output_data[i];
            best = i;
        }
    }

#if KWS20_LIVE_BENCH_ENABLED
    cycles = end_cycles - start_cycles;
    time_us = (hclk_hz > 0u) ? (uint32_t)(((uint64_t)cycles * 1000000ULL) / hclk_hz) : 0u;

    printf("BENCH,event=inference,run=%lu,mode=live,layout=%s,invert=%d,cnn_us=%lu,cycles=%lu,pred_idx=%lu,post_trigger_samples=%lu,audio_window_ms=%lu\r\n",
           (unsigned long)run_index,
           (transpose != 0) ? "simple_transpose" : "raw",
           invert,
           (unsigned long)time_us,
           (unsigned long)cycles,
           (unsigned long)best,
           (unsigned long)post_trigger_samples,
           (unsigned long)(((uint64_t)(DEMO_PREAMBLE_SIZE + post_trigger_samples) * 1000ULL) / LIVE_SAMPLE_RATE));
#endif

#if !KWS20_LIVE_MINIMAL_OUTPUT_ENABLED
    printf("\r\nLIVE inference layout=%s invert=%d\r\n",
           (transpose != 0) ? "simple_transpose" : "raw",
           invert);
    printf("best index: %lu predicted: %s logit_x1000: %ld\r\n",
           (unsigned long)best,
           labels[best],
           (long)(best_val * 1000.0f));
    print_top5();
#endif
}

int kws20_live_audio_half_callback(uint32_t instance)
{
    uint32_t write_seq;
    uint32_t read_seq;

    if (live_dma_active == 0u || instance != live_audio_callback_slot()) {
        return 0;
    }

    live_dma_half_count++;
    if (live_capture_enabled == 0u) {
        return 1;
    }

    write_seq = live_block_write_seq;
    read_seq = live_block_read_seq;
    if ((write_seq - read_seq) >= LIVE_BLOCK_FIFO_DEPTH) {
        live_block_overrun_count++;
        return 1;
    }

    memcpy(live_block_fifo[write_seq % LIVE_BLOCK_FIFO_DEPTH],
           &mic_dma[0],
           LIVE_DMA_HALF_SAMPLES * sizeof(int16_t));
    live_block_write_seq = write_seq + 1u;
    return 1;
}

int kws20_live_audio_full_callback(uint32_t instance)
{
    uint32_t write_seq;
    uint32_t read_seq;

    if (live_dma_active == 0u || instance != live_audio_callback_slot()) {
        return 0;
    }

    live_dma_full_count++;
    if (live_capture_enabled == 0u) {
        return 1;
    }

    write_seq = live_block_write_seq;
    read_seq = live_block_read_seq;
    if ((write_seq - read_seq) >= LIVE_BLOCK_FIFO_DEPTH) {
        live_block_overrun_count++;
        return 1;
    }

    memcpy(live_block_fifo[write_seq % LIVE_BLOCK_FIFO_DEPTH],
           &mic_dma[LIVE_DMA_HALF_SAMPLES],
           LIVE_DMA_HALF_SAMPLES * sizeof(int16_t));
    live_block_write_seq = write_seq + 1u;
    return 1;
}

int kws20_live_audio_error_callback(uint32_t instance)
{
    if (live_dma_active == 0u) {
        return 0;
    }

    if ((instance != 0u) && (instance != live_audio_callback_slot())) {
        return 0;
    }

    live_dma_error_count++;
    return 1;
}

void kws20_live_run_once(void)
{
    ai_handle network = AI_HANDLE_NULL;
    ai_handle act_addr[] = { activations };
    ai_error err = ai_network_create_and_init(&network, act_addr, NULL);
    uint32_t run_index = 0;
    uint32_t hclk_hz = HAL_RCC_GetHCLKFreq();

    if (err.type != AI_ERROR_NONE) {
        printf("ai_network_create_and_init failed: type=%d code=%d\r\n", err.type, err.code);
        return;
    }

    {
        ai_buffer *ai_input = ai_network_inputs_get(network, NULL);
        ai_buffer *ai_output = ai_network_outputs_get(network, NULL);

        if ((ai_input == NULL) || (ai_output == NULL)) {
            printf("ai_network_inputs_get / outputs_get failed\r\n");
            ai_network_destroy(network);
            return;
        }

        ai_input[0].data = AI_HANDLE_PTR(ai_input_data);
        ai_output[0].data = AI_HANDLE_PTR(ai_output_data);
        live_dwt_init();

#if KWS20_LIVE_BENCH_ENABLED
        printf("BENCH,event=model_info,mode=live,hclk_hz=%lu,input_elems=%u,input_bytes=%u,output_elems=%u,output_bytes=%u,activations_bytes=%u,weights_bytes=%u,sample_rate_hz=%u,full_window_samples=%u\r\n",
               (unsigned long)hclk_hz,
               (unsigned int)AI_NETWORK_IN_1_SIZE,
               (unsigned int)AI_NETWORK_IN_1_SIZE_BYTES,
               (unsigned int)AI_NETWORK_OUT_1_SIZE,
               (unsigned int)AI_NETWORK_OUT_1_SIZE_BYTES,
               (unsigned int)AI_NETWORK_DATA_ACTIVATIONS_SIZE,
               (unsigned int)AI_NETWORK_DATA_WEIGHTS_SIZE,
               (unsigned int)LIVE_SAMPLE_RATE,
               (unsigned int)KWS_INPUT_SIZE);
        printf("BENCH,event=acquisition,mode=live,sample_rate_hz=%u,sample_count=%u,audio_window_ms=%u,chunk=%u,preamble=%u,post_trigger_max=%u\r\n",
               (unsigned int)LIVE_SAMPLE_RATE,
               (unsigned int)KWS_INPUT_SIZE,
               (unsigned int)(((uint64_t)KWS_INPUT_SIZE * 1000ULL) / LIVE_SAMPLE_RATE),
               (unsigned int)DEMO_CHUNK,
               (unsigned int)DEMO_PREAMBLE_SIZE,
               (unsigned int)DEMO_POST_TRIGGER_SAMPLES);
#endif

        for (;;) {
            uint32_t post_trigger_samples;
            record_audio_once();

            if (!live_capture_ok) {
                continue;
            }

            post_trigger_samples = live_last_post_trigger_samples;
            if (post_trigger_samples == 0u) {
                post_trigger_samples = DEMO_POST_TRIGGER_SAMPLES;
            }

            run_inference(network, ai_input, ai_output, 0, 0, run_index, hclk_hz, post_trigger_samples);
            run_index++;

#if !KWS20_LIVE_MINIMAL_OUTPUT_ENABLED
            printf("\r\nKWS20 live inference done. Waiting for next input...\r\n");
#endif
        }
    }
}
