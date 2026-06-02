#include "kws20_live.h"
#include "ds_cnn_frontend.h"
#include "kws20_mode_config.h"
#include "kws20_model_io.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "main.h"
#include "network.h"
#include "network_data.h"
#include "network_data_params.h"

#include "../../Drivers/BSP/B-U585I-IOT02A/b_u585i_iot02a_audio.h"

#define DS_CNN_AUDIO_WINDOW_SAMPLES 16000u
#define DS_CNN_OUTPUT_SIZE          AI_NETWORK_OUT_1_SIZE
#define DS_CNN_SAMPLE_RATE          16000u
#define DS_CNN_FEATURE_FRAMES       49u
#define DS_CNN_FEATURE_BINS         10u

#define LIVE_DMA_SAMPLES            2048u
#define LIVE_DMA_BYTES              (LIVE_DMA_SAMPLES * sizeof(int16_t))
#define LIVE_DMA_HALF_SAMPLES       (LIVE_DMA_SAMPLES / 2u)
#define LIVE_BLOCK_FIFO_DEPTH       8u
#define LIVE_AUDIO_DEVICE           AUDIO_IN_DEVICE_DIGITAL_MIC2

#define DEMO_CHUNK                  128u
#define DEMO_PREAMBLE_SIZE          (30u * DEMO_CHUNK)
#define DEMO_TRIGGER_CONSEC_HIGH    3u
#define DEMO_THRESHOLD_HIGH         40u
#define DEMO_QUIET_THRESHOLD        20u
#define DEMO_ARM_QUIET_CHUNKS       0u
#define DEMO_INITIAL_WARMUP_SAMPLES 32000u
#define DEMO_POST_TRIGGER_SAMPLES   (DS_CNN_AUDIO_WINDOW_SAMPLES - DEMO_PREAMBLE_SIZE)
#define DEMO_SILENCE_COUNTER_THRESHOLD 20u
#define DEMO_MIN_KEYWORD_SAMPLES    (DS_CNN_AUDIO_WINDOW_SAMPLES / 3u - DEMO_PREAMBLE_SIZE)

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

AI_ALIGNED(32)
static ai_u8 activations[AI_NETWORK_DATA_ACTIVATIONS_SIZE];

AI_ALIGNED(32)
static kws20_input_elem_t ai_input_data[AI_NETWORK_IN_1_SIZE];

AI_ALIGNED(32)
static kws20_output_elem_t ai_output_data[AI_NETWORK_OUT_1_SIZE];

static int16_t mic_dma[LIVE_DMA_SAMPLES];
static int16_t live_block_fifo[LIVE_BLOCK_FIFO_DEPTH][LIVE_DMA_HALF_SAMPLES];
static int16_t live_audio_pcm[DS_CNN_AUDIO_WINDOW_SAMPLES];
static int16_t mic_ring_pcm[DS_CNN_AUDIO_WINDOW_SAMPLES];

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

static const char *labels[DS_CNN_OUTPUT_SIZE] = {
    "down", "go", "left", "no", "off", "on",
    "right", "stop", "up", "yes", "silence", "unknown"
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

#if !KWS20_LIVE_MINIMAL_OUTPUT_ENABLED
static long score_to_norm_x1000(float score, float min_score, float max_score)
{
    float norm;

    if (max_score <= min_score) {
        return 0;
    }

    norm = (score - min_score) / (max_score - min_score);
    if (norm < 0.0f) {
        norm = 0.0f;
    }
    if (norm > 1.0f) {
        norm = 1.0f;
    }

    return (long)(norm * 1000.0f + 0.5f);
}
#endif

#if !KWS20_LIVE_MINIMAL_OUTPUT_ENABLED
static void output_score_range(const ai_buffer *ai_output, float *min_score, float *max_score)
{
    float min_v = kws20_output_score(ai_output, 0, ai_output_data[0]);
    float max_v = min_v;

    for (uint32_t i = 1; i < DS_CNN_OUTPUT_SIZE; i++) {
        float score = kws20_output_score(ai_output, i, ai_output_data[i]);

        if (score < min_v) {
            min_v = score;
        }
        if (score > max_v) {
            max_v = score;
        }
    }

    *min_score = min_v;
    *max_score = max_v;
}
#endif

#if !KWS20_LIVE_MINIMAL_OUTPUT_ENABLED
static const char *live_audio_device_desc(void)
{
    return (LIVE_AUDIO_DEVICE == AUDIO_IN_DEVICE_DIGITAL_MIC1)
               ? "DIGITAL_MIC1 / ADF1_Filter0"
               : "DIGITAL_MIC2 / MDF1_Filter0";
}
#endif

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

static void copy_demo_ring_to_live_audio(uint32_t trigger_ring_index, uint32_t num_post_samples)
{
    uint32_t start = (trigger_ring_index + DS_CNN_AUDIO_WINDOW_SAMPLES - DEMO_PREAMBLE_SIZE) % DS_CNN_AUDIO_WINDOW_SAMPLES;
    uint32_t valid = DEMO_PREAMBLE_SIZE + num_post_samples;

    if (valid > DS_CNN_AUDIO_WINDOW_SAMPLES) {
        valid = DS_CNN_AUDIO_WINDOW_SAMPLES;
    }

    for (uint32_t i = 0; i < valid; i++) {
        live_audio_pcm[i] = mic_ring_pcm[(start + i) % DS_CNN_AUDIO_WINDOW_SAMPLES];
    }
    for (uint32_t i = valid; i < DS_CNN_AUDIO_WINDOW_SAMPLES; i++) {
        live_audio_pcm[i] = 0;
    }
}

static int process_capture_sample(live_capture_ctx_t *ctx, int16_t raw)
{
    int16_t hpf = live_hpf(raw);
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

    mic_ring_pcm[ctx->ring_index] = raw;
    ctx->ring_index = (ctx->ring_index + 1u) % DS_CNN_AUDIO_WINDOW_SAMPLES;
    if (ctx->ring_filled < DS_CNN_AUDIO_WINDOW_SAMPLES) {
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
                if ((avg < DEMO_QUIET_THRESHOLD) &&
                    (ctx->post_trigger_samples >= DEMO_MIN_KEYWORD_SAMPLES)) {
                    ctx->silence_run++;
                } else {
                    ctx->silence_run = 0;
                }

                if (ctx->silence_run >= DEMO_SILENCE_COUNTER_THRESHOLD) {
                    copy_demo_ring_to_live_audio(ctx->trigger_ring_index,
                                                 ctx->post_trigger_samples);
                    live_last_post_trigger_samples = ctx->post_trigger_samples;
                    return 1;
                }

                if (ctx->post_trigger_samples >= DEMO_POST_TRIGGER_SAMPLES) {
                    copy_demo_ring_to_live_audio(ctx->trigger_ring_index,
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

    /* Mic level (RMS) for the dashboard plot — throttled to ~25 Hz. */
    static uint32_t last_level_ms = 0u;
    uint32_t now = HAL_GetTick();
    if (sample_count && (now - last_level_ms) >= 40u) {
        last_level_ms = now;
        uint64_t acc = 0;
        for (uint32_t i = 0; i < sample_count; i++) {
            int32_t s = samples[i];
            acc += (uint32_t)(s * s);
        }
        uint32_t rms = (uint32_t)sqrtf((float)acc / (float)sample_count);
        printf("BENCH,event=level,rms=%lu\r\n", (unsigned long)rms);
    }

    return 0;
}

static int capture_keyword_like_demo_to_audio(void)
{
    live_capture_ctx_t ctx;
    int capture_result = 0;

    memset(&ctx, 0, sizeof(ctx));
    memset(mic_ring_pcm, 0, sizeof(mic_ring_pcm));
    memset(live_audio_pcm, 0, sizeof(live_audio_pcm));
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

static void record_audio_once(void)
{
    live_capture_ok = 0;

    if (!audio_stream_start()) {
        return;
    }

    live_capture_ok = capture_keyword_like_demo_to_audio();
}

static void fill_input_from_live_audio(void)
{
    memset(ai_input_data, 0, sizeof(ai_input_data));
    if (!ds_cnn_frontend_compute(live_audio_pcm,
                                 DS_CNN_AUDIO_WINDOW_SAMPLES,
                                 ai_input_data,
                                 AI_NETWORK_IN_1_SIZE)) {
        printf("ds_cnn_frontend_compute failed\r\n");
    }
}

#if !KWS20_LIVE_MINIMAL_OUTPUT_ENABLED
static void print_top5(const ai_buffer *ai_output, float min_score, float max_score)
{
    int used[DS_CNN_OUTPUT_SIZE];
    memset(used, 0, sizeof(used));

    printf("top5:\r\n");

    for (uint32_t rank = 0; rank < 5u && rank < DS_CNN_OUTPUT_SIZE; rank++) {
        uint32_t best = 0;
        float best_val = -3.4e38f;

        for (uint32_t i = 0; i < DS_CNN_OUTPUT_SIZE; i++) {
            float score = kws20_output_score(ai_output, i, ai_output_data[i]);

            if (!used[i] && score > best_val) {
                best_val = score;
                best = i;
            }
        }

        used[best] = 1;
        printf("  %02lu %-8s %ld\r\n",
               (unsigned long)best,
               labels[best],
               score_to_norm_x1000(best_val, min_score, max_score));
    }
}
#endif

static void run_inference(ai_handle network,
                          ai_buffer *ai_input,
                          ai_buffer *ai_output,
                          uint32_t run_index,
                          uint32_t hclk_hz,
                          uint32_t post_trigger_samples)
{
    ai_i32 batch;
    uint32_t best = 0;
    float best_val;
#if !KWS20_LIVE_MINIMAL_OUTPUT_ENABLED
    float min_score;
    float max_score;
#endif
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
    fill_input_from_live_audio();
    memcpy(ai_input[0].data, ai_input_data, AI_NETWORK_IN_1_SIZE_BYTES);

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

    memcpy(ai_output_data, ai_output[0].data, AI_NETWORK_OUT_1_SIZE_BYTES);
    best_val = kws20_output_score(ai_output, 0, ai_output_data[0]);
    for (uint32_t i = 1; i < DS_CNN_OUTPUT_SIZE; i++) {
        float score = kws20_output_score(ai_output, i, ai_output_data[i]);

        if (score > best_val) {
            best_val = score;
            best = i;
        }
    }

#if !KWS20_LIVE_MINIMAL_OUTPUT_ENABLED
    output_score_range(ai_output, &min_score, &max_score);
#endif

#if KWS20_LIVE_BENCH_ENABLED
    cycles = end_cycles - start_cycles;
    time_us = (hclk_hz > 0u) ? (uint32_t)(((uint64_t)cycles * 1000000ULL) / hclk_hz) : 0u;

    printf("BENCH,event=inference,run=%lu,mode=live,frontend=mfcc_tf_49x10,cnn_us=%lu,cycles=%lu,pred_idx=%lu,post_trigger_samples=%lu,audio_window_ms=%lu\r\n",
           (unsigned long)run_index,
           (unsigned long)time_us,
           (unsigned long)cycles,
           (unsigned long)best,
           (unsigned long)post_trigger_samples,
           (unsigned long)(((uint64_t)(DEMO_PREAMBLE_SIZE + post_trigger_samples) * 1000ULL) / DS_CNN_SAMPLE_RATE));
#endif

#if !KWS20_LIVE_MINIMAL_OUTPUT_ENABLED
    printf("\r\nDS-CNN live inference frontend=mfcc_tf_49x10\r\n");
    printf("best index: %lu predicted: %s conf_x1000: %ld\r\n",
           (unsigned long)best,
           labels[best],
           score_to_norm_x1000(best_val, min_score, max_score));
    print_top5(ai_output, min_score, max_score);
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
               (unsigned int)DS_CNN_SAMPLE_RATE,
               (unsigned int)DS_CNN_AUDIO_WINDOW_SAMPLES);
        printf("BENCH,event=acquisition,mode=live,sample_rate_hz=%u,sample_count=%u,audio_window_ms=%u,chunk=%u,preamble=%u,post_trigger_max=%u,feature_frames=%u,feature_bins=%u\r\n",
               (unsigned int)DS_CNN_SAMPLE_RATE,
               (unsigned int)DS_CNN_AUDIO_WINDOW_SAMPLES,
               (unsigned int)(((uint64_t)DS_CNN_AUDIO_WINDOW_SAMPLES * 1000ULL) / DS_CNN_SAMPLE_RATE),
               (unsigned int)DEMO_CHUNK,
               (unsigned int)DEMO_PREAMBLE_SIZE,
               (unsigned int)DEMO_POST_TRIGGER_SAMPLES,
               (unsigned int)DS_CNN_FEATURE_FRAMES,
               (unsigned int)DS_CNN_FEATURE_BINS);
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

            run_inference(network, ai_input, ai_output, run_index, hclk_hz, post_trigger_samples);
            run_index++;

#if !KWS20_LIVE_MINIMAL_OUTPUT_ENABLED
            printf("\r\nDS-CNN live inference done. Waiting for next input...\r\n");
#endif
        }
    }
}
