#include "kws20_measure.h"
#include "kws20_mode_config.h"
#include "kws20_model_io.h"

#include "main.h"
#include "network.h"
#include "network_data.h"
#include "ai_platform.h"

#include <stdio.h>
#include <string.h>

#ifndef KWS20_CFG_MEASURE_RUNS
#define KWS20_CFG_MEASURE_RUNS 50
#endif

#define DS_CNN_AUDIO_WINDOW_SAMPLES 16000u
#define DS_CNN_SAMPLE_RATE_HZ       16000u

AI_ALIGNED(32)
static ai_u8 measure_activations[AI_NETWORK_DATA_ACTIVATIONS_SIZE];

AI_ALIGNED(32)
static kws20_input_elem_t measure_input_data[AI_NETWORK_IN_1_SIZE];

AI_ALIGNED(32)
static kws20_output_elem_t measure_output_data[AI_NETWORK_OUT_1_SIZE];

static void dwt_init(void)
{
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

static void fill_measure_input(void)
{
    memset(measure_input_data, 0, sizeof(measure_input_data));
}

static int best_index(const ai_buffer *ai_output)
{
    int best = 0;
    float best_val = kws20_output_score(ai_output, 0, measure_output_data[0]);

    for (uint32_t i = 1; i < AI_NETWORK_OUT_1_SIZE; i++) {
        float score = kws20_output_score(ai_output, i, measure_output_data[i]);
        if (score > best_val) {
            best = (int)i;
            best_val = score;
        }
    }

    return best;
}

void kws20_measure_run_once(void)
{
    ai_handle network = AI_HANDLE_NULL;
    ai_error err;
    ai_buffer *ai_input;
    ai_buffer *ai_output;
    const ai_handle act_addr[] = { measure_activations };
    uint32_t hclk;

    printf("\r\n==============================\r\n");
    printf("DS-CNN MEASUREMENT MODE\r\n");
    printf("==============================\r\n");

    err = ai_network_create_and_init(&network, act_addr, NULL);
    if (err.type != AI_ERROR_NONE) {
        printf("BENCH,event=init_error,type=%d,code=%d\r\n", err.type, err.code);
        return;
    }

    ai_input = ai_network_inputs_get(network, NULL);
    ai_output = ai_network_outputs_get(network, NULL);
    if (ai_input == NULL || ai_output == NULL) {
        printf("BENCH,event=buffer_error\r\n");
        ai_network_destroy(network);
        return;
    }

    fill_measure_input();
    memcpy(ai_input[0].data, measure_input_data, AI_NETWORK_IN_1_SIZE_BYTES);
    dwt_init();
    hclk = HAL_RCC_GetHCLKFreq();

    printf("BENCH,event=model_info,mode=offline,hclk_hz=%lu,input_elems=%u,input_bytes=%u,output_elems=%u,output_bytes=%u,activations_bytes=%u,weights_bytes=%u,sample_rate_hz=%u,full_window_samples=%u,runs=%u\r\n",
           (unsigned long)hclk,
           (unsigned int)AI_NETWORK_IN_1_SIZE,
           (unsigned int)AI_NETWORK_IN_1_SIZE_BYTES,
           (unsigned int)AI_NETWORK_OUT_1_SIZE,
           (unsigned int)AI_NETWORK_OUT_1_SIZE_BYTES,
           (unsigned int)AI_NETWORK_DATA_ACTIVATIONS_SIZE,
           (unsigned int)AI_NETWORK_DATA_WEIGHTS_SIZE,
           (unsigned int)DS_CNN_SAMPLE_RATE_HZ,
           (unsigned int)DS_CNN_AUDIO_WINDOW_SAMPLES,
           (unsigned int)KWS20_CFG_MEASURE_RUNS);
    printf("BENCH,event=acquisition,mode=offline,sample_rate_hz=%u,sample_count=%u,audio_window_ms=%u,feature_frames=%u,feature_bins=%u\r\n",
           (unsigned int)DS_CNN_SAMPLE_RATE_HZ,
           (unsigned int)DS_CNN_AUDIO_WINDOW_SAMPLES,
           (unsigned int)(((uint64_t)DS_CNN_AUDIO_WINDOW_SAMPLES * 1000ULL) / DS_CNN_SAMPLE_RATE_HZ),
           49u,
           10u);

    for (uint32_t run = 0; run < KWS20_CFG_MEASURE_RUNS; run++) {
        ai_i32 batch;
        uint32_t start_cycles;
        uint32_t end_cycles;
        uint32_t cycles;
        uint32_t time_us;
        int pred;

        memset(measure_output_data, 0, sizeof(measure_output_data));
        DWT->CYCCNT = 0;
        start_cycles = DWT->CYCCNT;
        batch = ai_network_run(network, ai_input, ai_output);
        end_cycles = DWT->CYCCNT;

        if (batch != 1) {
            ai_error run_err = ai_network_get_error(network);
            printf("BENCH,event=run_error,run=%lu,type=%d,code=%d\r\n",
                   (unsigned long)run, run_err.type, run_err.code);
            ai_network_destroy(network);
            return;
        }

        memcpy(measure_output_data, ai_output[0].data, AI_NETWORK_OUT_1_SIZE_BYTES);
        cycles = end_cycles - start_cycles;
        time_us = (hclk > 0) ? (uint32_t)(((uint64_t)cycles * 1000000ULL) / hclk) : 0U;
        pred = best_index(ai_output);

        printf("BENCH,event=inference,run=%lu,mode=offline,cnn_us=%lu,cycles=%lu,pred_idx=%d\r\n",
               (unsigned long)run,
               (unsigned long)time_us,
               (unsigned long)cycles,
               pred);
    }

    ai_network_destroy(network);
    printf("BENCH,event=done\r\n");
}
