#include "kws20_measure.h"

#include "kws_five_stm32.h"
#include "main.h"
#include "network.h"
#include "network_data.h"
#include "ai_platform.h"

#include <stdio.h>
#include <string.h>

#define KWS_MEASURE_RUNS 50

AI_ALIGNED(32)
static ai_u8 measure_activations[AI_NETWORK_DATA_ACTIVATIONS_SIZE];

AI_ALIGNED(32)
static ai_float measure_input_data[AI_NETWORK_IN_1_SIZE];

AI_ALIGNED(32)
static ai_float measure_output_data[AI_NETWORK_OUT_1_SIZE];

static void dwt_init(void)
{
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

static void fill_measure_input(void)
{
    for (uint32_t i = 0; i < AI_NETWORK_IN_1_SIZE; i++) {
        measure_input_data[i] = kws_five_stm32[i];
    }
}

static int best_index(void)
{
    int best = 0;
    float best_val = measure_output_data[0];

    for (int i = 1; i < AI_NETWORK_OUT_1_SIZE; i++) {
        if (measure_output_data[i] > best_val) {
            best = i;
            best_val = measure_output_data[i];
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
    printf("KWS20 MEASUREMENT MODE\r\n");
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

    ai_input[0].data = AI_HANDLE_PTR(measure_input_data);
    ai_output[0].data = AI_HANDLE_PTR(measure_output_data);

    fill_measure_input();
    dwt_init();
    hclk = HAL_RCC_GetHCLKFreq();

    printf("BENCH,event=model_info,hclk_hz=%lu,input_elems=%u,input_bytes=%u,output_elems=%u,output_bytes=%u,activations_bytes=%u,weights_bytes=%u,runs=%u\r\n",
           (unsigned long)hclk,
           (unsigned int)AI_NETWORK_IN_1_SIZE,
           (unsigned int)AI_NETWORK_IN_1_SIZE_BYTES,
           (unsigned int)AI_NETWORK_OUT_1_SIZE,
           (unsigned int)AI_NETWORK_OUT_1_SIZE_BYTES,
           (unsigned int)AI_NETWORK_DATA_ACTIVATIONS_SIZE,
           (unsigned int)AI_NETWORK_DATA_WEIGHTS_SIZE,
           (unsigned int)KWS_MEASURE_RUNS);
    printf("BENCH,event=acquisition,sample_rate_hz=%u,sample_count=%u,audio_window_ms=%u\r\n",
           16000U,
           (unsigned int)AI_NETWORK_IN_1_SIZE,
           (unsigned int)(((uint64_t)AI_NETWORK_IN_1_SIZE * 1000ULL) / 16000ULL));

    for (uint32_t run = 0; run < KWS_MEASURE_RUNS; run++) {
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

        cycles = end_cycles - start_cycles;
        time_us = (hclk > 0) ? (uint32_t)(((uint64_t)cycles * 1000000ULL) / hclk) : 0U;
        pred = best_index();

        printf("BENCH,event=inference,run=%lu,cnn_us=%lu,cycles=%lu,pred_idx=%d\r\n",
               (unsigned long)run,
               (unsigned long)time_us,
               (unsigned long)cycles,
               pred);
    }

    ai_network_destroy(network);
    printf("BENCH,event=done\r\n");
}
