#include "kws20_test.h"
#include "kws_five_stm32.h"

#include "main.h"
#include "network.h"
#include "network_data.h"
#include "ai_platform.h"

#include <stdio.h>
#include <string.h>
#include <math.h>

#define KWS_INPUT_SIZE  (16384)
#define KWS_OUTPUT_SIZE (21)

AI_ALIGNED(32)
static ai_u8 activations[AI_NETWORK_DATA_ACTIVATIONS_SIZE];

AI_ALIGNED(32)
static ai_float ai_input_data[KWS_INPUT_SIZE];

AI_ALIGNED(32)
static ai_float ai_output_data[KWS_OUTPUT_SIZE];

static const char *labels[KWS_OUTPUT_SIZE] = {
    "up", "down", "left", "right", "stop", "go",
    "yes", "no", "on", "off",
    "one", "two", "three", "four", "five",
    "six", "seven", "eight", "nine", "zero",
    "unknown"
};

static void dwt_init(void)
{
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

static long safe_x1000(float x)
{
    if (x > 2000000.0f) return 2000000000L;
    if (x < -2000000.0f) return -2000000000L;
    return (long)(x * 1000.0f);
}

static void fill_input(float scale, int transpose)
{
    for (int i = 0; i < KWS_INPUT_SIZE; i++) {
        int src_i = i;

        if (transpose) {
            int r = i / 128;
            int c = i % 128;
            src_i = c * 128 + r;
        }

        ai_input_data[i] = kws_five_stm32[src_i] * scale;
    }
}

static void print_top5(void)
{
    int used[KWS_OUTPUT_SIZE] = {0};

    printf("top5:\r\n");

    for (int k = 0; k < 5; k++) {
        int best = -1;
        float best_val = -3.4e38f;

        for (int i = 0; i < KWS_OUTPUT_SIZE; i++) {
            if (!used[i] && ai_output_data[i] > best_val) {
                best = i;
                best_val = ai_output_data[i];
            }
        }

        used[best] = 1;

        printf("  %02d %-8s %ld\r\n",
               best,
               labels[best],
               safe_x1000(best_val));
    }
}

static int run_one(ai_handle network, ai_buffer *ai_input, ai_buffer *ai_output,
                   float scale, int transpose)
{
    fill_input(scale, transpose);
    memset(ai_output_data, 0, sizeof(ai_output_data));

    DWT->CYCCNT = 0;
    uint32_t start_cycles = DWT->CYCCNT;
    ai_i32 batch = ai_network_run(network, ai_input, ai_output);
    uint32_t end_cycles = DWT->CYCCNT;

    if (batch != 1) {
        ai_error err = ai_network_get_error(network);
        printf("AI run failed: batch=%ld type=%d code=%d\r\n",
               (long)batch, err.type, err.code);
        return -1;
    }

    int best = 0;
    float best_val = ai_output_data[0];

    for (int i = 1; i < KWS_OUTPUT_SIZE; i++) {
        if (ai_output_data[i] > best_val) {
            best = i;
            best_val = ai_output_data[i];
        }
    }

    uint32_t cycles = end_cycles - start_cycles;
    uint32_t hclk = HAL_RCC_GetHCLKFreq();
    uint32_t time_us = 0;

    if (hclk > 0) {
        time_us = (uint32_t)(((uint64_t)cycles * 1000000ULL) / hclk);
    }

    printf("\r\nscale x1e9: %ld  transpose: %d\r\n",
           (long)(scale * 1000000000.0f), transpose);
    printf("cycles: %lu  time_us: %lu\r\n",
           (unsigned long)cycles, (unsigned long)time_us);
    printf("best index: %d  predicted: %s  logit_x1000: %ld\r\n",
           best, labels[best], safe_x1000(best_val));

    print_top5();

    return best;
}

void kws20_test_run_once(void)
{
    printf("\r\n==============================\r\n");
    printf("KWS20 STM32U5 scale sweep\r\n");
    printf("==============================\r\n");

    printf("Input vector raw min/max check:\r\n");
    float mn = kws_five_stm32[0];
    float mx = kws_five_stm32[0];

    for (int i = 1; i < KWS_INPUT_SIZE; i++) {
        if (kws_five_stm32[i] < mn) mn = kws_five_stm32[i];
        if (kws_five_stm32[i] > mx) mx = kws_five_stm32[i];
    }

    printf("raw min x1000: %ld\r\n", safe_x1000(mn));
    printf("raw max x1000: %ld\r\n", safe_x1000(mx));

    ai_handle network = AI_HANDLE_NULL;
    ai_error err;

    const ai_handle act_addr[] = {
        activations
    };

    err = ai_network_create_and_init(&network, act_addr, NULL);
    if (err.type != AI_ERROR_NONE) {
        printf("AI create/init failed: type=%d code=%d\r\n", err.type, err.code);
        return;
    }

    ai_buffer *ai_input = ai_network_inputs_get(network, NULL);
    ai_buffer *ai_output = ai_network_outputs_get(network, NULL);

    if (ai_input == NULL || ai_output == NULL) {
        printf("AI buffer get failed\r\n");
        ai_network_destroy(network);
        return;
    }

    ai_input[0].data = AI_HANDLE_PTR(ai_input_data);
    ai_output[0].data = AI_HANDLE_PTR(ai_output_data);

    dwt_init();

    printf("\r\nRunning final QAT-like CubeAI test: scale=1.0 transpose=0\r\n");
    run_one(network, ai_input, ai_output, 1.0f, 0);
    printf("\r\nFinal single-run test done.\r\n");
    return;

    const float scales[] = {
        1.0f / 128.0f,
        1.0f / 256.0f,
        1.0f / 512.0f,
        1.0f / 1024.0f,
        1.0f / 2048.0f,
        1.0f / 4096.0f,
        1.0f / 8192.0f,
        1.0f / 16384.0f,
        1.0f / 32768.0f,
        1.0f / 65536.0f
    };

    int n = sizeof(scales) / sizeof(scales[0]);

    for (int t = 0; t <= 1; t++) {
        for (int i = 0; i < n; i++) {
            run_one(network, ai_input, ai_output, scales[i], t);
        }
    }

    ai_network_destroy(network);

    printf("\r\nSweep done.\r\n");
}
