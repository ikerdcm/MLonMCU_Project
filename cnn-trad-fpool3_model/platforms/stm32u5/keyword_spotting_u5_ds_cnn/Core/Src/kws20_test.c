jetzt #include "kws20_test.h"
#include "kws20_model_io.h"
#include "ds_cnn_test_input_left.h"

#include "main.h"
#include "network.h"
#include "network_data.h"
#include "ai_platform.h"

#include <stdio.h>
#include <string.h>

static const char *labels[AI_NETWORK_OUT_1_SIZE] = {
    "down", "go", "left", "no", "off", "on",
    "right", "stop", "up", "yes", "silence", "unknown"
};

AI_ALIGNED(32)
static ai_u8 activations[AI_NETWORK_DATA_ACTIVATIONS_SIZE];

AI_ALIGNED(32)
static kws20_input_elem_t ai_input_data[AI_NETWORK_IN_1_SIZE];

AI_ALIGNED(32)
static kws20_output_elem_t ai_output_data[AI_NETWORK_OUT_1_SIZE];

static void dwt_init(void)
{
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

static void fill_test_input(void)
{
    for (uint32_t i = 0; i < AI_NETWORK_IN_1_SIZE && i < DS_CNN_TEST_INPUT_LEFT_SIZE; i++) {
        ai_input_data[i] = kws20_input_from_float(ds_cnn_test_input_left[i]);
    }
}

static void output_score_range(const ai_buffer *ai_output, float *min_score, float *max_score)
{
    float min_v = kws20_output_score(ai_output, 0, ai_output_data[0]);
    float max_v = min_v;

    for (uint32_t i = 1; i < AI_NETWORK_OUT_1_SIZE; i++) {
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

static void print_top5(const ai_buffer *ai_output, float min_score, float max_score)
{
    int used[AI_NETWORK_OUT_1_SIZE];
    memset(used, 0, sizeof(used));

    printf("top5:\r\n");

    for (uint32_t rank = 0; rank < 5u && rank < AI_NETWORK_OUT_1_SIZE; rank++) {
        uint32_t best = 0;
        float best_val = -3.4e38f;

        for (uint32_t i = 0; i < AI_NETWORK_OUT_1_SIZE; i++) {
            float score = kws20_output_score(ai_output, i, ai_output_data[i]);
            if (!used[i] && score > best_val) {
                best = i;
                best_val = score;
            }
        }

        used[best] = 1;
        printf("  %02lu %-8s %ld\r\n",
               (unsigned long)best,
               labels[best],
               score_to_norm_x1000(best_val, min_score, max_score));
    }
}

void kws20_test_run_once(void)
{
    ai_handle network = AI_HANDLE_NULL;
    const ai_handle act_addr[] = { activations };
    ai_error err = ai_network_create_and_init(&network, act_addr, NULL);
    uint32_t hclk = HAL_RCC_GetHCLKFreq();

    printf("\r\n==============================\r\n");
    printf("DS-CNN OFFLINE TEST\r\n");
    printf("==============================\r\n");

    if (err.type != AI_ERROR_NONE) {
        printf("ai_network_create_and_init failed: type=%d code=%d\r\n", err.type, err.code);
        return;
    }

    {
        ai_buffer *ai_input = ai_network_inputs_get(network, NULL);
        ai_buffer *ai_output = ai_network_outputs_get(network, NULL);
        ai_i32 batch;
        uint32_t start_cycles;
        uint32_t end_cycles;
        uint32_t cycles;
        uint32_t time_us;
        uint32_t best = 0;
        float best_val;
        float min_score;
        float max_score;

        if ((ai_input == NULL) || (ai_output == NULL)) {
            printf("ai_network_inputs_get / outputs_get failed\r\n");
            ai_network_destroy(network);
            return;
        }

        fill_test_input();
        memcpy(ai_input[0].data, ai_input_data, AI_NETWORK_IN_1_SIZE_BYTES);
        dwt_init();

        memset(ai_output_data, 0, sizeof(ai_output_data));
        DWT->CYCCNT = 0;
        start_cycles = DWT->CYCCNT;
        batch = ai_network_run(network, ai_input, ai_output);
        end_cycles = DWT->CYCCNT;

        if (batch != 1) {
            ai_error run_err = ai_network_get_error(network);
            printf("ai_network_run failed: type=%d code=%d\r\n", run_err.type, run_err.code);
            ai_network_destroy(network);
            return;
        }

        memcpy(ai_output_data, ai_output[0].data, AI_NETWORK_OUT_1_SIZE_BYTES);

        cycles = end_cycles - start_cycles;
        time_us = (hclk > 0u) ? (uint32_t)(((uint64_t)cycles * 1000000ULL) / hclk) : 0u;

        best_val = kws20_output_score(ai_output, 0, ai_output_data[0]);
        for (uint32_t i = 1; i < AI_NETWORK_OUT_1_SIZE; i++) {
            float score = kws20_output_score(ai_output, i, ai_output_data[i]);
            if (score > best_val) {
                best = i;
                best_val = score;
            }
        }

        output_score_range(ai_output, &min_score, &max_score);

        printf("feature input: MFCC of 'left' keyword (%u elems, expected idx=%d)\r\n",
               (unsigned int)AI_NETWORK_IN_1_SIZE, DS_CNN_TEST_INPUT_LEFT_EXPECTED_IDX);
        printf("cycles: %lu  time_us: %lu\r\n",
               (unsigned long)cycles, (unsigned long)time_us);
        printf("best index: %lu  predicted: %s  conf_x1000: %ld  %s\r\n",
               (unsigned long)best,
               labels[best],
               score_to_norm_x1000(best_val, min_score, max_score),
               (best == DS_CNN_TEST_INPUT_LEFT_EXPECTED_IDX) ? "[PASS]" : "[FAIL]");

        print_top5(ai_output, min_score, max_score);
    }

    ai_network_destroy(network);
}
