#include "kws20_test.h"
#include "kws_five_stm32.h"
#include "kws_max78000_generated_vectors.h"

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

typedef struct {
    const char *name;
    const char *source_note;
    const float *src;
    float scale;
    int transpose;
    int expected_idx;
    int reference_idx;
} offline_case_t;

typedef struct {
    int best_idx;
    uint32_t cycles;
    uint32_t time_us;
    float best_val;
} offline_result_t;

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

static const char *label_or_qmark(int idx)
{
    if (idx >= 0 && idx < KWS_OUTPUT_SIZE) {
        return labels[idx];
    }

    return "?";
}

static void fill_input_from(const float *src, float scale, int transpose)
{
    for (int i = 0; i < KWS_INPUT_SIZE; i++) {
        int src_i = i;

        if (transpose) {
            int r = i / 128;
            int c = i % 128;
            src_i = c * 128 + r;
        }

        ai_input_data[i] = src[src_i] * scale;
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

static int find_rank(int target_idx)
{
    if (target_idx < 0 || target_idx >= KWS_OUTPUT_SIZE) {
        return -1;
    }

    int rank = 1;

    for (int i = 0; i < KWS_OUTPUT_SIZE; i++) {
        if (i != target_idx && ai_output_data[i] > ai_output_data[target_idx]) {
            rank++;
        }
    }

    return rank;
}

static void print_input_stats(const float *src, float scale, int transpose)
{
    float min_v = 0.0f;
    float max_v = 0.0f;
    double sum_abs = 0.0;
    uint32_t nonzero = 0;

    for (int i = 0; i < KWS_INPUT_SIZE; i++) {
        int src_i = i;
        if (transpose) {
            int r = i / 128;
            int c = i % 128;
            src_i = c * 128 + r;
        }

        float v = src[src_i] * scale;

        if (i == 0 || v < min_v) min_v = v;
        if (i == 0 || v > max_v) max_v = v;
        if (v != 0.0f) nonzero++;
        sum_abs += fabsf(v);
    }

    printf("input stats: min=%ld max=%ld avg_abs=%ld nonzero=%lu/%d\r\n",
           safe_x1000(min_v),
           safe_x1000(max_v),
           safe_x1000((float)(sum_abs / (double)KWS_INPUT_SIZE)),
           (unsigned long)nonzero,
           KWS_INPUT_SIZE);
}

static int run_one_with(const float *src, ai_handle network, ai_buffer *ai_input,
                        ai_buffer *ai_output, float scale, int transpose,
                        offline_result_t *result)
{
    fill_input_from(src, scale, transpose);
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
    uint32_t time_us = (hclk > 0) ? (uint32_t)(((uint64_t)cycles * 1000000ULL) / hclk) : 0;

    if (result != NULL) {
        result->best_idx = best;
        result->cycles = cycles;
        result->time_us = time_us;
        result->best_val = best_val;
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
    static const offline_case_t cases[] = {
        {
            .name = "STM32 reference tensor",
            .source_note = "kws_five_stm32.h: known-good CubeAI input tensor",
            .src = kws_five_stm32,
            .scale = 1.0f,
            .transpose = 0,
            .expected_idx = 14,
            .reference_idx = 14,
        },
        KWS_MAX78000_GENERATED_CASES
    };

    printf("\r\n==============================\r\n");
    printf("KWS20 OFFLINE EVAL\r\n");
    printf("==============================\r\n");
    printf("Direct-run tensors in this harness:\r\n");
    printf("  1) STM32 reference tensor\r\n");
    printf("  2..N) MAX78000 AI input dumps from generated registry (%d cases)\r\n",
           KWS_MAX78000_GENERATED_CASE_COUNT);
    printf("Not auto-run yet: MAX78000 demo kws_five.h\r\n");
    printf("  Reason: that file is a raw offline sample vector for the MAX frontend,\r\n");
    printf("          not the post-frontend AI input tensor used by this U5 model.\r\n");

    ai_handle network = AI_HANDLE_NULL;
    ai_error err;
    int pass = 0;
    int total = 0;

    const ai_handle act_addr[] = { activations };
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

    for (uint32_t i = 0; i < (sizeof(cases) / sizeof(cases[0])); i++) {
        offline_result_t result = {0};
        int predicted;
        int expected_rank;

        printf("\r\n--- CASE %lu: %s ---\r\n",
               (unsigned long)(i + 1),
               cases[i].name);
        printf("source: %s\r\n", cases[i].source_note);
        print_input_stats(cases[i].src, cases[i].scale, cases[i].transpose);

        predicted = run_one_with(cases[i].src, network, ai_input, ai_output,
                                 cases[i].scale, cases[i].transpose, &result);
        expected_rank = find_rank(cases[i].expected_idx);
        total++;

        printf("expected: %s  reference: %s  expected_rank=%d\r\n",
               label_or_qmark(cases[i].expected_idx),
               label_or_qmark(cases[i].reference_idx),
               expected_rank);
        printf("case summary: cycles=%lu time_us=%lu best_logit_x1000=%ld\r\n",
               (unsigned long)result.cycles,
               (unsigned long)result.time_us,
               safe_x1000(result.best_val));

        if (predicted == cases[i].expected_idx) {
            pass++;
            printf("CASE %lu PASS: %s -> %s\r\n",
                   (unsigned long)(i + 1),
                   label_or_qmark(cases[i].expected_idx),
                   label_or_qmark(predicted));
        } else {
            printf("CASE %lu FAIL: %s -> %s\r\n",
                   (unsigned long)(i + 1),
                   label_or_qmark(cases[i].expected_idx),
                   label_or_qmark(predicted));
        }

        if (i >= 1) {
            if (predicted == cases[i].reference_idx) {
                printf("       => U5 agrees with the MAX78000 dump reference on this tensor.\r\n");
            } else {
                printf("       => U5 disagrees with the MAX78000 dump reference on this tensor.\r\n");
            }
        }
    }

    ai_network_destroy(network);

    printf("\r\nOFFLINE EVAL RESULT: %d/%d PASSED\r\n", pass, total);
    printf("==============================\r\n");
}
