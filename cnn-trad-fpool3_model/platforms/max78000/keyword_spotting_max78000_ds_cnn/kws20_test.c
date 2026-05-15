/**
 * DS-CNN-L offline correctness test for MAX78000 FTHR.
 * Input: MFCC of "left" keyword.  Expected prediction: index 2.
 */

#include "kws20_test.h"
#include "ds_cnn_inference.h"
#include "ds_cnn_test_input_left.h"

#include "mxc_device.h"
#include "core_cm4.h"

#include <stdio.h>
#include <string.h>

static const char *labels[12] = {
    "down", "go", "left", "no", "off", "on",
    "right", "stop", "up", "yes", "silence", "unknown"
};

static float test_scores[12];

static void dwt_init(void)
{
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL  |= DWT_CTRL_CYCCNTENA_Msk;
}

static long score_to_norm_x1000(float score, float min_score, float max_score)
{
    float norm;
    if (max_score <= min_score) return 0;
    norm = (score - min_score) / (max_score - min_score);
    if (norm < 0.0f) norm = 0.0f;
    if (norm > 1.0f) norm = 1.0f;
    return (long)(norm * 1000.0f + 0.5f);
}

static void print_top5(void)
{
    int used[12] = {0};
    float min_score = test_scores[0];
    float max_score = test_scores[0];

    for (int i = 1; i < 12; i++) {
        if (test_scores[i] < min_score) min_score = test_scores[i];
        if (test_scores[i] > max_score) max_score = test_scores[i];
    }

    printf("top5:\r\n");
    for (int rank = 0; rank < 5; rank++) {
        int best = -1;
        float best_val = -3.4e38f;
        for (int i = 0; i < 12; i++) {
            if (!used[i] && test_scores[i] > best_val) {
                best = i;
                best_val = test_scores[i];
            }
        }
        if (best < 0) break;
        used[best] = 1;
        printf("  %02d %-8s %ld\r\n",
               best,
               labels[best],
               score_to_norm_x1000(best_val, min_score, max_score));
    }
}

void kws20_test_run_once(void)
{
    uint32_t hclk = SystemCoreClock;
    uint32_t start_cycles;
    uint32_t end_cycles;
    uint32_t cycles;
    uint32_t time_us;
    int pred;

    printf("\r\n==============================\r\n");
    printf("DS-CNN OFFLINE TEST\r\n");
    printf("==============================\r\n");

    dwt_init();

    memset(test_scores, 0, sizeof(test_scores));
    DWT->CYCCNT = 0;
    start_cycles = DWT->CYCCNT;
    pred = ds_cnn_infer(ds_cnn_test_input_left, test_scores);
    end_cycles = DWT->CYCCNT;

    cycles  = end_cycles - start_cycles;
    time_us = (hclk > 0u)
              ? (uint32_t)(((uint64_t)cycles * 1000000ULL) / (uint64_t)hclk)
              : 0u;

    printf("feature input: MFCC of 'left' keyword (%u elems, expected idx=%d)\r\n",
           (unsigned int)DS_CNN_TEST_INPUT_LEFT_SIZE,
           (int)DS_CNN_TEST_INPUT_LEFT_EXPECTED_IDX);
    printf("cycles: %lu  time_us: %lu\r\n",
           (unsigned long)cycles, (unsigned long)time_us);
    printf("best index: %d  predicted: %s  %s\r\n",
           pred,
           (pred >= 0 && pred < 12) ? labels[pred] : "?",
           (pred == DS_CNN_TEST_INPUT_LEFT_EXPECTED_IDX) ? "[PASS]" : "[FAIL]");

    print_top5();
}
