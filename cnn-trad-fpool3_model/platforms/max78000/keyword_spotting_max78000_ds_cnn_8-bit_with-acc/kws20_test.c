/*
 * Offline correctness test: run pre-computed MFCC of the "left" keyword
 * through the CNN accelerator. Expected result: class 2 ("left").
 */

#include "kws20_test.h"
#include "ds_cnn_test_input_left.h"
#include "cnn_inference.h"

#include <stdio.h>
#include <string.h>

static const char *labels[12] = {
    "down", "go", "left", "no", "off", "on",
    "right", "stop", "up", "yes", "silence", "unknown"
};

static float test_scores[12];

void kws20_test_run_once(void)
{
    int pred;

    printf("\r\n==============================\r\n");
    printf("DS-CNN v1 CORRECTNESS TEST\r\n");
    printf("Input: MFCC of 'left' keyword\r\n");
    printf("==============================\r\n");

    memset(test_scores, 0, sizeof(test_scores));
    pred = cnn_infer(ds_cnn_test_input_left, test_scores);

    printf("Predicted: %s (class %d)\r\n",
           (pred >= 0 && pred < 12) ? labels[pred] : "?", pred);
    printf("Expected:  left (class 2)\r\n");

    for (int i = 0; i < 12; i++)
        printf("  [%2d] %-8s %.4f\r\n", i, labels[i], test_scores[i]);

    if (pred == DS_CNN_TEST_INPUT_LEFT_EXPECTED_IDX)
        printf("TEST PASSED\r\n");
    else
        printf("TEST FAILED (got %d, expected %d)\r\n",
               pred, DS_CNN_TEST_INPUT_LEFT_EXPECTED_IDX);
}
