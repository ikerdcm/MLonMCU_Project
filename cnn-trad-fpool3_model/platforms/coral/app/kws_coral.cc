// DS-CNN-L keyword spotting on Coral Dev Board Micro (Edge TPU).
//
// Runs a single offline inference using the fixed "left" MFCC test vector
// and prints the result + timing over UART at 115200 baud.
//
// Build: see README.md (coralmicro SDK + CMake)
// Model: app/model_data.cc  (ds_cnn_l_static_edgetpu.tflite embedded as C array)

#include <cstdint>
#include <cstdio>
#include <cstring>

#include "libs/base/led.h"
#include "libs/base/timer.h"
#include "libs/base/uart.h"
#include "libs/tpu/edgetpu_manager.h"
#include "libs/tpu/edgetpu_op.h"
#include "third_party/freertos_kernel/include/FreeRTOS.h"
#include "third_party/freertos_kernel/include/task.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_error_reporter.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_interpreter.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "third_party/tflite-micro/tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"
#include "ds_cnn_test_input_left_int8.h"

static const char* kLabels[] = {
    "down", "go", "left", "no", "off", "on",
    "right", "stop", "up", "yes", "silence", "unknown"
};
static constexpr int kNumClasses = 12;

// Output dequantization: scale=1/256, zero_point=-128
static constexpr float kOutScale     = 0.00390625f;
static constexpr int   kOutZeroPoint = -128;

static constexpr int kTensorArenaSize = 256 * 1024;
static uint8_t tensor_arena[kTensorArenaSize] __attribute__((aligned(16)));

static void run_kws() {
    coralmicro::UartInit();
    printf("\r\n=== KWS DS-CNN-L on Coral Dev Board Micro ===\r\n");
    printf("Model size: %u bytes\r\n", g_model_data_len);

    const tflite::Model* model = tflite::GetModel(g_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("ERROR: schema version mismatch\r\n");
        return;
    }

    auto* tpu_mgr = coralmicro::EdgeTpuManager::GetSingleton();
    auto  tpu_ctx = tpu_mgr->OpenDevice();
    if (!tpu_ctx) {
        printf("ERROR: failed to open Edge TPU\r\n");
        return;
    }

    tflite::MicroMutableOpResolver<1> resolver;
    resolver.AddCustom(coralmicro::kCustomOp, coralmicro::RegisterCustomOp());

    tflite::MicroErrorReporter error_reporter;
    tflite::MicroInterpreter interpreter(
        model, resolver, tensor_arena, kTensorArenaSize, &error_reporter);

    if (interpreter.AllocateTensors() != kTfLiteOk) {
        printf("ERROR: AllocateTensors() failed\r\n");
        return;
    }

    TfLiteTensor* input  = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);

    if (input->bytes != KWS_TEST_INPUT_SIZE) {
        printf("ERROR: input size mismatch (model=%u, expected=%d)\r\n",
               (unsigned)input->bytes, KWS_TEST_INPUT_SIZE);
        return;
    }

    memcpy(input->data.int8, kws_test_input_int8, KWS_TEST_INPUT_SIZE);

    uint64_t t0 = coralmicro::TimerMicros();
    TfLiteStatus status = interpreter.Invoke();
    uint64_t t1 = coralmicro::TimerMicros();

    if (status != kTfLiteOk) {
        printf("ERROR: Invoke() failed\r\n");
        return;
    }

    uint32_t elapsed_us = (uint32_t)(t1 - t0);
    printf("Inference time: %lu us (%.3f ms)\r\n",
           (unsigned long)elapsed_us, elapsed_us / 1000.0f);

    printf("\r\nScores:\r\n");
    int   best_idx   = 0;
    float best_score = -1e9f;
    for (int i = 0; i < kNumClasses; ++i) {
        float score = (output->data.int8[i] - kOutZeroPoint) * kOutScale;
        printf("  [%2d] %-8s: %.4f\r\n", i, kLabels[i], score);
        if (score > best_score) { best_score = score; best_idx = i; }
    }

    printf("\r\nPrediction: \"%s\" (idx=%d, score=%.4f)\r\n",
           kLabels[best_idx], best_idx, best_score);
    printf("Expected  : \"%s\" (idx=%d)\r\n",
           kLabels[KWS_TEST_EXPECTED_IDX], KWS_TEST_EXPECTED_IDX);

    bool correct = (best_idx == KWS_TEST_EXPECTED_IDX);
    printf("Result    : %s\r\n", correct ? "PASS" : "FAIL");

    coralmicro::LedSet(coralmicro::Led::kUser, correct);
}

extern "C" void app_main(void* param) {
    (void)param;
    run_kws();
    while (true)
        vTaskDelay(pdMS_TO_TICKS(1000));
}
