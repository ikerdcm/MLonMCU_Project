// DS-CNN-L keyword spotting on Coral Dev Board Micro (Edge TPU).
// Runs one offline inference with the fixed "left" MFCC test vector.
// Output via USB-CDC (printf routed automatically by coralmicro).
// Model loaded from LittleFS (embedded via DATA in CMakeLists.txt).

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "libs/base/filesystem.h"
#include "libs/base/led.h"
#include "libs/base/timer.h"
#include "libs/tensorflow/utils.h"
#include "libs/tpu/edgetpu_manager.h"
#include "libs/tpu/edgetpu_op.h"
#include "third_party/freertos_kernel/include/FreeRTOS.h"
#include "third_party/freertos_kernel/include/task.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_error_reporter.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_interpreter.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "third_party/tflite-micro/tensorflow/lite/schema/schema_generated.h"

#include "ds_cnn_test_input_left_int8.h"

namespace coralmicro {
namespace {

const char* kModelPath = "/models/ds_cnn_l_static_edgetpu.tflite";

const char* kLabels[] = {
    "down", "go", "left", "no", "off", "on",
    "right", "stop", "up", "yes", "silence", "unknown"
};
constexpr int kNumClasses = 12;

constexpr float kOutScale     = 0.00390625f;
constexpr int   kOutZeroPoint = -128;

constexpr int kTensorArenaSize = 256 * 1024;
STATIC_TENSOR_ARENA_IN_SDRAM(tensor_arena, kTensorArenaSize);

bool run_kws() {
    printf("\r\n=== KWS DS-CNN-L on Coral Dev Board Micro ===\r\n");

    std::vector<uint8_t> model_data;
    if (!LfsReadFile(kModelPath, &model_data)) {
        printf("ERROR: failed to load model from %s\r\n", kModelPath);
        return false;
    }
    printf("Model loaded: %u bytes\r\n", (unsigned)model_data.size());

    const tflite::Model* model = tflite::GetModel(model_data.data());
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("ERROR: schema version mismatch\r\n");
        return false;
    }

    auto tpu_ctx = EdgeTpuManager::GetSingleton()->OpenDevice();
    if (!tpu_ctx) {
        printf("ERROR: failed to open Edge TPU\r\n");
        return false;
    }

    tflite::MicroErrorReporter error_reporter;
    tflite::MicroMutableOpResolver<1> resolver;
    resolver.AddCustom(kCustomOp, RegisterCustomOp());

    tflite::MicroInterpreter interpreter(
        model, resolver, tensor_arena, kTensorArenaSize, &error_reporter);

    if (interpreter.AllocateTensors() != kTfLiteOk) {
        printf("ERROR: AllocateTensors() failed\r\n");
        return false;
    }

    TfLiteTensor* input  = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);

    if (input->bytes != KWS_TEST_INPUT_SIZE) {
        printf("ERROR: input size mismatch (model=%u expected=%d)\r\n",
               (unsigned)input->bytes, KWS_TEST_INPUT_SIZE);
        return false;
    }

    memcpy(input->data.int8, kws_test_input_int8, KWS_TEST_INPUT_SIZE);

    uint64_t t0 = TimerMicros();
    TfLiteStatus status = interpreter.Invoke();
    uint64_t t1 = TimerMicros();

    if (status != kTfLiteOk) {
        printf("ERROR: Invoke() failed\r\n");
        return false;
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

    LedSet(Led::kUser, correct);

    printf("\r\n(repeating every 5 s — connect serial any time)\r\n");
    return correct;
}

}  // namespace
}  // namespace coralmicro

extern "C" void app_main(void* param) {
    (void)param;
    bool correct = coralmicro::run_kws();
    while (true) {
        vTaskDelay(pdMS_TO_TICKS(5000));
        printf("\r\n--- repeat ---\r\n");
        coralmicro::run_kws();
    }
}
