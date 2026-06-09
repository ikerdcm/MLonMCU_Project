// DS-CNN-L offline benchmark — 50 back-to-back inferences on Edge TPU.
// Outputs BENCH CSV compatible with the MAX78000 kws20_measure format.
// Model loaded from LittleFS; fixed "left" MFCC test vector used as input.

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

constexpr int kBenchRuns = 50;

const char* kModelPath = "/models/ds_cnn_l_static_v2_edgetpu.tflite";

const char* kLabels[] = {
    "down", "go", "left", "no", "off", "on",
    "right", "stop", "up", "yes", "silence", "unknown"
};
constexpr int kNumClasses = 12;

constexpr float kOutScale     = 0.00390625f;
constexpr int   kOutZeroPoint = -128;

constexpr int kTensorArenaSize = 256 * 1024;
STATIC_TENSOR_ARENA_IN_SDRAM(tensor_arena, kTensorArenaSize);

// RT1176 M7 core runs at 800 MHz
constexpr uint32_t kCpuHz = 800000000u;

void run_bench() {
    printf("\r\n==============================\r\n");
    printf("DS-CNN-L BENCHMARK (Coral Edge TPU)\r\n");
    printf("==============================\r\n");

    std::vector<uint8_t> model_data;
    if (!LfsReadFile(kModelPath, &model_data)) {
        printf("ERROR: failed to load model from %s\r\n", kModelPath);
        return;
    }

    const tflite::Model* model = tflite::GetModel(model_data.data());
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("ERROR: schema version mismatch\r\n");
        return;
    }

    auto tpu_ctx = EdgeTpuManager::GetSingleton()->OpenDevice();
    if (!tpu_ctx) {
        printf("ERROR: failed to open Edge TPU\r\n");
        return;
    }

    tflite::MicroErrorReporter error_reporter;
    tflite::MicroMutableOpResolver<1> resolver;
    resolver.AddCustom(kCustomOp, RegisterCustomOp());

    tflite::MicroInterpreter interpreter(
        model, resolver, tensor_arena, kTensorArenaSize, &error_reporter);

    if (interpreter.AllocateTensors() != kTfLiteOk) {
        printf("ERROR: AllocateTensors() failed\r\n");
        return;
    }

    TfLiteTensor* input  = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);

    if (input->bytes != KWS_TEST_INPUT_SIZE) {
        printf("ERROR: input size mismatch (model=%u expected=%d)\r\n",
               (unsigned)input->bytes, KWS_TEST_INPUT_SIZE);
        return;
    }

    printf("BENCH,event=model_info,mode=offline"
           ",cpu_hz=%lu"
           ",input_elems=%u"
           ",output_elems=%u"
           ",sample_rate_hz=16000"
           ",full_window_samples=16000"
           ",runs=%u"
           ",version=ds_cnn_l_edgetpu_int8\r\n",
           (unsigned long)kCpuHz,
           (unsigned)input->bytes,
           (unsigned)kNumClasses,
           (unsigned)kBenchRuns);

    printf("BENCH,event=acquisition,mode=offline"
           ",sample_rate_hz=16000"
           ",sample_count=16000"
           ",audio_window_ms=1000"
           ",feature_frames=49"
           ",feature_bins=10\r\n");

    uint32_t min_us = UINT32_MAX;
    uint32_t max_us = 0;
    uint64_t sum_us = 0;
    int last_pred   = -1;

    for (int run = 0; run < kBenchRuns; run++) {
        memcpy(input->data.int8, kws_test_input_int8, KWS_TEST_INPUT_SIZE);

        uint64_t t0 = TimerMicros();
        TfLiteStatus status = interpreter.Invoke();
        uint64_t t1 = TimerMicros();

        if (status != kTfLiteOk) {
            printf("ERROR: Invoke() failed at run=%d\r\n", run);
            return;
        }

        uint32_t elapsed_us = (uint32_t)(t1 - t0);
        uint32_t cycles = (uint32_t)(((uint64_t)elapsed_us * kCpuHz) / 1000000ULL);

        int best_idx   = 0;
        float best_val = -1e9f;
        for (int i = 0; i < kNumClasses; i++) {
            float v = (output->data.int8[i] - kOutZeroPoint) * kOutScale;
            if (v > best_val) { best_val = v; best_idx = i; }
        }
        last_pred = best_idx;

        if (elapsed_us < min_us) min_us = elapsed_us;
        if (elapsed_us > max_us) max_us = elapsed_us;
        sum_us += elapsed_us;

        printf("BENCH,event=inference,run=%d,mode=offline"
               ",cnn_us=%lu,cycles=%lu,pred_idx=%d\r\n",
               run,
               (unsigned long)elapsed_us,
               (unsigned long)cycles,
               best_idx);
    }

    uint32_t avg_us = (uint32_t)(sum_us / kBenchRuns);

    printf("BENCH,event=stats"
           ",min_us=%lu,max_us=%lu,avg_us=%lu"
           ",pred_idx=%d,pred_label=%s\r\n",
           (unsigned long)min_us,
           (unsigned long)max_us,
           (unsigned long)avg_us,
           last_pred,
           (last_pred >= 0 && last_pred < kNumClasses) ? kLabels[last_pred] : "?");

    printf("BENCH,event=done\r\n");

    LedSet(Led::kUser, last_pred == KWS_TEST_EXPECTED_IDX);
}

}  // namespace
}  // namespace coralmicro

extern "C" void app_main(void* param) {
    (void)param;
    coralmicro::run_bench();
    while (true) {
        vTaskDelay(pdMS_TO_TICKS(10000));

        printf("\r\n--- re-running benchmark ---\r\n");
        coralmicro::run_bench();
    }
}
