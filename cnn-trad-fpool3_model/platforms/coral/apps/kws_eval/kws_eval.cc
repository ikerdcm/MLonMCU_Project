// On-device (device-in-the-loop) accuracy for DS-CNN-L on the Edge TPU.
// Runs an embedded stratified test subset (ds_cnn_eval_set.h, int8 MFCC features
// quantized for ds_cnn_l_static_v2) through the Edge TPU and tallies accuracy.
// Compare to the HOST accuracy on the same subset (printed by make_eval_set.py)
// to isolate accelerator faithfulness. Model: the corrected v2 int8 model.

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

#include "ds_cnn_eval_set.h"

namespace coralmicro {
namespace {

const char* kModelPath = "/models/ds_cnn_l_static_v2_edgetpu.tflite";
constexpr int kNumClasses = 12;

constexpr int kTensorArenaSize = 256 * 1024;
STATIC_TENSOR_ARENA_IN_SDRAM(tensor_arena, kTensorArenaSize);

void run_eval() {
    printf("\r\n=== KWS on-device accuracy (Edge TPU, v2 int8) ===\r\n");

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
    if (!tpu_ctx) { printf("ERROR: failed to open Edge TPU\r\n"); return; }

    tflite::MicroErrorReporter error_reporter;
    tflite::MicroMutableOpResolver<1> resolver;
    resolver.AddCustom(kCustomOp, RegisterCustomOp());

    tflite::MicroInterpreter interpreter(
        model, resolver, tensor_arena, kTensorArenaSize, &error_reporter);
    if (interpreter.AllocateTensors() != kTfLiteOk) {
        printf("ERROR: AllocateTensors() failed\r\n"); return;
    }

    TfLiteTensor* input  = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);
    if ((int)input->bytes != EVAL_INPUT_SIZE) {
        printf("ERROR: input size mismatch (model=%u expected=%d)\r\n",
               (unsigned)input->bytes, EVAL_INPUT_SIZE);
        return;
    }

    printf("EVAL,event=start,total=%d\r\n", EVAL_N);
    int correct = 0;
    for (int i = 0; i < EVAL_N; i++) {
        memcpy(input->data.int8, &eval_features[i * EVAL_INPUT_SIZE], EVAL_INPUT_SIZE);
        if (interpreter.Invoke() != kTfLiteOk) {
            printf("ERROR: Invoke() failed at i=%d\r\n", i); return;
        }
        int   best_idx = 0;
        int8_t best_val = output->data.int8[0];
        for (int j = 1; j < kNumClasses; j++) {
            if (output->data.int8[j] > best_val) { best_val = output->data.int8[j]; best_idx = j; }
        }
        if (best_idx == (int)eval_labels[i]) correct++;
        if ((i % 60) == 59)
            printf("EVAL,progress=%d/%d,correct=%d\r\n", i + 1, EVAL_N, correct);
    }

    // acc_x100 = accuracy * 100 in basis points-ish (e.g. 9208 = 92.08%)
    int acc_x100 = (int)((10000LL * correct) / EVAL_N);
    printf("EVAL,event=done,correct=%d,total=%d,acc_x100=%d\r\n",
           correct, EVAL_N, acc_x100);
    LedSet(Led::kUser, true);
}

}  // namespace
}  // namespace coralmicro

extern "C" void app_main(void* param) {
    (void)param;
    while (true) {
        coralmicro::run_eval();
        vTaskDelay(pdMS_TO_TICKS(8000));  // re-run so a late serial reader still catches it
    }
}
