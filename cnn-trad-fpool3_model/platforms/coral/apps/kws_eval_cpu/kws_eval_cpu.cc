// On-device accuracy for the FP32 model on the M7 CPU (config: fp32-cpu).
// Counterpart of kws_eval (Edge-TPU). The CPU build's float kernels leave no
// room to embed the test subset in internal flash, so it loads the subset from
// LittleFS (/eval/eval_set.bin: uint32 N + N*490 int8 features + N uint8 labels;
// features are int8 with the v2 input quant, dequantized to float here).

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "libs/base/filesystem.h"
#include "libs/base/led.h"
#include "libs/base/timer.h"
#include "libs/tensorflow/utils.h"
#include "third_party/freertos_kernel/include/FreeRTOS.h"
#include "third_party/freertos_kernel/include/task.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_error_reporter.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_interpreter.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "third_party/tflite-micro/tensorflow/lite/schema/schema_generated.h"

namespace coralmicro {
namespace {

const char* kModelPath = "/models/ds_cnn_l_float.tflite";
const char* kEvalPath  = "/eval/eval_set.bin";
constexpr int kNumClasses    = 12;
constexpr int kInputElems    = 490;

// embedded features are int8 (v2 input quant); dequantize to MFCC float.
constexpr float kInScale     = 0.5847026705741882f;
constexpr int   kInZeroPoint = 83;

constexpr int kTensorArenaSize = 2 * 1024 * 1024;  // FP32 activations
STATIC_TENSOR_ARENA_IN_SDRAM(tensor_arena, kTensorArenaSize);

void run_eval() {
    printf("\r\n=== KWS on-device accuracy (M7 CPU, FP32) ===\r\n");

    std::vector<uint8_t> model_data, eval_data;
    if (!LfsReadFile(kModelPath, &model_data)) {
        printf("ERROR: failed to load model from %s\r\n", kModelPath); return;
    }
    if (!LfsReadFile(kEvalPath, &eval_data) || eval_data.size() < 4) {
        printf("ERROR: failed to load eval set from %s\r\n", kEvalPath); return;
    }
    uint32_t N = 0;
    memcpy(&N, eval_data.data(), 4);
    const signed char*   feats  = reinterpret_cast<const signed char*>(eval_data.data() + 4);
    const unsigned char* labels = reinterpret_cast<const unsigned char*>(
        eval_data.data() + 4 + (size_t)N * kInputElems);

    const tflite::Model* model = tflite::GetModel(model_data.data());
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("ERROR: schema version mismatch\r\n"); return;
    }

    tflite::MicroErrorReporter error_reporter;
    tflite::MicroMutableOpResolver<6> resolver;
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddAveragePool2D();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddReshape();

    tflite::MicroInterpreter interpreter(
        model, resolver, tensor_arena, kTensorArenaSize, &error_reporter);
    if (interpreter.AllocateTensors() != kTfLiteOk) {
        printf("ERROR: AllocateTensors() failed\r\n"); return;
    }
    TfLiteTensor* input  = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);
    if ((int)input->bytes != kInputElems * (int)sizeof(float)) {
        printf("ERROR: input size mismatch (model=%u)\r\n", (unsigned)input->bytes); return;
    }

    printf("EVAL,event=start,total=%lu\r\n", (unsigned long)N);
    int correct = 0;
    for (uint32_t i = 0; i < N; i++) {
        const signed char* q = &feats[(size_t)i * kInputElems];
        for (int k = 0; k < kInputElems; k++)
            input->data.f[k] = (q[k] - kInZeroPoint) * kInScale;
        if (interpreter.Invoke() != kTfLiteOk) {
            printf("ERROR: Invoke() failed at i=%lu\r\n", (unsigned long)i); return;
        }
        int   best_idx = 0;
        float best_val = output->data.f[0];
        for (int j = 1; j < kNumClasses; j++)
            if (output->data.f[j] > best_val) { best_val = output->data.f[j]; best_idx = j; }
        if (best_idx == (int)labels[i]) correct++;
        if ((i % 60) == 59)
            printf("EVAL,progress=%lu/%lu,correct=%d\r\n",
                   (unsigned long)(i + 1), (unsigned long)N, correct);
    }
    int acc_x100 = (int)((10000LL * correct) / (long long)N);
    printf("EVAL,event=done,correct=%d,total=%lu,acc_x100=%d\r\n",
           correct, (unsigned long)N, acc_x100);
    LedSet(Led::kUser, true);
}

}  // namespace
}  // namespace coralmicro

extern "C" void app_main(void* param) {
    (void)param;
    while (true) {
        coralmicro::run_eval();
        vTaskDelay(pdMS_TO_TICKS(8000));
    }
}
