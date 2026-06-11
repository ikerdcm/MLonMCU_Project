// Device-in-the-loop accuracy eval — M7 CPU, fp32 (NO Edge TPU) — full pipeline.
// CPU counterpart of kws_eval_stream: reads the host-baked GSC audio subset from
// LittleFS (/eval/audio_set.bin), runs Coral's MFCC frontend (int8) + dequantize
// to float + the float DS-CNN-L on the Cortex-M7, and prints pred + the embedded
// true label for testbench.py.
//
//   /eval/audio_set.bin : uint32 N | N*16000 int16 LE audio | N uint8 labels
//   board prints         : BENCH,event=eval,idx=i,pred_idx=P,true_idx=T,cnn_us=U

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

#include "kws_mfcc.h"
#include "kws_mfcc_tables.h"   // MFCC_AUDIO_SAMPLES, MFCC_OUTPUT_ELEMS

namespace coralmicro {
namespace {

const char* kModelPath = "/models/ds_cnn_l_float.tflite";
const char* kEvalPath  = "/eval/audio_set.bin";
constexpr int kNumClasses = 12;
constexpr int kAudioSamples = MFCC_AUDIO_SAMPLES;   // 16000
constexpr int kMfccElems = MFCC_OUTPUT_ELEMS;       // 490
constexpr int kClipBytes = kAudioSamples * 2;       // 32000
constexpr size_t kEvalMaxBytes = 8u * 1024 * 1024;
// int8 MFCC (KwsMfccCompute) -> float model input: invert the EXACT quant
// KwsMfccCompute baked in (MFCC_INPUT_SCALE / _ZERO_POINT from kws_mfcc_tables.h).
// NB: the hardcoded 0.0368/-9 in kws_live_cpu are stale (old static-model quant)
// and produce all-"unknown" — use the macros so this stays in sync.
constexpr float kInScale     = MFCC_INPUT_SCALE;       // 0.5847...
constexpr int   kInZeroPoint = MFCC_INPUT_ZERO_POINT;  // 83
constexpr int kTensorArenaSize = 2 * 1024 * 1024;

STATIC_TENSOR_ARENA_IN_SDRAM(tensor_arena, kTensorArenaSize);
STATIC_TENSOR_ARENA_IN_SDRAM(g_eval_buf, kEvalMaxBytes);
static int8_t g_mfcc[kMfccElems];

void run_eval() {
    size_t got = LfsReadFile(kEvalPath, g_eval_buf, kEvalMaxBytes);
    if (got < 4) { printf("ERROR: missing %s (run make_eval_audio_set.py)\r\n", kEvalPath); return; }
    uint32_t N;
    memcpy(&N, g_eval_buf, 4);
    const uint8_t* audio  = g_eval_buf + 4;
    const uint8_t* labels = g_eval_buf + 4 + (size_t)N * kClipBytes;

    std::vector<uint8_t> model_data;
    if (!LfsReadFile(kModelPath, &model_data)) { printf("ERROR: model load\r\n"); return; }
    const tflite::Model* model = tflite::GetModel(model_data.data());
    if (model->version() != TFLITE_SCHEMA_VERSION) { printf("ERROR: schema\r\n"); return; }

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
    if (interpreter.AllocateTensors() != kTfLiteOk) { printf("ERROR: AllocateTensors\r\n"); return; }
    TfLiteTensor* input  = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);

    printf("BENCH,event=eval_ready,classes=12,total=%lu\r\n", (unsigned long)N);
    for (uint32_t i = 0; i < N; i++) {
        KwsMfccCompute(reinterpret_cast<const int16_t*>(audio + (size_t)i * kClipBytes), g_mfcc);
        for (int k = 0; k < kMfccElems; k++)
            input->data.f[k] = (g_mfcc[k] - kInZeroPoint) * kInScale;
        uint64_t t0 = TimerMicros();
        if (interpreter.Invoke() != kTfLiteOk) { printf("BENCH,event=eval_error,idx=%lu\r\n", (unsigned long)i); continue; }
        uint64_t us = TimerMicros() - t0;
        int best = 0; float bv = output->data.f[0];
        for (int j = 1; j < kNumClasses; j++)
            if (output->data.f[j] > bv) { bv = output->data.f[j]; best = j; }
        printf("BENCH,event=eval,idx=%lu,pred_idx=%d,true_idx=%d,cnn_us=%lu\r\n",
               (unsigned long)i, best, (int)labels[i], (unsigned long)us);
    }
    printf("BENCH,event=eval_done,total=%lu\r\n", (unsigned long)N);
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
