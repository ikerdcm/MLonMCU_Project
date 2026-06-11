// Device-in-the-loop accuracy eval — Edge TPU, v2 INT8 — full pipeline.
// Coral's console drops host->device USB bytes, so instead of live streaming the
// host bakes the GSC test subset as RAW AUDIO into LittleFS (/eval/audio_set.bin,
// via make_eval_audio_set.py). This app reads it, runs Coral's OWN MFCC frontend
// (KwsMfccCompute) + the Edge TPU on each clip, and prints pred + the embedded
// true label, for testbench.py to tally (accuracy + confusion + latency).
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
#include "libs/tpu/edgetpu_manager.h"
#include "libs/tpu/edgetpu_op.h"
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

// Baked under a fixed LittleFS name; the SOURCE model is chosen at build time
// via -DKWS_EVAL_STREAM_MODEL=<basename> (see CMakeLists.txt). This lets the same
// firmware eval the v2 6-block or any pruned variant without touching this file.
const char* kModelPath = "/models/eval_model_edgetpu.tflite";
// Which source model was baked (set by -DKWS_EVAL_STREAM_MODEL at build time), so
// the board self-reports its identity in eval_ready. Falls back if not defined.
#ifndef KWS_EVAL_STREAM_MODEL_NAME
#define KWS_EVAL_STREAM_MODEL_NAME "unknown"
#endif
#ifndef KWS_EVAL_STREAM_VERSION_NAME
#define KWS_EVAL_STREAM_VERSION_NAME "unknown"
#endif
const char* kModelName = KWS_EVAL_STREAM_MODEL_NAME;
const char* kVersion   = KWS_EVAL_STREAM_VERSION_NAME;
const char* kEvalPath  = "/eval/audio_set.bin";
constexpr int kNumClasses = 12;
constexpr int kAudioSamples = MFCC_AUDIO_SAMPLES;   // 16000
constexpr int kMfccElems = MFCC_OUTPUT_ELEMS;       // 490
constexpr int kClipBytes = kAudioSamples * 2;       // 32000
constexpr size_t kEvalMaxBytes = 8u * 1024 * 1024;  // ~250 clips
constexpr int kTensorArenaSize = 256 * 1024;

STATIC_TENSOR_ARENA_IN_SDRAM(tensor_arena, kTensorArenaSize);
STATIC_TENSOR_ARENA_IN_SDRAM(g_eval_buf, kEvalMaxBytes);   // baked audio set
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
    auto tpu_ctx = EdgeTpuManager::GetSingleton()->OpenDevice();
    if (!tpu_ctx) { printf("ERROR: TPU\r\n"); return; }

    tflite::MicroErrorReporter error_reporter;
    tflite::MicroMutableOpResolver<1> resolver;
    resolver.AddCustom(kCustomOp, RegisterCustomOp());
    tflite::MicroInterpreter interpreter(
        model, resolver, tensor_arena, kTensorArenaSize, &error_reporter);
    if (interpreter.AllocateTensors() != kTfLiteOk) { printf("ERROR: AllocateTensors\r\n"); return; }
    TfLiteTensor* input  = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);

    printf("BENCH,event=eval_ready,classes=12,total=%lu,version=%s,model=%s\r\n",
           (unsigned long)N, kVersion, kModelName);
    for (uint32_t i = 0; i < N; i++) {
        KwsMfccCompute(reinterpret_cast<const int16_t*>(audio + (size_t)i * kClipBytes), g_mfcc);
        memcpy(input->data.int8, g_mfcc, kMfccElems);
        uint64_t t0 = TimerMicros();
        if (interpreter.Invoke() != kTfLiteOk) { printf("BENCH,event=eval_error,idx=%lu\r\n", (unsigned long)i); continue; }
        uint64_t us = TimerMicros() - t0;
        int best = 0; int8_t bv = output->data.int8[0];
        for (int j = 1; j < kNumClasses; j++)
            if (output->data.int8[j] > bv) { bv = output->data.int8[j]; best = j; }
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
        vTaskDelay(pdMS_TO_TICKS(8000));  // re-run so a late reader catches the pass
    }
}
