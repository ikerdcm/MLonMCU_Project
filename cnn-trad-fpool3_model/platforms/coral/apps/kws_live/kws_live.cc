// Live keyword spotting on Coral Dev Board Micro.
// Captures mic audio at 16 kHz, computes MFCC, runs DS-CNN-L on Edge TPU.
// Prints result over USB-CDC (screen /dev/ttyACM0 115200).

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "kws_mfcc.h"
#include "kws_mfcc_tables.h"

#include "libs/audio/audio_service.h"
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

namespace coralmicro {
namespace {

const char* kModelPath = "/models/ds_cnn_l_static_edgetpu.tflite";

// DMA buffers must be global (DMA controller accesses them directly).
constexpr int kNumDmaBuffers   = 4;
constexpr int kDmaBufferSizeMs = 20;  // 20 ms → 320 samples per DMA chunk
constexpr int kDmaTotalSamples = kNumDmaBuffers * kDmaBufferSizeMs * 16;  // 1280
AudioDriverBuffers<kNumDmaBuffers, kDmaTotalSamples> g_audio_buffers;

const char* kLabels[] = {
    "down", "go", "left", "no", "off", "on",
    "right", "stop", "up", "yes", "silence", "unknown"
};
constexpr int kNumClasses = 12;
constexpr float kConfidenceThreshold = 0.5f;

constexpr float kOutScale     = 0.00390625f;
constexpr int   kOutZeroPoint = -128;

constexpr int kTensorArenaSize = 256 * 1024;
STATIC_TENSOR_ARENA_IN_SDRAM(tensor_arena, kTensorArenaSize);

// Audio ring buffer — holds exactly 1 second of 16 kHz audio.
constexpr int kAudioSamples = MFCC_AUDIO_SAMPLES;  // 16000
static int16_t g_audio_ring[kAudioSamples];
static volatile int g_ring_write_pos = 0;

// Accumulate new samples (int32 from DMA → int16 by dropping 16 LSBs).
// Called every 20 ms from AudioReader polling loop.
static void AppendSamples(const int32_t* buf, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        g_audio_ring[g_ring_write_pos] = (int16_t)(buf[i] >> 16);
        g_ring_write_pos = (g_ring_write_pos + 1) % kAudioSamples;
    }
}

// Copy ring buffer into a flat array in chronological order.
static void SnapshotRing(int16_t* dst) {
    int pos = g_ring_write_pos;  // oldest sample
    int tail = kAudioSamples - pos;
    memcpy(dst,       g_audio_ring + pos, tail * sizeof(int16_t));
    memcpy(dst + tail, g_audio_ring,       pos  * sizeof(int16_t));
}

}  // namespace
}  // namespace coralmicro

extern "C" void app_main(void* param) {
    using namespace coralmicro;
    (void)param;

    printf("\r\n=== KWS Live — DS-CNN-L on Edge TPU ===\r\n");

    // --- Load model ---
    std::vector<uint8_t> model_data;
    if (!LfsReadFile(kModelPath, &model_data)) {
        printf("ERROR: model not found at %s\r\n", kModelPath);
        vTaskSuspend(nullptr);
    }
    printf("Model loaded: %u bytes\r\n", (unsigned)model_data.size());

    // --- Edge TPU + interpreter ---
    auto tpu_ctx = EdgeTpuManager::GetSingleton()->OpenDevice();
    if (!tpu_ctx) { printf("ERROR: Edge TPU open failed\r\n"); vTaskSuspend(nullptr); }

    tflite::MicroErrorReporter error_reporter;
    tflite::MicroMutableOpResolver<1> resolver;
    resolver.AddCustom(kCustomOp, RegisterCustomOp());

    const tflite::Model* model = tflite::GetModel(model_data.data());
    tflite::MicroInterpreter interpreter(
        model, resolver, tensor_arena, kTensorArenaSize, &error_reporter);

    if (interpreter.AllocateTensors() != kTfLiteOk) {
        printf("ERROR: AllocateTensors failed\r\n"); vTaskSuspend(nullptr);
    }

    TfLiteTensor* input  = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);
    printf("Input tensor: %u bytes\r\n", (unsigned)input->bytes);

    // --- Microphone setup ---
    AudioDriver audio_driver(g_audio_buffers);
    const AudioDriverConfig audio_cfg{
        AudioSampleRate::k16000_Hz,
        kNumDmaBuffers,
        kDmaBufferSizeMs
    };
    AudioReader reader(&audio_driver, audio_cfg);

    // Drop first 150 ms to let mic settle
    reader.Drop(MsToSamples(AudioSampleRate::k16000_Hz, 150));
    printf("Microphone ready. Listening...\r\n\r\n");

    // --- Inference loop ---
    // Run inference every 500 ms (= 8000 new samples).
    constexpr int kInferenceStrideMs = 500;
    constexpr int kInferenceSrideSamples =
        kInferenceStrideMs * 16;  // 8000 samples

    int samples_since_last_inference = kInferenceSrideSamples;  // run immediately
    static int8_t  mfcc_buf[MFCC_OUTPUT_ELEMS];
    static int16_t audio_snap[kAudioSamples];

    while (true) {
        size_t got = reader.FillBuffer();
        const auto& buf32 = reader.Buffer();
        AppendSamples(buf32.data(), got);
        samples_since_last_inference += (int)got;

        if (samples_since_last_inference < kInferenceSrideSamples) continue;
        samples_since_last_inference = 0;

        // Snapshot ring buffer and compute MFCC
        SnapshotRing(audio_snap);

        uint64_t t0 = TimerMicros();
        KwsMfccCompute(audio_snap, mfcc_buf);
        uint64_t t1 = TimerMicros();

        memcpy(input->data.int8, mfcc_buf, MFCC_OUTPUT_ELEMS);
        TfLiteStatus st = interpreter.Invoke();
        uint64_t t2 = TimerMicros();

        if (st != kTfLiteOk) { printf("ERROR: Invoke failed\r\n"); continue; }

        int   best_idx   = 0;
        float best_score = -1e9f;
        for (int i = 0; i < kNumClasses; ++i) {
            float score = (output->data.int8[i] - kOutZeroPoint) * kOutScale;
            if (score > best_score) { best_score = score; best_idx = i; }
        }

        uint32_t mfcc_us   = (uint32_t)(t1 - t0);
        uint32_t invoke_us = (uint32_t)(t2 - t1);

        bool confident = (best_score >= kConfidenceThreshold);
        if (confident) {
            printf(">>> %-8s (%.0f%%)  mfcc=%lu us  infer=%lu us\r\n",
                   kLabels[best_idx], best_score * 100.0f,
                   (unsigned long)mfcc_us, (unsigned long)invoke_us);
            LedSet(Led::kUser, true);
        } else {
            printf("    %-8s (%.0f%%)  mfcc=%lu us  infer=%lu us\r\n",
                   kLabels[best_idx], best_score * 100.0f,
                   (unsigned long)mfcc_us, (unsigned long)invoke_us);
            LedSet(Led::kUser, false);
        }
    }
}
