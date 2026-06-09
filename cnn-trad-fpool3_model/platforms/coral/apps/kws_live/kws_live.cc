// Live keyword spotting on Coral Dev Board Micro.
// Captures mic audio at 16 kHz, computes MFCC, runs DS-CNN-L on Edge TPU.
// Prints result over USB-CDC (screen /dev/ttyACM0 115200).

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
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

const char* kModelPath = "/models/ds_cnn_l_static_v2_edgetpu.tflite";

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

}  // namespace
}  // namespace coralmicro

// Report gating — flash_all.sh sets these at flash time (dashboard dropdown).
// idx 10 = silence, 11 = unknown in the 12-class label order.
#define KWS_REPORT_UNKNOWN 0
#define KWS_REPORT_SILENCE 0
#define KWS_REPORT_OK(idx) (((idx) != 10 || (KWS_REPORT_SILENCE)) && \
                            ((idx) != 11 || (KWS_REPORT_UNKNOWN)))

namespace coralmicro {
namespace {

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

    // --- Adaptive voice-activity trigger (shared design across all 3 boards) ---
    // Threshold is relative to a tracked noise floor, so it self-normalizes to
    // this board's mic scale and follows ambient noise. EMA tracker = a few
    // flops per chunk (negligible power). thr_high/thr_low use shared unitless K.
    constexpr float kVadKHigh           = 4.0f;  // trigger when L >= nf*K_HIGH
    constexpr float kVadKLow            = 2.0f;  // end when L <  nf*K_LOW
    constexpr int   kVadTrigConsec      = 2;     // consecutive high chunks to fire
    constexpr int   kVadWarmupChunks    = 25;    // ~0.5 s @ 20 ms chunks to seed nf
    constexpr int   kVadSilenceChunks   = 15;    // ~0.3 s below low ends an utterance
    constexpr int   kVadPostTrigMaxChunks = 50;  // ~1 s safety cap (= ring length)

    float nf = 0.0f, nf_seed = 0.0f; // adaptive noise floor + startup-ambient floor
    int   warmup_n = 0;
    int   vad_state = 0;             // 0 = idle, 1 = collecting an utterance
    int   high_run = 0, sil_run = 0, post_chunks = 0;
    int   level_div = 0;

    static int8_t  mfcc_buf[MFCC_OUTPUT_ELEMS];
    static int16_t audio_snap[kAudioSamples];

    while (true) {
        size_t got = reader.FillBuffer();
        const auto& buf32 = reader.Buffer();
        AppendSamples(buf32.data(), got);
        if (!got) continue;

        // Per-chunk level L (RMS) — drives both the plot and the VAD.
        uint64_t acc = 0;
        for (size_t i = 0; i < got; ++i) {
            int32_t s = (int16_t)(buf32[i] >> 16);
            acc += (uint32_t)(s * s);
        }
        float L = sqrtf((float)acc / (float)got);

        // Noise floor: running mean during warmup, then asymmetric EMA while
        // idle (rise slow, fall fast) so speech bursts don't inflate it.
        if (warmup_n < kVadWarmupChunks) {
            nf += (L - nf) / (float)(warmup_n + 1);
            if (++warmup_n == kVadWarmupChunks) nf_seed = (nf < 1.0f) ? 1.0f : nf;
        } else if (vad_state == 0) {
            nf += (L > nf ? (1.0f / 64.0f) : (1.0f / 8.0f)) * (L - nf);
        }
        if (nf < 1.0f) nf = 1.0f;
        if (nf < nf_seed) nf = nf_seed;   // floor at startup ambient: never collapse
        float thr_high = nf * kVadKHigh;
        float thr_low  = nf * kVadKLow;

        // Normalized level (x100 of noise floor) + unified threshold for the plot.
        if (++level_div & 1) {
            printf("BENCH,event=level,rms=%lu,thr=%lu\r\n",
                   (unsigned long)(100.0f * L / nf),
                   (unsigned long)(100.0f * kVadKHigh));
        }

        if (warmup_n < kVadWarmupChunks) continue;  // no detection until seeded

        // VAD state machine.
        if (vad_state == 0) {
            high_run = (L >= thr_high) ? (high_run + 1) : 0;
            if (high_run >= kVadTrigConsec) {
                vad_state = 1; sil_run = 0; post_chunks = 0; high_run = 0;
            }
            continue;
        }
        // Collecting: wait for end-of-utterance (sustained quiet) or safety cap.
        post_chunks++;
        sil_run = (L < thr_low) ? (sil_run + 1) : 0;
        if (sil_run < kVadSilenceChunks && post_chunks < kVadPostTrigMaxChunks) {
            continue;
        }
        vad_state = 0;  // utterance done → run one inference on the 1 s window

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

        // Decision view: per-class scores 0..100 for the dashboard bar chart.
        {
            char sb[256];
            int n = snprintf(sb, sizeof(sb), "BENCH,event=scores,s=");
            for (int i = 0; i < kNumClasses; ++i) {
                int s = (int)((output->data.int8[i] - kOutZeroPoint) * kOutScale * 100.0f + 0.5f);
                if (s < 0) s = 0; else if (s > 100) s = 100;
                n += snprintf(sb + n, sizeof(sb) - n, "%s%d", i ? ";" : "", s);
            }
            printf("%s\r\n", sb);
        }

        // MFCC the model saw (49 frames × 10 bins) for the dashboard spectrogram.
        // Emitted as many tiny writes (one big USB-CDC write of ~2.5 KB gets
        // truncated); the host reassembles the line at the trailing newline.
        printf("BENCH,event=mfcc,w=10,h=49,d=");
        for (int i = 0; i < MFCC_OUTPUT_ELEMS; ++i)
            printf("%s%d", i ? ";" : "", (int)mfcc_buf[i]);
        printf("\r\n");

        // Only report a real keyword detection: confident AND not the
        // "silence"/"unknown" catch-all classes. Otherwise stay quiet so the
        // stream is event-driven like the STM32/MAX boards (no idle spam).
        const char* lbl = kLabels[best_idx];
        bool confident  = (best_score >= kConfidenceThreshold);
        if (confident && KWS_REPORT_OK(best_idx)) {
            printf(">>> %-8s (%.0f%%)  mfcc=%lu us  infer=%lu us\r\n",
                   lbl, best_score * 100.0f,
                   (unsigned long)mfcc_us, (unsigned long)invoke_us);
            LedSet(Led::kUser, true);
        } else {
            LedSet(Led::kUser, false);
        }
    }
}
