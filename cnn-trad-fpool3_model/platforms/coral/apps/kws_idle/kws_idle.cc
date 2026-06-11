// DS-CNN-L duty-cycled power firmware — ONE offline inference every kPeriodMs,
// with the M7 in low-power sleep (WFI) between inferences. INT8 on the Edge TPU
// (config: int8-accel, v1 = 6-block ds_cnn_l_static_v2). Purpose: power-meter
// measurement of the active-vs-idle duty cycle (a short Edge-TPU spike, then
// ~5 s WFI baseline). TPU counterpart of kws_idle_cpu.
//
// "Sleep" here = FreeRTOS tickless idle (configUSE_TICKLESS_IDLE=2): while this
// task is blocked in vTaskDelay and no other task is ready, the port suppresses
// ticks and the M7 core enters __WFI() (clock-gated; PLLs stay up — USB/serial
// survive so we keep reporting). coralmicro exposes no deeper STOP/SetPoint mode.
//
// Model: /models/ds_cnn_l_static_v2_edgetpu.tflite. Offline input: the baked
//        "left" int8 MFCC test vector (ds_cnn_test_input_left_int8.h) — no mic.
// Output (one block per cycle, for aligning the power trace):
//   BENCH,event=inference,cycle=<n>,mode=offline,cnn_us=<us>,pred_idx=<i>
//   BENCH,event=sleep,cycle=<n>,sleep_ms=<period>

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

// One inference, then sleep this long. Change here to retune the duty cycle.
constexpr uint32_t kPeriodMs = 5000;
// Light the user LED during the active inference window so the power trace has a
// visual/electrical marker. Set false for the cleanest active-current reading.
constexpr bool kMarkerLed = false;

// Baked under a fixed LittleFS name; the SOURCE network is chosen at build time
// via -DKWS_IDLE_MODEL / -DKWS_IDLE_VERSION (see CMakeLists.txt + build script).
const char* kModelPath = "/models/idle_model_edgetpu.tflite";
#ifndef KWS_IDLE_MODEL_NAME
#define KWS_IDLE_MODEL_NAME "unknown"
#endif
#ifndef KWS_IDLE_VERSION_NAME
#define KWS_IDLE_VERSION_NAME "unknown"
#endif
const char* kModelName = KWS_IDLE_MODEL_NAME;
const char* kVersion   = KWS_IDLE_VERSION_NAME;

const char* kLabels[] = {
    "down", "go", "left", "no", "off", "on",
    "right", "stop", "up", "yes", "silence", "unknown"
};
constexpr int kNumClasses = 12;

constexpr int kTensorArenaSize = 256 * 1024;
STATIC_TENSOR_ARENA_IN_SDRAM(tensor_arena, kTensorArenaSize);

constexpr uint32_t kCpuHz = 800000000u;   // RT1176 M7

void run_idle_loop() {
    printf("\r\n==============================\r\n");
    printf("DS-CNN-L DUTY-CYCLE POWER (Coral Edge TPU, INT8 v1, offline)\r\n");
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
    if (!tpu_ctx) { printf("ERROR: failed to open Edge TPU\r\n"); return; }

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
           ",cpu_hz=%lu,input_elems=%u,output_elems=%u"
           ",period_ms=%lu,sleep=wfi_tickless"
           ",version=%s,model=%s,app=kws_idle\r\n",
           (unsigned long)kCpuHz, (unsigned)input->bytes,
           (unsigned)kNumClasses, (unsigned long)kPeriodMs,
           kVersion, kModelName);

    // Set up the input once — offline, the vector never changes.
    memcpy(input->data.int8, kws_test_input_int8, KWS_TEST_INPUT_SIZE);

    uint32_t cycle = 0;
    while (true) {
        if (kMarkerLed) LedSet(Led::kUser, true);
        uint64_t t0 = TimerMicros();
        TfLiteStatus status = interpreter.Invoke();
        uint64_t t1 = TimerMicros();
        if (kMarkerLed) LedSet(Led::kUser, false);

        if (status != kTfLiteOk) {
            printf("ERROR: Invoke() failed at cycle=%lu\r\n", (unsigned long)cycle);
            return;
        }

        int best_idx = 0;
        int8_t best_val = output->data.int8[0];
        for (int i = 1; i < kNumClasses; i++)
            if (output->data.int8[i] > best_val) { best_val = output->data.int8[i]; best_idx = i; }

        printf("BENCH,event=inference,cycle=%lu,mode=offline,cnn_us=%lu,pred_idx=%d,pred_label=%s\r\n",
               (unsigned long)cycle, (unsigned long)(t1 - t0), best_idx,
               (best_idx >= 0 && best_idx < kNumClasses) ? kLabels[best_idx] : "?");
        printf("BENCH,event=sleep,cycle=%lu,sleep_ms=%lu\r\n",
               (unsigned long)cycle, (unsigned long)kPeriodMs);

        // Blocked here -> tickless idle -> M7 enters WFI for ~kPeriodMs.
        vTaskDelay(pdMS_TO_TICKS(kPeriodMs));
        cycle++;
    }
}

}  // namespace
}  // namespace coralmicro

extern "C" void app_main(void* param) {
    (void)param;
    coralmicro::run_idle_loop();
    while (true) vTaskDelay(pdMS_TO_TICKS(10000));  // only reached on init error
}
