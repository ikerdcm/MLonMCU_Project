// DS-CNN-L offline benchmark — 50 back-to-back inferences on Edge TPU.
// Outputs BENCH CSV compatible with the MAX78000 kws20_measure format.
// Model loaded from LittleFS; fixed "left" MFCC test vector used as input.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "ds_cnn_test_input_left_int8.h"
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

constexpr int kBenchRuns = 1;

const char* kModelPath = "/models/ds_cnn_l_static_edgetpu.tflite";

const char* kLabels[] = {"down",  "go",   "left", "no",  "off",     "on",
                         "right", "stop", "up",   "yes", "silence", "unknown"};
constexpr int kNumClasses = 12;

constexpr float kOutScale = 0.00390625f;
constexpr int kOutZeroPoint = -128;

constexpr int kTensorArenaSize = 256 * 1024;
STATIC_TENSOR_ARENA_IN_SDRAM(tensor_arena, kTensorArenaSize);

// RT1176 M7 core runs at 800 MHz
constexpr uint32_t kCpuHz = 800000000u;

// Pre-sleep and post-sleep durations for power-trace isolation.
// The Nordic PPK2 sees: flat_idle → inference_pulse → flat_idle → nothing
constexpr uint32_t kSleepPreMs  = 500;
constexpr uint32_t kSleepPostMs = 500;

}  // namespace
}  // namespace coralmicro

extern "C" void app_main(void* param) {
  using namespace coralmicro;
  (void)param;

  // ── Phase 1: Setup (outside measurement window) ──────────────────────────
  printf("\r\n=== single_inference setup ===\r\n");

  std::vector<uint8_t> model_data;
  if (!LfsReadFile(kModelPath, &model_data)) {
    printf("ERROR: failed to load model from %s\r\n", kModelPath);
    vTaskSuspend(nullptr);
  }

  const tflite::Model* model = tflite::GetModel(model_data.data());
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    printf("ERROR: schema version mismatch\r\n");
    vTaskSuspend(nullptr);
  }

  auto tpu_ctx = EdgeTpuManager::GetSingleton()->OpenDevice();
  if (!tpu_ctx) {
    printf("ERROR: failed to open Edge TPU\r\n");
    vTaskSuspend(nullptr);
  }

  tflite::MicroErrorReporter error_reporter;
  tflite::MicroMutableOpResolver<1> resolver;
  resolver.AddCustom(kCustomOp, RegisterCustomOp());

  tflite::MicroInterpreter interpreter(model, resolver, tensor_arena,
                                       kTensorArenaSize, &error_reporter);

  if (interpreter.AllocateTensors() != kTfLiteOk) {
    printf("ERROR: AllocateTensors() failed\r\n");
    vTaskSuspend(nullptr);
  }

  TfLiteTensor* input  = interpreter.input(0);
  TfLiteTensor* output = interpreter.output(0);

  if (input->bytes != KWS_TEST_INPUT_SIZE) {
    printf("ERROR: input size mismatch (model=%u expected=%d)\r\n",
           (unsigned)input->bytes, KWS_TEST_INPUT_SIZE);
    vTaskSuspend(nullptr);
  }

  memcpy(input->data.int8, kws_test_input_int8, KWS_TEST_INPUT_SIZE);



  vTaskDelay(pdMS_TO_TICKS(kSleepPreMs));

  while (true)
  {
    vTaskDelay(pdMS_TO_TICKS(5000));  
    TfLiteStatus status = interpreter.Invoke();
  }
  

  // Suspend forever — power trace shows exactly one pulse
  vTaskSuspend(nullptr);
}
