# DS-CNN-L KWS — Coral Dev Board Micro

Runs DS-CNN-L keyword spotting (INT8) on the [Coral Dev Board Micro](https://coral.ai/products/dev-board-micro/)
using the on-chip Edge TPU via the [coralmicro SDK](https://github.com/google-coral/coralmicro).

---

## Directory layout

```
coral/
├── README.md
├── model/                         # compiled Edge TPU model goes here
│   └── ds_cnn_l_edgetpu.tflite    # produced by scripts/compile_edgetpu.sh
├── scripts/
│   ├── compile_edgetpu.sh         # Step 1 – compile model for Edge TPU
│   └── flash.sh                   # Step 3 – flash firmware to board
└── app/
    ├── CMakeLists.txt             # coralmicro app definition
    ├── kws_coral.cc               # inference app (offline test, single run)
    └── ds_cnn_test_input_left_int8.h  # INT8-quantized "left" MFCC test vector
```

**Input model**: `../../models/ds_cnn_l.tflite`  
Already INT8-quantized (all-ops INT8 required by Edge TPU).  
Quantization: input scale=0.584702671, zero_point=83; output scale=1/256, zero_point=−128.

---

## Prerequisites

| Tool | Notes |
|------|-------|
| Docker | Required for `edgetpu_compiler` (runs in `gcr.io/google-coral/model-compiler`) |
| Python ≥ 3.8 | For `flashtool.py` |
| coralmicro SDK | Cloned into `coral/coralmicro/` (see Step 2) |
| CMake ≥ 3.13 | Standard build system |
| `arm-none-eabi-gcc` | Toolchain, installed by coralmicro bootstrap |

---

## Step 1 — Create static-shape INT8 model

The original `ds_cnn_l.tflite` has dynamic batch in its shape_signature (batch=-1).
The Edge TPU compiler requires fully static shapes, so we re-convert from the
Keras SavedModel with a fixed `[1, 49, 10, 1]` input signature:

```bash
# Without real calibration data (uses random data — structure correct, minor accuracy loss):
python3 scripts/convert_static_int8.py

# With real calibration data (best accuracy):
python3 scripts/convert_static_int8.py --data_dir ~/data/speech_commands_v002
```

Output: `../../models/ds_cnn_l_static.tflite`

## Step 2 — Compile for Edge TPU

```bash
chmod +x scripts/compile_edgetpu.sh
./scripts/compile_edgetpu.sh ../../models/ds_cnn_l_static.tflite
```

This runs `edgetpu_compiler` on the static model and writes
`model/ds_cnn_l_static_edgetpu.tflite`.

Expected output — all 12 ops mapped to the Edge TPU, 0 CPU fallbacks:
```
AVERAGE_POOL_2D   1   Mapped to Edge TPU
DEPTHWISE_CONV_2D 4   Mapped to Edge TPU
CONV_2D           5   Mapped to Edge TPU
SOFTMAX           1   Mapped to Edge TPU
FULLY_CONNECTED   1   Mapped to Edge TPU
```

---

## Step 3 — Clone coralmicro SDK

```bash
# From this coral/ directory:
git clone --recurse-submodules https://github.com/google-coral/coralmicro coralmicro
cd coralmicro
bash setup.sh     # installs toolchain, pulls submodules (~5 min)
```

---

## Step 4 — Build firmware

```bash
cd coralmicro
mkdir -p ../build && cd ../build

cmake ../coralmicro \
    -DCMAKE_BUILD_TYPE=Release \
    -DAPP_SOURCE_DIR=../app \
    -DAPP_NAME=kws_coral

cmake --build . -j$(nproc) --target kws_coral
```

The compiled ELF will be at `build/kws_coral`.

---

## Step 5 — Flash firmware

Connect the Coral Dev Board Micro via USB, then:

```bash
chmod +x scripts/flash.sh
./scripts/flash.sh
```

Or manually:

```bash
python3 coralmicro/scripts/flashtool.py \
    --build_dir build \
    --elf_path  build/kws_coral \
    --nodata
```

---

## Step 6 — Read output

Connect a serial terminal at 115200 baud (e.g. `screen /dev/ttyACM0 115200` or
PuTTY on Windows). After reset you should see:

```
=== KWS DS-CNN-L on Coral Dev Board Micro ===
Model size: XXXXX bytes
Input  type: 9  dims: [1,49,10,1]
Output type: 9  dims: [1,12]

Inference time: XXXX us (X.XX ms)

Scores:
  [ 0] down    : 0.0000
  [ 1] go      : 0.0000
  [ 2] left    : 0.9961
  ...

Prediction: "left" (idx=2, score=0.9961)
Expected  : "left" (idx=2)
Result    : PASS
```

The green user LED lights on PASS.

---

## Expected performance

| Metric | Value |
|--------|-------|
| Model MACs | 3 937 717 |
| Model params (INT8) | ~460 KB |
| Edge TPU inference | < 5 ms (typical) |

---

## Notes

- The `app_main()` entry point is called by the coralmicro FreeRTOS scheduler.
- The model is embedded in flash via the `target_add_resource` CMake macro.
- To run live microphone inference, replace the test vector with PDM audio capture
  using `libs/audio/audio_service.h` from the coralmicro SDK.
- edgetpu_compiler requires x86-64 Linux host (Docker image handles this automatically).
