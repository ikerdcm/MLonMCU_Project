# DS-CNN-L KWS — Coral Dev Board Micro

Runs DS-CNN-L keyword spotting (INT8) on the [Coral Dev Board Micro](https://coral.ai/products/dev-board-micro/)
using the on-chip Edge TPU via the [coralmicro SDK](https://github.com/google-coral/coralmicro).

---

## Directory layout

```text
coral/
├── README.md
├── model/
│   └── ds_cnn_l_static_edgetpu.tflite   # compiled Edge TPU model (committed)
├── scripts/                             # model conversion + build/flash helpers
│   ├── setup.sh                         # install edgetpu_compiler (Linux/apt)
│   ├── fix_static_shape.py              # make static model by byte-patching (keeps quantization)
│   ├── convert_static_int8.py          # make static model by re-quantizing from SavedModel
│   ├── compile_edgetpu.sh              # compile a static .tflite for the Edge TPU
│   ├── build_and_flash_bench.sh        # build+flash kws_bench
│   ├── build_and_flash_live.sh         # build+flash kws_live
│   ├── build_and_flash_single_inference.sh
│   ├── flash.sh                        # low-level flashtool wrapper
│   └── sync_coral_apps.sh              # mirror coralmicro/apps → apps/ (legacy helper)
├── apps/                                # firmware apps (BUILD SOURCE — see Build section)
│   ├── CMakeLists.txt                   # add_subdirectory for each app
│   ├── common/
│   │   └── ds_cnn_test_input_left_int8.h  # shared INT8 "left" MFCC test vector
│   ├── kws_coral/         # single offline inference (one "left" vector)
│   ├── single_inference/  # single offline inference (verbose)
│   ├── kws_bench/         # 50-run offline benchmark, BENCH CSV output
│   └── kws_live/          # live mic → MFCC → Edge TPU, prints detections
├── tools/                               # host-side measurement drivers
│   ├── kws_measure_coral.py
│   ├── run_offline_measurement.sh
│   ├── run_live_measurement.sh
│   ├── coral_bench_config.json
│   └── coral_live_config.json
├── measurements/                        # captured runs (summary.json + csv)
├── coralmicro/   (gitignored)           # SDK clone — see Step 3
└── build/        (gitignored)           # CMake build tree
```

**Input model**: `../../models/ds_cnn_l.tflite` (INT8-quantized; all-ops INT8 required by Edge TPU).
Quantization: input scale = 0.584702671, zero_point = 83; output scale = 1/256, zero_point = −128.

---

## Prerequisites

| Tool | Notes |
|------|-------|
| `edgetpu_compiler` | x86-64 **Linux** only. Run `scripts/setup.sh` (apt) on Linux, or use the Coral Docker image `gcr.io/google-coral/model-compiler`. |
| Python ≥ 3.8 | For `flashtool.py` and the static-model scripts. A venv at `coral/.venv` is expected by the build/flash scripts. |
| coralmicro SDK | Cloned into `coral/coralmicro/` (Step 3). |
| CMake ≥ 3.13 | Build system. |
| `arm-none-eabi-gcc` | Toolchain — installed by the coralmicro bootstrap. |

---

## Step 1 — Create a static-shape INT8 model

The Edge TPU compiler requires fully static shapes, but `ds_cnn_l.tflite` carries a
dynamic batch dimension (`-1`). Produce `../../models/ds_cnn_l_static.tflite` one of two ways:

**Option A — byte-patch (preserves the existing quantization, recommended):**

```bash
python3 scripts/fix_static_shape.py
# in: ../../models/ds_cnn_l.tflite   out: ../../models/ds_cnn_l_static.tflite
```

**Option B — re-convert from the Keras SavedModel** (re-quantizes; pass real
calibration data, otherwise it falls back to *random* calibration and loses accuracy):

```bash
python3 scripts/convert_static_int8.py --data_dir ~/data/speech_commands_v002
```

> Validate accuracy of the result with `../../training/eval_quantized_model.py`
> before shipping a re-quantized model.

## Step 2 — Compile for Edge TPU

```bash
chmod +x scripts/compile_edgetpu.sh
./scripts/compile_edgetpu.sh ../../models/ds_cnn_l_static.tflite
```

Writes `model/ds_cnn_l_static_edgetpu.tflite`. Expected — all 12 ops on the Edge TPU,
0 CPU fallbacks:

```text
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
cd coralmicro && bash setup.sh     # installs toolchain, pulls submodules (~5 min)
```

---

## Step 4 — Build

The KWS apps live in `apps/` and are pulled into the SDK build via
`add_subdirectory(../apps …)` in `coralmicro/CMakeLists.txt` — so **`apps/` is the
build source of truth** (edit apps here, not inside the SDK clone).

Configure once, then build any target:

```bash
# one-time configure (from coral/)
cmake -S coralmicro -B build -DCMAKE_BUILD_TYPE=Release

# build a target
cmake --build build -j"$(nproc)" --target kws_bench   # or kws_live / kws_coral / single_inference
```

Compiled ELFs land at `build/kws_apps/<app>/<app>`.

---

## Step 5 — Flash

Connect the board via USB and use the helper scripts (they build, then flash via
`coralmicro/scripts/flashtool.py`):

```bash
./scripts/build_and_flash_bench.sh      # offline 50-run benchmark
./scripts/build_and_flash_live.sh       # live microphone KWS
./scripts/build_and_flash_single_inference.sh
```

> **macOS:** the scripts set `DYLD_LIBRARY_PATH=/opt/homebrew/lib` for `flashtool.py`
> and expect a Python venv at `coral/.venv`.
> **Crash-loop recovery:** put the board in SDP mode first — hold USER, press RESET,
> release USER — then re-run the flash script.

---

## Step 6 — Read output

Serial is USB-CDC @115200. **The Coral CDC console drops bytes until DTR is asserted**,
so `screen`/`cat` may show nothing — use a terminal that raises DTR:

```bash
# macOS (port is usually /dev/cu.usbmodem*)
.venv/bin/python3 -m serial.tools.miniterm $(ls /dev/cu.usbmodem* | head -1) 115200

# Linux
screen /dev/ttyACM0 115200
```

> ⚠️ Never open a Coral port at **1200 baud** — that resets it to the bootloader.

Offline apps print a `BENCH,event=inference,…` CSV (compatible with the MAX78000 /
STM32 measurement format). `kws_live` prints `>>> left (73%)  mfcc=… us  infer=… us`
on a confident keyword and lights the green user LED.

---

## Apps

| App | Purpose | Input |
|-----|---------|-------|
| `kws_coral` | One offline inference | fixed "left" MFCC vector (`common/`) |
| `single_inference` | One offline inference, verbose scores | fixed "left" MFCC vector |
| `kws_bench` | 50 back-to-back inferences, BENCH CSV | fixed "left" MFCC vector |
| `kws_live` | Continuous mic → MFCC → Edge TPU | PDM microphone @16 kHz |

All apps load the model from LittleFS (`/models/ds_cnn_l_static_edgetpu.tflite`),
embedded into flash via the `DATA` clause in each app's `CMakeLists.txt`.

---

## Measure

```bash
./tools/run_offline_measurement.sh   # → measurements/offline_measurements/...
./tools/run_live_measurement.sh      # → measurements/live_measurements/...
```

These drive `kws_measure_coral.py`, which records `summary.json` + CSV (latency,
cycles, throughput). Keep prior runs when iterating so versions stay comparable.

---

## Expected performance

| Metric | Value |
|--------|-------|
| Model MACs | 3,937,717 |
| Model size (INT8 weights) | ~35–48 KB |
| Edge TPU inference | ~1.8 ms (measured; p95 ~1.9 ms) |
| Throughput | ~550 inferences/s |

---

## Notes

- `app_main()` is the entry point, called by the coralmicro FreeRTOS scheduler.
- `edgetpu_compiler` needs an x86-64 Linux host (use the Docker image on macOS/Windows).
- `sync_coral_apps.sh` is a legacy helper from when apps were edited inside the SDK
  clone; today `apps/` is edited directly and is the build source.
