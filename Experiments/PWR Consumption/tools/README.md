# PWR Consumption Tools Tutorial

This directory contains the helper scripts used to flash the STM32U5 board, capture power traces with the PPK2, and analyze the resulting measurements.

## Directory layout

- `plot_power_traces.py` - plot and inspect saved power traces.
- `analyze_power_peaks.py` - post-process current spikes and summarize inference windows.
- `ppk2/` - automated PPK2 capture and live-view tooling.

## Typical workflow

The flow is:

1. Build and flash the firmware from the model-specific STM32U5 directory.
2. Capture the power trace with the PPK2 tools.
3. Save the resulting CSV/JSON/NPZ files under the matching `PWR Consumption/<Board>/<model>/<mode>/` folder.
4. Plot or analyze the saved trace.

## 1. Flash the STM32U5 firmware

The build-and-flash script is not run from this `tools/` directory. You need to move into the model folder first.

Example for the DS-CNN U5 offline build:

```bash
cd cnn-trad-fpool3_model/platforms/stm32u5/keyword_spotting_u5_ds_cnn_8-bit
./tools/build_flash.sh --mode offline
```

What this does:

- Builds the firmware with the selected CMake preset.
- Flashes the `keyword_spotting_u5_ds_cnn_v1.elf` image over ST-Link.
- Resets the board so it starts running the measurement loop.

If you only want to flash an already built binary, use:

```bash
./tools/build_flash.sh --mode offline --flash-only
```

## 2. Capture power measurements with the PPK2

The automated PPK2 scripts live in:

```bash
cd "Experiments/PWR Consumption/tools/ppk2"
```

### Recommended capture script

Use `run_power_test.py` for a complete capture and analysis run:

```bash
/tmp/ppk2venv/bin/python run_power_test.py 38 on
```

The first argument is the capture duration in seconds. The second argument is the label written into the output JSON and CSV files.

This script:

- Finds the connected PPK2.
- Captures current samples at roughly 100 kHz.
- Detects inference spikes from the current trace.
- Computes idle power, inference power, latency, and energy per inference.
- Saves `.csv`, `.json`, `.npz`, and inference summary files.

### Live view

If you want to watch the trace in a browser while capturing, use:

```bash
/tmp/ppk2venv/bin/python ppk2_live.py
```

### Raw capture

If you only want the raw samples and logic channels, use the lower-level capture script:

```bash
/tmp/ppk2venv/bin/python ppk2_capture.py
```

## 3. What the measurements should look like

For the U5 int8 setup used in this project, a healthy offline capture typically looks like this:

- Idle current: about 10 mA.
- Inference current: about 21 mA.
- Mean inference time: about 64-72 ms.
- Energy per inference: about 4.7-4.8 mJ.

The important signal is the gap between idle and inference current. If the firmware stays too active while waiting, the inference spikes become hard to detect. The sleep-idle firmware path fixes that by lowering the idle current enough that each inference shows up clearly.

## 4. Save and organize the outputs

Keep the exported files with the experiment they belong to. A typical capture produces:

- `on.csv`
- `on.json`
- `on.npz`
- `on_inferences.csv`

The broader project convention is to store traces under:

```text
Experiments/PWR Consumption/<Board>/<model>/<mode>/
```

Example:

```text
Experiments/PWR Consumption/U5/cnn-trad-fpool3_model_v1/offline/
```

## 5. Inspect the results

Use the plotting and analysis helpers from the parent `tools/` directory:

```bash
python3 plot_power_traces.py
python3 analyze_power_peaks.py
```

These are useful for confirming:

- Where the inference spikes happen.
- Whether the trace has the expected idle/inference separation.
- Whether the energy numbers match the measurement summary.

## 6. Common pitfalls

- Running `./tools/build_flash.sh` from the repo root will fail unless you are already inside the model-specific STM32U5 directory.
- The PPK2 cannot be shared by multiple programs at once. Close the nRF Connect Power Profiler app before using the automated scripts.
- Make sure the board and the PPK2 wiring match the measurement mode you want. For U5 measurements, the PPK2 should be in series with the MCU VDD path and share ground with the board.

## Quick reference

Flash offline build:

```bash
cd cnn-trad-fpool3_model/platforms/stm32u5/keyword_spotting_u5_ds_cnn_8-bit
./tools/build_flash.sh --mode offline
```

Capture power:

```bash
cd "Experiments/PWR Consumption/tools/ppk2"
/tmp/ppk2venv/bin/python run_power_test.py 38 on
```

Plot results:

```bash
cd "Experiments/PWR Consumption/tools"
python3 plot_power_traces.py
```