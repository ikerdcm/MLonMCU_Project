# STM32U5 Experiments

This folder contains the STM32U5 keyword spotting project, measurement tools, and stored measurement results.

## Structure

- `u5_keyword_spotting/`
  Main STM32CubeIDE project.
- `tools/`
  Host-side scripts for export and measurement.
- `measurements/offline_measurements/`
  Saved offline benchmark runs.
- `measurements/live_measurements/`
  Saved live benchmark runs.

## Important Files

- [u5_keyword_spotting/Core/Inc/kws20_mode_config.h](/home/pascal/Documents/ml_on_mcu/MLonMCU_Project/ai85kws20netv3_model/platforms/stm32u5/u5_keyword_spotting/Core/Inc/kws20_mode_config.h)
  Selects normal live mode vs offline measurement vs live measurement.
- [tools/kws20_measure_metrics_u5.py](/home/pascal/Documents/ml_on_mcu/MLonMCU_Project/ai85kws20netv3_model/platforms/stm32u5/tools/kws20_measure_metrics_u5.py)
  Host-side measurement collector.
- [tools/kws20_measure_u5_config.json](/home/pascal/Documents/ml_on_mcu/MLonMCU_Project/ai85kws20netv3_model/platforms/stm32u5/tools/kws20_measure_u5_config.json)
  Default host-side config for baud, port, ELF path, MAC count, voltage, current, timeout.

## Firmware Modes

Set the flags in [kws20_mode_config.h](/home/pascal/Documents/ml_on_mcu/MLonMCU_Project/ai85kws20netv3_model/platforms/stm32u5/u5_keyword_spotting/Core/Inc/kws20_mode_config.h).

### 1. Normal Live App

```c
#define KWS20_CFG_ENABLE_MEASURE 0
#define KWS20_CFG_MEASURE_LIVE   0
#define KWS20_CFG_LIVE_MEASURE_MINIMAL_OUTPUT 0
```

Use this if you want the ordinary live demo without `BENCH` measurement output.

### 2. Offline Measurement

```c
#define KWS20_CFG_ENABLE_MEASURE 1
#define KWS20_CFG_MEASURE_LIVE   0
#define KWS20_CFG_LIVE_MEASURE_MINIMAL_OUTPUT 1
```

This runs a fixed offline tensor through the network and measures only inference performance.

### 3. Live Measurement

```c
#define KWS20_CFG_ENABLE_MEASURE 1
#define KWS20_CFG_MEASURE_LIVE   1
#define KWS20_CFG_LIVE_MEASURE_MINIMAL_OUTPUT 1
```

This runs microphone-triggered inference and emits `BENCH,event=...` lines over UART.

`KWS20_CFG_LIVE_MEASURE_MINIMAL_OUTPUT=1` suppresses non-essential live logs so the UART stream is mostly measurement output.

## Build / Flash

After changing the mode flags:

1. Open `u5_keyword_spotting` in STM32CubeIDE.
2. Build the project.
3. Flash the board.

The Python script does not change firmware mode. It only reads UART and summarizes results. If the wrong firmware is flashed, the summary will be incomplete.

## Offline Experiment

### Firmware

Set:

```c
#define KWS20_CFG_ENABLE_MEASURE 1
#define KWS20_CFG_MEASURE_LIVE   0
#define KWS20_CFG_LIVE_MEASURE_MINIMAL_OUTPUT 1
```

Then build and flash.

### Host Command

```bash
python /home/pascal/Documents/ml_on_mcu/MLonMCU_Project/ai85kws20netv3_model/platforms/stm32u5/tools/kws20_measure_metrics_u5.py --mode offline
```

### What It Measures

- ELF size
- `text`, `data`, `bss`
- static SRAM = `data + bss`
- `cnn_us`
- `cycles`
- estimated compute throughput
- estimated energy metrics if voltage and current are set in the JSON config

### Notes

- Offline mode uses a fixed test tensor.
- No microphone interaction is needed.
- A clean run should contain `BENCH,event=done`.

## Live Experiment

### Firmware

Set:

```c
#define KWS20_CFG_ENABLE_MEASURE 1
#define KWS20_CFG_MEASURE_LIVE   1
#define KWS20_CFG_LIVE_MEASURE_MINIMAL_OUTPUT 1
```

Then build and flash.

### Host Command

```bash
python /home/pascal/Documents/ml_on_mcu/MLonMCU_Project/ai85kws20netv3_model/platforms/stm32u5/tools/kws20_measure_metrics_u5.py --mode live
```

Optional longer run:

```bash
python /home/pascal/Documents/ml_on_mcu/MLonMCU_Project/ai85kws20netv3_model/platforms/stm32u5/tools/kws20_measure_metrics_u5.py --mode live --timeout 90
```

### What It Measures

- per-inference `cnn_us`
- per-inference `cycles`
- measured live-triggered inference events
- estimated compute throughput
- estimated energy metrics if voltage and current are set in the JSON config

### Notes

- Live mode does not emit `BENCH,event=done` because it runs continuously.
- The summary depends on how many live inferences are actually triggered during the timeout.
- If `num_inference_events` is `0`, then the board did not emit live `BENCH,event=inference` lines.

## Output Location

Results are stored automatically under:

- [measurements/offline_measurements](/home/pascal/Documents/ml_on_mcu/MLonMCU_Project/ai85kws20netv3_model/platforms/stm32u5/measurements/offline_measurements)
- [measurements/live_measurements](/home/pascal/Documents/ml_on_mcu/MLonMCU_Project/ai85kws20netv3_model/platforms/stm32u5/measurements/live_measurements)

Each run gets its own timestamped folder with:

- `serial_raw.log`
- `uart_events.csv`
- `inference_events.csv`
- `summary.json`

## Config Values and Sources

The default host config is in [tools/kws20_measure_u5_config.json](/home/pascal/Documents/ml_on_mcu/MLonMCU_Project/ai85kws20netv3_model/platforms/stm32u5/tools/kws20_measure_u5_config.json).

Important values:

- `baud=115200`
  Source: [main.c](/home/pascal/Documents/ml_on_mcu/MLonMCU_Project/ai85kws20netv3_model/platforms/stm32u5/u5_keyword_spotting/Core/Src/main.c:226)
- `clock_mhz=160`
  Source: [u5_keyword_spotting.ioc](/home/pascal/Documents/ml_on_mcu/MLonMCU_Project/ai85kws20netv3_model/platforms/stm32u5/u5_keyword_spotting/u5_keyword_spotting.ioc:116)
- `mac_ops=8797410`
  Sources:
  [network.c](/home/pascal/Documents/ml_on_mcu/MLonMCU_Project/ai85kws20netv3_model/platforms/stm32u5/u5_keyword_spotting/X-CUBE-AI/App/network.c:2744),
  [network_generate_report.txt](/home/pascal/Documents/ml_on_mcu/MLonMCU_Project/ai85kws20netv3_model/platforms/stm32u5/u5_keyword_spotting/X-CUBE-AI/App/network_generate_report.txt:23)

Voltage and current are user-provided measurement inputs. They are not measured by the firmware.

## Typical Failure Case

If `summary.json` shows:

- `num_inference_events = 0`
- `cnn_latency_us = null`
- `cycles = null`
- `model_info = null`

then the board is almost certainly not running the correct measurement firmware mode. Check [kws20_mode_config.h](/home/pascal/Documents/ml_on_mcu/MLonMCU_Project/ai85kws20netv3_model/platforms/stm32u5/u5_keyword_spotting/Core/Inc/kws20_mode_config.h), rebuild, and reflash.
