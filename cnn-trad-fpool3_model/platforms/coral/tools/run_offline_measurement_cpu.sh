#!/usr/bin/env bash
# Offline benchmark capture for the FP32 CPU app (kws_bench_cpu, config: fp32-cpu).
# Board must already be flashed with kws_bench_cpu (build_and_flash_bench_cpu.sh).
# Counterpart of run_offline_measurement.sh (which targets the int8 kws_bench).
#
# Usage:  ./tools/run_offline_measurement_cpu.sh [--port /dev/tty.usbmodem*] [other overrides]
# On macOS pass --port /dev/tty.usbmodem* ; the config defaults to /dev/ttyACM0 (Linux).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORAL_DIR="$(dirname "$SCRIPT_DIR")"   # tools/ -> coral/
cd "$CORAL_DIR"                         # so relative build/elf paths in the config resolve
python3 "$SCRIPT_DIR/kws_measure_coral.py" \
  --config "$SCRIPT_DIR/coral_bench_cpu_config.json" \
  --mode offline "$@"
