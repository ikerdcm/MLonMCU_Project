#!/usr/bin/env bash
set -euo pipefail
ROOT='/home/pascal/Documents/ml_on_mcu/MLonMCU_Project/cnn-trad-fpool3_model/platforms/stm32u5'
python3 "$ROOT/tools/kws20_measure_metrics_u5.py" \
  --config "$ROOT/tools/kws20_measure_u5_ds_cnn_config.json" \
  --mode live "$@"
