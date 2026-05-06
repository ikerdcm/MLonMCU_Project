#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /home/pascal/max78000/ai8x-synthesis/.venv/bin/activate
python "$SCRIPT_DIR/quantize_kws20_u5_onnx.py" --config "$SCRIPT_DIR/quantize_kws20_u5_onnx_config.json"
