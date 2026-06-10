#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CONFIG="${SCRIPT_DIR}/kws20_ds_cnn_max78000_config.json"

CONFIG_PATH="${DEFAULT_CONFIG}"
DO_BUILD=1
DO_FLASH=1

usage() {
  cat <<'EOF'
Usage:
  build_flash_max78000_ds_cnn.sh [--config path] [--build-only] [--flash-only]

Options:
  --config PATH   JSON config file (default: kws20_ds_cnn_max78000_config.json)
  --build-only    Only build, do not flash
  --flash-only    Only flash, do not build
  -h, --help      Show this help

The config JSON must contain:
  project       Path to the project directory
  board         MSDK board name (default: FTHR_RevA)
  maxim_path    Path to the MSDK root
  openocd_root  Path to the OpenOCD installation with max78000.cfg
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)     CONFIG_PATH="$2"; shift 2 ;;
    --build-only) DO_FLASH=0; shift ;;
    --flash-only) DO_BUILD=0; shift ;;
    -h|--help)    usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

cfg_get() {
  python3 - <<'PY' "$CONFIG_PATH" "$1"
import json, sys
from pathlib import Path
cfg = json.loads(Path(sys.argv[1]).read_text())
v = cfg.get(sys.argv[2])
print("" if v is None else str(v))
PY
}

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH" >&2
  exit 1
fi

PROJECT="$(cfg_get project)"
BOARD="$(cfg_get board)"
MAXIM_PATH="$(cfg_get maxim_path)"
OPENOCD_ROOT="$(cfg_get openocd_root)"

BOARD="${BOARD:-FTHR_RevA}"
ELF="${PROJECT}/build/max78000.elf"

if [[ -z "$PROJECT" ]]; then
  echo "Config value 'project' is required." >&2
  exit 1
fi

# ── Build ──────────────────────────────────────────────────────────────────────
if [[ $DO_BUILD -eq 1 ]]; then
  echo "[BUILD] project=${PROJECT}"
  echo "[BUILD] board=${BOARD}, MAXIM_PATH=${MAXIM_PATH}"

  MAKE_ARGS=("BOARD=${BOARD}" "COMPILER=GCC")
  [[ -n "$MAXIM_PATH" ]] && MAKE_ARGS+=("MAXIM_PATH=${MAXIM_PATH}")

  make -C "$PROJECT" distclean "${MAKE_ARGS[@]}"
  make -C "$PROJECT" -j "${MAKE_ARGS[@]}"
  echo "[BUILD] Done → ${ELF}"
fi

# ── Flash ──────────────────────────────────────────────────────────────────────
if [[ $DO_FLASH -eq 1 ]]; then
  OPENOCD_BIN="${OPENOCD_ROOT}/bin/Linux_x86_64/openocd"

  if [[ ! -x "$OPENOCD_BIN" ]]; then
    echo "openocd not found or not executable: $OPENOCD_BIN" >&2
    exit 1
  fi

  if [[ ! -f "$ELF" ]]; then
    echo "ELF not found: $ELF  (run with --build-only first?)" >&2
    exit 1
  fi

  echo "[FLASH] elf=${ELF}"
  "$OPENOCD_BIN" \
    -s "$OPENOCD_ROOT" \
    -f interface/cmsis-dap.cfg \
    -f max78000.cfg \
    -c "transport select swd; program ${ELF} verify reset exit"

  echo "[FLASH] Done"
fi

echo "DONE"
