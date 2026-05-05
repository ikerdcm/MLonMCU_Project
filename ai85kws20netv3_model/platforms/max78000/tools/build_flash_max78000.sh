#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CONFIG="${SCRIPT_DIR}/kws20_measure_max78000_config.json"

CONFIG_PATH="${DEFAULT_CONFIG}"
DO_BUILD=1
DO_FLASH=1

usage() {
  cat <<'EOF'
Usage:
  build_flash_max78000.sh [--config path] [--build-only] [--flash-only]

Options:
  --config PATH   JSON config file, defaults to kws20_measure_max78000_config.json
  --build-only    Only build, do not flash
  --flash-only    Only flash, do not build
  -h, --help      Show this help

Config / env:
  Reads project, elf, board, maxim_path, openocd_root from the JSON config.
  Environment variables override config values:
    MAXIM_PATH
    OPENOCD_ROOT
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --build-only)
      DO_FLASH=0
      shift
      ;;
    --flash-only)
      DO_BUILD=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

CONFIG_PATH="$(python - <<'PY' "$CONFIG_PATH"
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve())
PY
)"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH" >&2
  exit 1
fi

cfg_get() {
  python - <<'PY' "$CONFIG_PATH" "$1"
import json, sys
from pathlib import Path
cfg = json.loads(Path(sys.argv[1]).read_text())
value = cfg.get(sys.argv[2])
if value is None:
    print("")
elif isinstance(value, bool):
    print("true" if value else "false")
else:
    print(value)
PY
}

resolve_path() {
  python - <<'PY' "$1"
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve())
PY
}

find_openocd_root() {
  local candidates=(
    "${OPENOCD_ROOT:-}"
    "${CFG_OPENOCD_ROOT:-}"
    "${MAXIM_PATH:-}/Tools/OpenOCD"
    "${MAXIM_PATH:-}/sdk/Tools/OpenOCD"
    "/home/pascal/max78000/ai8x-synthesis/sdk/Tools/OpenOCD"
    "/opt/MaximSDK/Tools/OpenOCD"
  )

  local cand=""
  local bin=""
  for cand in "${candidates[@]}"; do
    [[ -z "$cand" ]] && continue
    cand="$(resolve_path "$cand")"
    bin="${cand}/bin/Linux_x86_64/openocd"
    if [[ -x "$bin" ]]; then
      echo "$cand"
      return 0
    fi
  done

  return 1
}

PROJECT="$(cfg_get project)"
ELF="$(cfg_get elf)"
BOARD="$(cfg_get board)"
CFG_MAXIM_PATH="$(cfg_get maxim_path)"
CFG_OPENOCD_ROOT="$(cfg_get openocd_root)"

PROJECT="${PROJECT:-}"
ELF="${ELF:-}"
BOARD="${BOARD:-FTHR_RevA}"
MAXIM_PATH="${MAXIM_PATH:-$CFG_MAXIM_PATH}"
OPENOCD_ROOT="${OPENOCD_ROOT:-$CFG_OPENOCD_ROOT}"

if [[ -z "$PROJECT" ]]; then
  echo "Config value 'project' is required." >&2
  exit 1
fi

PROJECT="$(resolve_path "$PROJECT")"

if [[ -z "$ELF" ]]; then
  ELF="${PROJECT}/build/max78000.elf"
else
  ELF="$(resolve_path "$ELF")"
fi

if [[ -n "$MAXIM_PATH" ]]; then
  export MAXIM_PATH
fi

if [[ $DO_BUILD -eq 1 ]]; then
  echo "[BUILD] project=${PROJECT}"
  echo "[BUILD] board=${BOARD}"
  make -C "$PROJECT" distclean
  make -C "$PROJECT" -j "BOARD=${BOARD}"
fi

if [[ $DO_FLASH -eq 1 ]]; then
  if ! OPENOCD_ROOT="$(find_openocd_root)"; then
    echo "Could not find OpenOCD. Set OPENOCD_ROOT or openocd_root in the config." >&2
    exit 1
  fi

  OPENOCD_BIN="${OPENOCD_ROOT}/bin/Linux_x86_64/openocd"

  if [[ ! -x "$OPENOCD_BIN" ]]; then
    echo "openocd not found or not executable: $OPENOCD_BIN" >&2
    exit 1
  fi

  if [[ ! -f "$ELF" ]]; then
    echo "ELF not found: $ELF" >&2
    exit 1
  fi

  echo "[FLASH] elf=${ELF}"
  "$OPENOCD_BIN" \
    -s "$OPENOCD_ROOT" \
    -f interface/cmsis-dap.cfg \
    -f max78000.cfg \
    -c "transport select swd; program ${ELF} verify reset exit"
fi

echo "DONE"
