#!/usr/bin/env bash
# Build + flash the STM32U5 DS-CNN KWS firmware from the CLI (CMake project).
#
# Usage:
#   ./tools/build_flash.sh --mode {live|offline} [--build-only|--flash-only]
#
# Mode is compile-time via Core/Inc/kws20_mode_config.h:
#   live    → KWS20_CFG_ENABLE_MEASURE=1, KWS20_CFG_MEASURE_LIVE=1  (mic + BENCH)
#   offline → KWS20_CFG_ENABLE_MEASURE=1, KWS20_CFG_MEASURE_LIVE=0  (test tensor, 50 runs)
#
# Requires STM32CubeCLT (cmake/ninja/arm-none-eabi-gcc/STM32_Programmer_CLI).
# Flashing targets the ST-Link probe, not a serial port.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLT="${STM32CUBECLT:-/opt/ST/STM32CubeCLT_1.19.0}"
MODE="live"
DO_BUILD=1
DO_FLASH=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)       MODE="$2"; shift 2 ;;
    --build-only) DO_FLASH=0; shift ;;
    --flash-only) DO_BUILD=0; shift ;;
    -h|--help)    grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

case "$MODE" in
  live)    MEASURE_LIVE=1 ;;
  offline) MEASURE_LIVE=0 ;;
  *) echo "Invalid --mode '$MODE' (use live|offline)" >&2; exit 1 ;;
esac

export PATH="$CLT/CMake/bin:$CLT/Ninja/bin:$CLT/GNU-tools-for-STM32/bin:$PATH"
PROG_CLI="$CLT/STM32CubeProgrammer/bin/STM32_Programmer_CLI"
CFG="$PROJECT_DIR/Core/Inc/kws20_mode_config.h"
ELF="$PROJECT_DIR/build/Debug/keyword_spotting_u5_ds_cnn_v1.elf"

echo "=== STM32U5: mode=$MODE (ENABLE_MEASURE=1, MEASURE_LIVE=$MEASURE_LIVE) ==="
# Set the compile-time mode (portable BSD/GNU sed).
sed -i.bak \
  -e 's/^#define KWS20_CFG_APP_MODE .*/#define KWS20_CFG_APP_MODE KWS20_CFG_APP_MODE_KWS_LIVE/' \
  -e 's/^#define KWS20_CFG_ENABLE_MEASURE .*/#define KWS20_CFG_ENABLE_MEASURE 1/' \
  -e "s/^#define KWS20_CFG_MEASURE_LIVE .*/#define KWS20_CFG_MEASURE_LIVE   $MEASURE_LIVE/" \
  "$CFG"
rm -f "$CFG.bak"

if [[ $DO_BUILD -eq 1 ]]; then
  echo "=== Build (cmake --preset Debug) ==="
  cd "$PROJECT_DIR"   # CMakePresets.json is resolved from the working directory
  # A CMakeCache.txt generated in another checkout (e.g. a sibling worktree)
  # aborts the configure; drop a stale Debug cache so the preset reconfigures.
  if [[ -f build/Debug/CMakeCache.txt ]] && \
     ! grep -qF "CMAKE_HOME_DIRECTORY:INTERNAL=$PROJECT_DIR" build/Debug/CMakeCache.txt; then
    echo "  (clearing stale build/Debug cache from another worktree)"
    rm -rf build/Debug
  fi
  cmake --preset Debug >/dev/null
  cmake --build --preset Debug
fi

if [[ $DO_FLASH -eq 1 ]]; then
  [[ -f "$ELF" ]] || { echo "ELF not found: $ELF (build first)" >&2; exit 1; }
  [[ -x "$PROG_CLI" ]] || { echo "STM32_Programmer_CLI not found at $PROG_CLI (set STM32CUBECLT)" >&2; exit 1; }
  echo "=== Flash via ST-Link (SWD) ==="
  "$PROG_CLI" -c port=SWD mode=UR -w "$ELF" -rst
fi

echo "STM32U5 done ($MODE). Serial: /dev/cu.usbmodem* @115200 (BENCH lines)."
