#!/usr/bin/env bash
# Build + flash the STM32U5 DS-CNN KWS *32-bit / fp32-cpu* (v0) firmware from the
# CLI. Unlike the 8-bit (v1) sibling, this folder is a STM32CubeIDE *managed*
# project (.cproject, no CMake), so the build is driven by a HEADLESS CubeIDE
# invocation. The dashboard's flash_all.sh calls this with --mode {live|offline}.
#
# Usage:
#   ./tools/build_flash.sh --mode {live|offline} [--build-only|--flash-only]
#
# Mode is compile-time via Core/Inc/kws20_mode_config.h:
#   live    → KWS20_CFG_ENABLE_MEASURE=1, KWS20_CFG_MEASURE_LIVE=1  (mic + BENCH)
#   offline → KWS20_CFG_ENABLE_MEASURE=1, KWS20_CFG_MEASURE_LIVE=0  (test tensor)
#
# Requires:
#   - STM32CubeIDE (default /Applications/STM32CubeIDE.app; override $STM32CUBEIDE)
#   - STM32CubeCLT for STM32_Programmer_CLI (default /opt/ST/STM32CubeCLT_1.19.0;
#     override $STM32CUBECLT). Flashing targets the ST-Link probe, not a serial port.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_NAME="keyword_spotting_u5_ds_cnn"   # <name> in .project / artifact ${ProjName}
BUILD_CONFIG="Debug"                         # .cproject configuration to build
CLT="${STM32CUBECLT:-/opt/ST/STM32CubeCLT_1.19.0}"

# CubeIDE headless launcher (the binary inside the .app, not the .app bundle).
CUBEIDE="${STM32CUBEIDE:-/Applications/STM32CubeIDE.app/Contents/MacOS/STM32CubeIDE}"

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

PROG_CLI="$CLT/STM32CubeProgrammer/bin/STM32_Programmer_CLI"
CFG="$PROJECT_DIR/Core/Inc/kws20_mode_config.h"
ELF="$PROJECT_DIR/$BUILD_CONFIG/$PROJECT_NAME.elf"

echo "=== STM32U5 v0 (32-bit/fp32): mode=$MODE (ENABLE_MEASURE=1, MEASURE_LIVE=$MEASURE_LIVE) ==="
# Set the compile-time mode (portable BSD/GNU sed), same defines as the 8-bit folder.
sed -i.bak \
  -e 's/^#define KWS20_CFG_APP_MODE .*/#define KWS20_CFG_APP_MODE KWS20_CFG_APP_MODE_KWS_LIVE/' \
  -e 's/^#define KWS20_CFG_ENABLE_MEASURE .*/#define KWS20_CFG_ENABLE_MEASURE 1/' \
  -e "s/^#define KWS20_CFG_MEASURE_LIVE .*/#define KWS20_CFG_MEASURE_LIVE   $MEASURE_LIVE/" \
  "$CFG"
rm -f "$CFG.bak"

if [[ $DO_BUILD -eq 1 ]]; then
  [[ -x "$CUBEIDE" ]] || { echo "STM32CubeIDE not found at $CUBEIDE (set STM32CUBEIDE)" >&2; exit 1; }
  # Headless managed build. Use a throwaway workspace so a project already
  # imported in the user's GUI workspace can't cause an "already exists" abort.
  WS="$(mktemp -d "${TMPDIR:-/tmp}/u5_v0_ws.XXXXXX")"
  # Importing the project rewrites .settings/language.settings.xml with an
  # environment-specific env-hash (tracked file → spurious git churn on every
  # flash). Snapshot it and restore after the build.
  SETTINGS="$PROJECT_DIR/.settings/language.settings.xml"
  SETTINGS_BAK=""
  if [[ -f "$SETTINGS" ]]; then
    SETTINGS_BAK="$(mktemp "${TMPDIR:-/tmp}/u5_v0_langsettings.XXXXXX")"
    cp "$SETTINGS" "$SETTINGS_BAK"
  fi
  trap 'rm -rf "$WS"; [[ -n "$SETTINGS_BAK" && -f "$SETTINGS_BAK" ]] && cp "$SETTINGS_BAK" "$SETTINGS" && rm -f "$SETTINGS_BAK"' EXIT
  echo "=== Headless CubeIDE build ($PROJECT_NAME/$BUILD_CONFIG), workspace=$WS ==="
  "$CUBEIDE" \
    --launcher.suppressErrors -nosplash \
    -application org.eclipse.cdt.managedbuilder.core.headlessbuild \
    -data "$WS" \
    -import "$PROJECT_DIR" \
    -cleanBuild "$PROJECT_NAME/$BUILD_CONFIG"
  [[ -f "$ELF" ]] || { echo "Build reported success but ELF not found: $ELF" >&2; exit 1; }
fi

if [[ $DO_FLASH -eq 1 ]]; then
  [[ -f "$ELF" ]] || { echo "ELF not found: $ELF (build first)" >&2; exit 1; }
  [[ -x "$PROG_CLI" ]] || { echo "STM32_Programmer_CLI not found at $PROG_CLI (set STM32CUBECLT)" >&2; exit 1; }
  echo "=== Flash via ST-Link (SWD) ==="
  "$PROG_CLI" -c port=SWD mode=UR -w "$ELF" -rst
fi

echo "STM32U5 v0 done ($MODE). Serial: /dev/cu.usbmodem* @115200 (BENCH lines)."
