#!/usr/bin/env bash
# Build + flash all three MCUs (Coral, MAX78000, STM32U5) in one chosen mode.
#
# Usage:
#   ./flash_all.sh --mode {live|offline} [--only coral,max,stm32] [--sequential]
#
# Each MCU flashes through its OWN debug interface (Coral=USB VID/PID,
# MAX78000=CMSIS-DAP/openocd, STM32=ST-Link), so there is no serial-port
# ambiguity and all three can run in parallel. Per-MCU logs go to /tmp.
#
# Mode mapping:
#   live    → mic + BENCH output      offline → fixed test tensor + BENCH output
#   Coral:    live=kws_live           offline=kws_bench
#   MAX/STM:  ENABLE_MEASURE=1, MEASURE_LIVE=1 (live) / 0 (offline)
set -uo pipefail

PLATFORMS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORAL_DIR="$PLATFORMS_DIR/coral"
MAX_DIR="$PLATFORMS_DIR/max78000/keyword_spotting_max78000_ds_cnn_8-bit_with-acc"
STM_DIR="$PLATFORMS_DIR/stm32u5/keyword_spotting_u5_ds_cnn_8-bit"

MODE="live"
ONLY="coral,max,stm32"
PARALLEL=1
REPORT="keywords"   # keywords | unknown | silence | both (applies to all 3 boards)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)       MODE="$2"; shift 2 ;;
    --only)       ONLY="$2"; shift 2 ;;
    --report)     REPORT="$2"; shift 2 ;;
    --sequential) PARALLEL=0; shift ;;
    -h|--help)    grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

[[ "$MODE" == "live" || "$MODE" == "offline" ]] || { echo "Invalid --mode '$MODE'" >&2; exit 1; }
[[ "$MODE" == "live" ]] && MEASURE_LIVE=1 || MEASURE_LIVE=0

# Report-class gating — which non-keyword classes the firmware reports, applied
# to ALL three boards' live source before building (idx 10=silence, 11=unknown).
case "$REPORT" in
  keywords) RU=0; RS=0 ;;
  unknown)  RU=1; RS=0 ;;
  silence)  RU=0; RS=1 ;;
  both)     RU=1; RS=1 ;;
  *) echo "Invalid --report '$REPORT' (keywords|unknown|silence|both)" >&2; exit 1 ;;
esac
set_report() {  # $1 = source file containing the #define KWS_REPORT_* lines
  [[ -f "$1" ]] && sed -i.bak \
    -e "s/^#define KWS_REPORT_UNKNOWN .*/#define KWS_REPORT_UNKNOWN $RU/" \
    -e "s/^#define KWS_REPORT_SILENCE .*/#define KWS_REPORT_SILENCE $RS/" \
    "$1" && rm -f "$1.bak"
}
set_report "$CORAL_DIR/apps/kws_live/kws_live.cc"
set_report "$MAX_DIR/kws20_live.c"
set_report "$STM_DIR/Core/Src/kws20_live.c"
echo "=== report gating: $REPORT (unknown=$RU silence=$RS) on all 3 boards ==="

set_measure_mode() {  # $1 = path to kws20_mode_config.h
  sed -i.bak \
    -e 's/^#define KWS20_CFG_ENABLE_MEASURE .*/#define KWS20_CFG_ENABLE_MEASURE 1/' \
    -e "s/^#define KWS20_CFG_MEASURE_LIVE .*/#define KWS20_CFG_MEASURE_LIVE   $MEASURE_LIVE/" \
    "$1" && rm -f "$1.bak"
}

flash_coral() {
  cd "$CORAL_DIR" || return 1
  if [[ "$MODE" == "live" ]]; then ./scripts/build_and_flash_live.sh
  else ./scripts/build_and_flash_bench.sh; fi
}

flash_max() {
  set_measure_mode "$MAX_DIR/kws20_mode_config.h" || return 1
  cd "$MAX_DIR" || return 1
  ./tools/build_flash_max78000_ds_cnn_v1.sh
}

flash_stm32() {
  "$STM_DIR/tools/build_flash.sh" --mode "$MODE"
}

run_one() {  # $1 = mcu name
  local mcu="$1" log="/tmp/flash_all_$1.log"
  echo "[$mcu] starting (mode=$MODE) → $log"
  ( "flash_$mcu" ) >"$log" 2>&1
  local rc=$?
  echo "$rc" > "/tmp/flash_all_$mcu.rc"
  if [[ $rc -eq 0 ]]; then echo "[$mcu] ✅ done"; else echo "[$mcu] ❌ FAILED (rc=$rc) — see $log"; fi
  return $rc
}

IFS=',' read -ra MCUS <<< "$ONLY"
echo "=== flash_all: mode=$MODE, targets=${MCUS[*]}, parallel=$PARALLEL ==="

pids=()
for mcu in "${MCUS[@]}"; do
  case "$mcu" in coral|max|stm32) ;; *) echo "Unknown MCU '$mcu' (use coral,max,stm32)"; exit 1 ;; esac
  rm -f "/tmp/flash_all_$mcu.rc"
  if [[ $PARALLEL -eq 1 ]]; then run_one "$mcu" & pids+=("$!"); else run_one "$mcu"; fi
done
[[ $PARALLEL -eq 1 ]] && wait "${pids[@]}" 2>/dev/null

echo; echo "=== summary (mode=$MODE) ==="
fail=0
for mcu in "${MCUS[@]}"; do
  rc=$(cat "/tmp/flash_all_$mcu.rc" 2>/dev/null || echo "?")
  if [[ "$rc" == "0" ]]; then echo "  $mcu: OK"; else echo "  $mcu: FAILED (rc=$rc, log: /tmp/flash_all_$mcu.log)"; fail=1; fi
done
[[ $fail -eq 0 ]] && echo "All requested MCUs flashed in '$MODE' mode." || echo "Some MCUs failed — check their logs."
exit $fail
