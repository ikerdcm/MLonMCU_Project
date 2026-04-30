#!/usr/bin/env bash
set -euo pipefail

PLATFORM="${1:?usage: ./tools/log_uart.sh <stm32u5|max78000> [port] [baud]}"
PORT="${2:-/dev/ttyACM0}"
BAUD="${3:-115200}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIR="$REPO/logs/$PLATFORM"
mkdir -p "$DIR"

TS="$(date +%Y%m%d_%H%M%S)"
LOG="$DIR/${PLATFORM}_${TS}.log"

ln -sfn "$LOG" "$DIR/latest.log"

echo "[UART] platform: $PLATFORM"
echo "[UART] port:     $PORT"
echo "[UART] baud:     $BAUD"
echo "[UART] log:      $LOG"
echo "[UART] exit screen with: Ctrl+A, K, y"

sudo fuser -k "$PORT" 2>/dev/null || true
screen -L -Logfile "$LOG" "$PORT" "$BAUD"

echo "[UART] saved: $LOG"
