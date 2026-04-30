#!/usr/bin/env bash
set -euo pipefail

# Default settings
PROJECT="$HOME/max78000/ai8x-synthesis/sdk/Examples/MAX78000/CNN/kws20_demo"
BOARD="FTHR_RevA"
PORT="/dev/ttyACM0"
DURATION="${1:-120}"
VENV="$HOME/max78000/ai8x-synthesis/.venv"
BENCH_SCRIPT="$HOME/max78000/tools/bench_kws20.py"

echo "========================================"
echo " MAX78000 KWS20 Benchmark"
echo "========================================"
echo "Project:  $PROJECT"
echo "Board:    $BOARD"
echo "Port:     $PORT"
echo "Duration: ${DURATION}s"
echo

# Check files
if [ ! -d "$PROJECT" ]; then
    echo "ERROR: Project folder not found:"
    echo "$PROJECT"
    exit 1
fi

if [ ! -f "$BENCH_SCRIPT" ]; then
    echo "ERROR: Benchmark script not found:"
    echo "$BENCH_SCRIPT"
    exit 1
fi

if [ ! -d "$VENV" ]; then
    echo "ERROR: Python venv not found:"
    echo "$VENV"
    exit 1
fi

# Check serial port
if [ ! -e "$PORT" ]; then
    echo "ERROR: Serial port not found: $PORT"
    echo "Available ports:"
    ls /dev/ttyACM* /dev/ttyUSB* 2>/dev/null || true
    exit 1
fi

# Check if port is busy
if sudo lsof "$PORT" >/tmp/kws20_port_check.txt 2>/dev/null; then
    echo "ERROR: Serial port is busy:"
    cat /tmp/kws20_port_check.txt
    echo
    echo "Close screen/serial monitor first, or run:"
    echo "  sudo fuser -k $PORT"
    exit 1
fi

# Activate Python environment
source "$VENV/bin/activate"

# Run benchmark
"$BENCH_SCRIPT" \
  --project "$PROJECT" \
  --board "$BOARD" \
  --duration "$DURATION" \
  --port "$PORT"

echo
echo "Latest measurement folder:"
ls -td "$HOME/max78000/measurements/"* | head -1
