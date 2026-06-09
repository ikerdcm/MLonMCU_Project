#!/usr/bin/env bash
# Launch the MCU KWS dashboard.
# Prefers an interpreter that already has PySide6 + pyserial (e.g. your conda
# base); otherwise creates a local venv and installs them.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

for PY in python python3; do
    if command -v "$PY" >/dev/null 2>&1 && "$PY" -c 'import PySide6, serial' 2>/dev/null; then
        exec "$PY" -m mcu_stream
    fi
done

VENV="$DIR/.venv-dashboard"
if [[ ! -d "$VENV" ]]; then
    echo "=== Creating venv at $VENV ==="
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install --upgrade pip
    "$VENV/bin/pip" install -r "$DIR/requirements.txt"
fi
exec "$VENV/bin/python" -m mcu_stream
