#!/usr/bin/env python3
import argparse
import sys
import time
from pathlib import Path

import serial

parser = argparse.ArgumentParser()
parser.add_argument("--port", default="/dev/ttyACM0")
parser.add_argument("--baud", type=int, default=115200)
parser.add_argument("--log", required=True)
args = parser.parse_args()

log_path = Path(args.log)
log_path.parent.mkdir(parents=True, exist_ok=True)

print(f"[UART] opening {args.port} @ {args.baud}")
print(f"[UART] logging to {log_path}")
print("[UART] press Ctrl+C to stop")

with serial.Serial(args.port, args.baud, timeout=0.1) as ser, open(log_path, "wb") as f:
    ser.reset_input_buffer()
    while True:
        data = ser.read(4096)
        if data:
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()
            f.write(data)
            f.flush()
        else:
            time.sleep(0.01)
