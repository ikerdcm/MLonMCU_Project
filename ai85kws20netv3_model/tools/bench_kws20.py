#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    print("pyserial missing. Install with: pip install pyserial")
    sys.exit(1)


def run_cmd(cmd, cwd=None, env=None, log_file=None):
    print(f"\n[RUN] {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    lines = []
    with open(log_file, "w") if log_file else open(os.devnull, "w") as f:
        for line in proc.stdout:
            print(line, end="")
            lines.append(line)
            f.write(line)

    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"Command failed with exit code {rc}: {' '.join(cmd)}")
    return "".join(lines)


def parse_memory_from_build_log(text):
    result = {}

    flash_match = re.search(r"FLASH:\s+([0-9]+)\s+B\s+512\s+KB\s+([0-9.]+)%", text)
    sram_match = re.search(r"SRAM:\s+([0-9]+)\s+B\s+128\s+KB\s+([0-9.]+)%", text)

    if flash_match:
        result["flash_bytes"] = int(flash_match.group(1))
        result["flash_percent"] = float(flash_match.group(2))

    if sram_match:
        result["sram_bytes"] = int(sram_match.group(1))
        result["sram_percent"] = float(sram_match.group(2))

    size_match = re.search(
        r"\n\s*text\s+data\s+bss\s+dec\s+hex\s+filename\s*\n\s*"
        r"([0-9]+)\s+([0-9]+)\s+([0-9]+)\s+([0-9]+)\s+([0-9a-fA-F]+)",
        text,
    )

    if size_match:
        result["text_bytes"] = int(size_match.group(1))
        result["data_bytes"] = int(size_match.group(2))
        result["bss_bytes"] = int(size_match.group(3))
        result["dec_bytes"] = int(size_match.group(4))
        result["hex"] = size_match.group(5)

    return result


def find_serial_port(preferred=None):
    if preferred:
        return preferred

    ports = list(serial.tools.list_ports.comports())
    if not ports:
        raise RuntimeError("No serial ports found. Check USB cable / board connection.")

    print("\nAvailable serial ports:")
    for p in ports:
        print(f"  {p.device}: {p.description}")

    # Prefer ttyACM because MAX78000FTHR virtual COM often appears there.
    for p in ports:
        if "ttyACM" in p.device:
            return p.device

    return ports[0].device


def parse_bench_line(line):
    """
    Parses structured benchmark lines from UART.

    Supported formats:

    1) Explicit BENCH line:
       BENCH,event=inference,cnn_cycles=123456,cnn_us=2500,result=five,confidence=87

    2) Existing KWS20 debug line:
       CNN Time: 2534 us
    """
    stripped = line.strip()

    # Existing firmware debug output
    m = re.search(r"CNN Time:\s*(\d+)\s*us", stripped)
    if m:
        return {
            "raw": stripped,
            "event": "inference",
            "source": "debug_cnn_time",
            "cnn_us": int(m.group(1)),
        }

    # Optional explicit benchmark format
    if not stripped.startswith("BENCH,"):
        return None

    out = {"raw": stripped}
    parts = stripped.split(",")[1:]

    for part in parts:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip()
        value = value.strip()

        if re.fullmatch(r"-?\d+", value):
            out[key] = int(value)
        elif re.fullmatch(r"-?\d+\.\d+", value):
            out[key] = float(value)
        else:
            out[key] = value

    return out

def collect_serial(port, baud, duration_s, raw_log_path, csv_path):
    print(f"\n[UART] Opening {port} @ {baud}")
    print(f"[UART] Logging for {duration_s} seconds")
    print("[UART] Press Ctrl+C to stop early\n")

    rows = []
    start = time.time()

    with serial.Serial(port, baud, timeout=1) as ser, \
            open(raw_log_path, "w") as raw_log:

        try:
            while time.time() - start < duration_s:
                line_bytes = ser.readline()
                if not line_bytes:
                    continue

                ts = time.time()
                try:
                    line = line_bytes.decode("utf-8", errors="replace").strip()
                except Exception:
                    line = repr(line_bytes)

                print(line)
                raw_log.write(f"{ts:.6f},{line}\n")
                raw_log.flush()

                parsed = parse_bench_line(line)
                if parsed is not None:
                    parsed["host_time_s"] = ts
                    rows.append(parsed)

        except KeyboardInterrupt:
            print("\n[UART] stopped by user")

    # Write CSV with union of all keys.
    keys = []
    for row in rows:
        for k in row.keys():
            if k not in keys:
                keys.append(k)

    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    else:
        with open(csv_path, "w") as f:
            f.write("# No BENCH lines found. Raw UART log still saved.\n")

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project",
        default=str(Path.home() / "max78000/ai8x-synthesis/sdk/Examples/MAX78000/CNN/kws20_demo"),
        help="MSDK project directory",
    )
    parser.add_argument("--board", default="FTHR_RevA")
    parser.add_argument("--port", default=None)
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--no-flash", action="store_true")
    parser.add_argument(
        "--out-root",
        default=str(Path.home() / "max78000/measurements"),
    )
    parser.add_argument(
        "--ocd-root",
        default=str(Path.home() / "max78000/ai8x-synthesis/openocd"),
    )
    args = parser.parse_args()

    project = Path(args.project).expanduser().resolve()
    out_dir = Path(args.out_root).expanduser().resolve() / datetime.now().strftime("%Y%m%d_%H%M%S_kws20")
    out_dir.mkdir(parents=True, exist_ok=True)

    ocd_root = Path(args.ocd_root).expanduser().resolve()
    ocd_bin = ocd_root / "bin/Linux_x86_64/openocd"

    env = os.environ.copy()
    env["MAXIM_PATH"] = str(Path.home() / "max78000/ai8x-synthesis/sdk")

    summary = {
        "timestamp": datetime.now().isoformat(),
        "project": str(project),
        "board": args.board,
        "baud": args.baud,
        "duration_s": args.duration,
        "ocd_root": str(ocd_root),
        "ocd_bin": str(ocd_bin),
    }

    if not project.exists():
        raise RuntimeError(f"Project directory not found: {project}")

    if not args.no_build:
        build_log = out_dir / "build.log"
        build_text = run_cmd(["make", "distclean"], cwd=project, env=env, log_file=out_dir / "distclean.log")
        build_text += run_cmd(["make", "-j", f"BOARD={args.board}"], cwd=project, env=env, log_file=build_log)
        summary["memory"] = parse_memory_from_build_log(build_text)

    elf = project / "build/max78000.elf"
    if not elf.exists():
        raise RuntimeError(f"ELF not found: {elf}")

    summary["elf"] = str(elf)
    summary["elf_size_bytes"] = elf.stat().st_size

    if not args.no_flash:
        flash_log = out_dir / "flash.log"
        cmd = [
            str(ocd_bin),
            "-s", str(ocd_root),
            "-f", "interface/cmsis-dap.cfg",
            "-f", "max78000.cfg",
            "-c", f"transport select swd; program {elf} verify reset exit",
        ]
        run_cmd(cmd, cwd=project, env=env, log_file=flash_log)

    port = find_serial_port(args.port)
    summary["serial_port"] = port

    rows = collect_serial(
        port=port,
        baud=args.baud,
        duration_s=args.duration,
        raw_log_path=out_dir / "serial_raw.log",
        csv_path=out_dir / "bench.csv",
    )

    summary["num_bench_rows"] = len(rows)

    if rows:
        cnn_us_values = [r["cnn_us"] for r in rows if "cnn_us" in r and isinstance(r["cnn_us"], (int, float))]
        if cnn_us_values:
            summary["cnn_us_min"] = min(cnn_us_values)
            summary["cnn_us_max"] = max(cnn_us_values)
            summary["cnn_us_avg"] = sum(cnn_us_values) / len(cnn_us_values)

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[DONE] Measurement directory:")
    print(out_dir)
    print("\nFiles:")
    for p in sorted(out_dir.iterdir()):
        print(f"  {p}")


if __name__ == "__main__":
    main()
