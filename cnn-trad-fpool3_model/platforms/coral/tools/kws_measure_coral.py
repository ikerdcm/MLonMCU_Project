#!/usr/bin/env python3
"""
Coral Dev Board Micro — KWS measurement tool.

Offline mode  : reads BENCH CSV from kws_bench (50 inferences)
Live mode     : reads kws_live output (>>> label (score%)  mfcc=X us  infer=Y us)

Results are saved under platforms/coral/measurements/<mode>_measurements/kws_metrics_<timestamp>/
"""

import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import serial
except ImportError:
    print("ERROR: pyserial fehlt. Installiere mit: pip install pyserial")
    sys.exit(1)

DEFAULT_CONFIG_PATH = Path(__file__).with_name("coral_bench_config.json")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def run_cmd(cmd, cwd, env, log_path):
    print(f"\n[RUN] {' '.join(map(str, cmd))}")
    lines = []
    with open(log_path, "w") as log:
        p = subprocess.Popen(
            list(map(str, cmd)),
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in p.stdout:
            print(line, end="")
            log.write(line)
            lines.append(line)
        ret = p.wait()
    if ret != 0:
        raise RuntimeError(f"Command failed: {' '.join(map(str, cmd))}")
    return "".join(lines)


def write_csv(path, rows):
    if not rows:
        path.write_text("# no rows\n")
        return
    keys = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def stats(values):
    values = [float(v) for v in values]
    if not values:
        return None
    v = sorted(values)
    return {
        "count": len(v),
        "min": min(v),
        "max": max(v),
        "avg": sum(v) / len(v),
        "median": statistics.median(v),
        "std": statistics.stdev(v) if len(v) > 1 else 0.0,
        "p95": v[int(0.95 * (len(v) - 1))],
    }


def estimate_uart_tx_time_s(line, baud):
    return (len(line.encode("utf-8", errors="replace")) * 10) / baud


def parse_size_output(text):
    out = {}
    m = re.search(
        r"^\s*([0-9]+)\s+([0-9]+)\s+([0-9]+)\s+([0-9]+)\s+([0-9a-fA-F]+)\s+.+$",
        text, re.M
    )
    if m:
        out["text_bytes"] = int(m.group(1))
        out["data_bytes"] = int(m.group(2))
        out["bss_bytes"]  = int(m.group(3))
        out["dec_bytes"]  = int(m.group(4))
        out["hex"]        = m.group(5)
    return out


def run_size_on_elf(elf_path, out_dir):
    text = run_cmd(
        ["arm-none-eabi-size", str(elf_path)],
        elf_path.parent, os.environ.copy(), out_dir / "size.log"
    )
    return parse_size_output(text)


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_bench_line(line):
    """Parse BENCH,event=... CSV lines (kws_bench / kws_coral format)."""
    line = line.strip()
    if not line.startswith("BENCH,"):
        return None
    fields = {}
    for part in line.split(",")[1:]:
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        fields[k.strip()] = v.strip()
    return fields


def parse_live_line(line):
    """
    Parse kws_live output lines:
      >>> left     (73%)  mfcc=112345 us  infer=1234 us
          unknown  ( 5%)  mfcc=112345 us  infer=1234 us
    """
    line = line.strip()
    confident = line.startswith(">>>")
    m = re.match(
        r'^(?:>>>|)\s*(\w+)\s+\(\s*(\d+)%\)\s+mfcc=(\d+)\s+us\s+infer=(\d+)\s+us',
        line
    )
    if not m:
        return None
    return {
        "confident": confident,
        "label":     m.group(1),
        "score_pct": int(m.group(2)),
        "mfcc_us":   int(m.group(3)),
        "infer_us":  int(m.group(4)),
    }


# ---------------------------------------------------------------------------
# UART collection
# ---------------------------------------------------------------------------

def collect_uart_offline(ser, baud, timeout_s, out_dir):
    """Collect BENCH CSV lines until event=done or timeout."""
    raw_path = out_dir / "serial_raw.log"
    uart_rows, inf_rows = [], []
    model_info = acquisition = None
    start = time.time()
    done = False

    with open(raw_path, "w") as raw:
        while time.time() - start < timeout_s:
            data = ser.readline()
            if not data:
                continue
            now     = time.time()
            elapsed = now - start
            iso     = datetime.fromtimestamp(now).isoformat(timespec="milliseconds")
            line    = data.decode("utf-8", errors="replace").strip()
            print(line)
            raw.write(f"{iso},elapsed_s={elapsed:.6f},{line}\n")
            raw.flush()

            uart_rows.append({
                "time_iso":          iso,
                "elapsed_s":         elapsed,
                "bytes":             len(line.encode("utf-8", errors="replace")),
                "uart_tx_time_s_est": estimate_uart_tx_time_s(line, baud),
                "raw":               line,
            })

            fields = parse_bench_line(line)
            if not fields:
                continue
            event = fields.get("event", "")
            if event == "model_info":
                model_info = fields
            elif event == "acquisition":
                acquisition = fields
            elif event == "inference":
                inf_rows.append(fields)
            elif event == "done":
                done = True
                break

    return uart_rows, inf_rows, model_info, acquisition, done


def collect_uart_live(ser, baud, timeout_s, out_dir):
    """Collect kws_live output lines for the given duration."""
    raw_path  = out_dir / "serial_raw.log"
    uart_rows, inf_rows = [], []
    start = time.time()
    run_idx = 0

    with open(raw_path, "w") as raw:
        while time.time() - start < timeout_s:
            data = ser.readline()
            if not data:
                continue
            now     = time.time()
            elapsed = now - start
            iso     = datetime.fromtimestamp(now).isoformat(timespec="milliseconds")
            line    = data.decode("utf-8", errors="replace").strip()
            print(line)
            raw.write(f"{iso},elapsed_s={elapsed:.6f},{line}\n")
            raw.flush()

            uart_rows.append({
                "time_iso":          iso,
                "elapsed_s":         elapsed,
                "bytes":             len(line.encode("utf-8", errors="replace")),
                "uart_tx_time_s_est": estimate_uart_tx_time_s(line, baud),
                "raw":               line,
            })

            parsed = parse_live_line(line)
            if parsed:
                inf_rows.append({
                    "run":       run_idx,
                    "mode":      "live",
                    "label":     parsed["label"],
                    "score_pct": parsed["score_pct"],
                    "confident": parsed["confident"],
                    "mfcc_us":   parsed["mfcc_us"],
                    "cnn_us":    parsed["infer_us"],   # unified name for stats
                    "cycles":    parsed["infer_us"] * 800,  # 800 MHz
                })
                run_idx += 1

    return uart_rows, inf_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_config(path):
    cfg_path = Path(path).expanduser().resolve()
    if not cfg_path.exists():
        raise RuntimeError(f"Config file not found: {cfg_path}")
    data = json.loads(cfg_path.read_text())
    return cfg_path, data


def main():
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    bootstrap_args, _ = bootstrap.parse_known_args()
    config_path, config = load_config(bootstrap_args.config)

    ap = argparse.ArgumentParser()
    ap.add_argument("--config",       default=str(config_path))
    ap.add_argument("--mode",         choices=["offline", "live"],
                                      default=config.get("mode", "offline"))
    ap.add_argument("--build-dir",    default=config.get("build_dir"))
    ap.add_argument("--target",       default=config.get("target"))
    ap.add_argument("--elf",          default=config.get("elf"))
    ap.add_argument("--port",         default=config.get("port", "/dev/ttyACM0"))
    ap.add_argument("--baud",         type=int,   default=config.get("baud", 115200))
    ap.add_argument("--clock-mhz",   type=float, default=config.get("clock_mhz", 800.0))
    ap.add_argument("--mac-ops",      type=float, default=config.get("mac_ops", 3937717.0))
    ap.add_argument("--sample-rate",  type=float, default=config.get("sample_rate", 16000.0))
    ap.add_argument("--sample-count", type=float, default=config.get("sample_count", 16000.0))
    ap.add_argument("--voltage",      type=float, default=config.get("voltage"))
    ap.add_argument("--current-ma",   type=float, default=config.get("current_ma"))
    ap.add_argument("--timeout",      type=float, default=config.get("timeout", 60.0))
    ap.add_argument("--no-build",     action="store_true",
                                      default=bool(config.get("no_build", True)))
    args = ap.parse_args()

    repo_measure_root = Path(__file__).resolve().parent.parent / "measurements"
    mode_dir = "live_measurements" if args.mode == "live" else "offline_measurements"
    out_dir  = repo_measure_root / mode_dir / datetime.now().strftime("kws_metrics_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    summary = {
        "timestamp": datetime.now().isoformat(),
        "mode":      args.mode,
        "config":    str(Path(args.config).expanduser().resolve()),
        "port":      args.port,
        "baud":      args.baud,
        "platform":  "coral_dev_board_micro",
    }

    # Optional cmake build
    if not args.no_build and args.build_dir and args.target:
        build_dir = Path(args.build_dir).expanduser().resolve()
        run_cmd(
            ["cmake", "--build", str(build_dir), "-j", str(os.cpu_count() or 4),
             "--target", args.target],
            build_dir, env, out_dir / "build.log"
        )

    # ELF size
    if args.elf:
        elf = Path(args.elf).expanduser().resolve()
    elif args.build_dir and args.target:
        elf = Path(args.build_dir).expanduser().resolve() / "apps" / args.target / args.target
    else:
        elf = None

    if elf and elf.exists():
        summary["elf"] = str(elf)
        summary["elf_size_bytes"] = elf.stat().st_size
        summary["memory"] = run_size_on_elf(elf, out_dir)

    memory = summary.get("memory")
    if memory and "data_bytes" in memory and "bss_bytes" in memory:
        static_sram = int(memory["data_bytes"]) + int(memory["bss_bytes"])
        summary["static_sram_usage"] = {
            "static_sram_bytes": static_sram,
            "static_sram_kib":   static_sram / 1024.0,
            "data_bytes":        int(memory["data_bytes"]),
            "bss_bytes":         int(memory["bss_bytes"]),
        }

    # UART collection
    print(f"\n[UART] Opening {args.port} @ {args.baud}")
    with serial.Serial(args.port, args.baud, timeout=0.2) as ser:
        time.sleep(1.0)
        ser.reset_input_buffer()
        if args.mode == "live":
            uart_rows, inf_rows = collect_uart_live(
                ser, args.baud, args.timeout, out_dir
            )
            model_info = acquisition = None
            done = False
        else:
            uart_rows, inf_rows, model_info, acquisition, done = collect_uart_offline(
                ser, args.baud, args.timeout, out_dir
            )

    write_csv(out_dir / "uart_events.csv",    uart_rows)
    write_csv(out_dir / "inference_events.csv", inf_rows)

    cnn_us_values  = [float(r["cnn_us"])  for r in inf_rows if "cnn_us"  in r]
    cycles_values  = [float(r["cycles"])  for r in inf_rows if "cycles"  in r]
    uart_tx_values = [r["uart_tx_time_s_est"] for r in uart_rows]

    summary["counts"] = {
        "num_uart_events":      len(uart_rows),
        "num_inference_events": len(inf_rows),
        "done_seen":            done,
    }
    summary["model_info"] = model_info
    summary["sensor_acquisition"] = {
        "sample_rate_hz":  args.sample_rate,
        "sample_count":    args.sample_count,
        "audio_window_s":  args.sample_count / args.sample_rate,
        "audio_window_ms": (args.sample_count / args.sample_rate) * 1000.0,
    }
    if acquisition:
        summary["sensor_acquisition"]["fw_reported"] = acquisition

    summary["cnn_latency_us"] = stats(cnn_us_values)
    summary["cycles"]         = stats(cycles_values)
    summary["uart_transmission_time_estimate_s"] = stats(uart_tx_values)

    if summary["cnn_latency_us"]:
        avg_us = summary["cnn_latency_us"]["avg"]
        avg_s  = avg_us * 1e-6
        summary["cnn_latency_ms_avg"] = avg_us / 1000.0
        summary["inferences_per_second_from_cnn_time"] = 1.0 / avg_s if avg_s > 0 else None
        summary["estimated_end_to_end_lower_bound_ms"] = (
            summary["sensor_acquisition"]["audio_window_ms"] + (avg_us / 1000.0)
        )
        summary["cycles_estimate"] = {
            "clock_mhz":                   args.clock_mhz,
            "cycles_per_inference_avg":    avg_s * args.clock_mhz * 1e6,
        }
        if args.mac_ops:
            cyc = summary["cycles_estimate"]["cycles_per_inference_avg"]
            summary["cycles_estimate"]["mac_ops_per_cycle"] = args.mac_ops / cyc
            summary["compute_estimate"] = {
                "mac_ops_per_inference": args.mac_ops,
                "mac_ops_per_second":    args.mac_ops / avg_s,
                "mops_per_second":       (args.mac_ops / avg_s) / 1e6,
            }
        if args.voltage and args.current_ma:
            power_w  = args.voltage * args.current_ma / 1000.0
            energy_j = power_w * avg_s
            summary["energy_estimate"] = {
                "voltage_v":                args.voltage,
                "current_ma":               args.current_ma,
                "power_w":                  power_w,
                "energy_per_inference_j":   energy_j,
                "energy_per_inference_uj":  energy_j * 1e6,
            }
            if args.mac_ops:
                summary["energy_estimate"]["mac_ops_per_joule"] = args.mac_ops / energy_j
                summary["energy_estimate"]["mac_ops_per_second_per_watt"] = (
                    (args.mac_ops / avg_s) / power_w
                )

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nDONE")
    print(f"Output: {out_dir}")
    print(json.dumps({
        "memory":                              summary.get("memory"),
        "static_sram_usage":                   summary.get("static_sram_usage"),
        "elf_size_bytes":                       summary.get("elf_size_bytes"),
        "cnn_latency_us":                      summary.get("cnn_latency_us"),
        "cycles":                              summary.get("cycles"),
        "sensor_acquisition":                  summary.get("sensor_acquisition"),
        "estimated_end_to_end_lower_bound_ms": summary.get("estimated_end_to_end_lower_bound_ms"),
        "compute_estimate":                    summary.get("compute_estimate"),
        "energy_estimate":                     summary.get("energy_estimate"),
        "counts":                              summary.get("counts"),
    }, indent=2))


if __name__ == "__main__":
    main()
