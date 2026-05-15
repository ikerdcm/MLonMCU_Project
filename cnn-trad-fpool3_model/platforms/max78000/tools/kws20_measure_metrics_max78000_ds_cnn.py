#!/usr/bin/env python3
"""
Measurement script for DS-CNN-L on MAX78000 FTHR (CPU-only, float32).

Collects BENCH CSV output from UART, computes latency / memory / energy summary.
Saves results to platforms/max78000/measurements/{offline,live}_measurements/.

Usage:
  python3 kws20_measure_metrics_max78000_ds_cnn.py --mode offline
  python3 kws20_measure_metrics_max78000_ds_cnn.py --mode live
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
    print("ERROR: pyserial not found. Install with: pip install pyserial")
    sys.exit(1)

DEFAULT_CONFIG_PATH = Path(__file__).with_name("kws20_measure_max78000_ds_cnn_offline_config.json")

# DS-CNN-L 12-class labels (index order matches firmware)
LABELS = [
    "down", "go", "left", "no", "off", "on",
    "right", "stop", "up", "yes", "silence", "unknown",
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def run_cmd(cmd, cwd, env, log_path):
    print(f"\n[RUN] {' '.join(map(str, cmd))}")
    lines = []
    with open(log_path, "w") as log:
        p = subprocess.Popen(
            list(map(str, cmd)),
            cwd=cwd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
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
        "count":  len(v),
        "min":    min(v),
        "max":    max(v),
        "avg":    sum(v) / len(v),
        "median": statistics.median(v),
        "std":    statistics.stdev(v) if len(v) > 1 else 0.0,
        "p95":    v[int(0.95 * (len(v) - 1))],
    }


def estimate_uart_tx_time_s(line, baud):
    return (len(line.encode("utf-8", errors="replace")) * 10) / baud


def parse_size_output(text):
    out = {}
    m = re.search(r"^\s*([0-9]+)\s+([0-9]+)\s+([0-9]+)\s+([0-9]+)\s+([0-9a-fA-F]+)\s+.+$",
                  text, re.M)
    if m:
        out["text_bytes"] = int(m.group(1))
        out["data_bytes"] = int(m.group(2))
        out["bss_bytes"]  = int(m.group(3))
        out["dec_bytes"]  = int(m.group(4))
        out["hex"]        = m.group(5)
    return out


def parse_memory_from_build(text):
    out = parse_size_output(text)
    m = re.search(r"FLASH:\s+([0-9]+)\s+B\s+512\s+KB\s+([0-9.]+)%", text)
    if m:
        out["flash_bytes"]   = int(m.group(1))
        out["flash_percent"] = float(m.group(2))
    m = re.search(r"SRAM:\s+([0-9]+)\s+B\s+128\s+KB\s+([0-9.]+)%", text)
    if m:
        out["sram_bytes"]   = int(m.group(1))
        out["sram_percent"] = float(m.group(2))
    return out


def run_size_on_elf(elf_path, out_dir):
    text = run_cmd(["arm-none-eabi-size", str(elf_path)],
                   elf_path.parent, os.environ.copy(), out_dir / "size.log")
    return parse_size_output(text)


def parse_bench_fields(line):
    if not line.startswith("BENCH,"):
        return None
    out = {}
    for part in line.split(",")[1:]:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        out[key.strip()] = value.strip()
    return out


# ── UART collection ────────────────────────────────────────────────────────────

def collect_uart(ser, baud, timeout_s, out_dir, mode):
    raw_path  = out_dir / "serial_raw.log"
    uart_rows = []
    inf_rows  = []
    model_info   = None
    acquisition  = None
    done_seen    = False
    start = time.time()
    print(f"\n[UART] capturing for up to {timeout_s:.0f} s "
          f"(Ctrl+C = stop early, summary still written)")

    with open(raw_path, "w") as raw:
        try:
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
                    "time_iso":           iso,
                    "elapsed_s":          elapsed,
                    "bytes":              len(line.encode("utf-8", errors="replace")),
                    "uart_tx_time_s_est": estimate_uart_tx_time_s(line, baud),
                    "raw":                line,
                })

                bench = parse_bench_fields(line)
                if bench:
                    event = bench.get("event", "")
                    if event == "model_info":
                        model_info = bench
                    elif event == "acquisition":
                        acquisition = bench
                    elif event == "inference":
                        row = {"time_iso": iso, "elapsed_s": elapsed, "raw": line}
                        for key in ("run", "cnn_us", "cycles", "pred_idx",
                                    "mode", "frontend", "audio_window_ms"):
                            if key in bench:
                                row[key] = bench[key]
                        inf_rows.append(row)
                    elif event == "done" and mode == "offline":
                        done_seen = True
                        break
        except KeyboardInterrupt:
            print(f"\n[UART] Ctrl+C — finishing after {time.time()-start:.1f} s")

    if mode == "offline" and not done_seen:
        print("[WARN] BENCH,event=done not received — timeout or incomplete run")

    return uart_rows, inf_rows, model_info, acquisition


# ── Derived metrics ────────────────────────────────────────────────────────────

def add_derived(summary, args):
    cnn = summary.get("cnn_latency_us")
    if not cnn:
        return
    avg_us = cnn["avg"]
    avg_s  = avg_us * 1e-6
    summary["cnn_latency_ms_avg"] = avg_us / 1000.0
    summary["inferences_per_second"] = 1.0 / avg_s if avg_s > 0 else None

    if args.mode == "live" and "sensor_acquisition" not in summary:
        summary["sensor_acquisition"] = {
            "sample_rate_hz": args.sample_rate,
            "sample_count":   args.sample_count,
            "audio_window_s": args.sample_count / args.sample_rate,
            "audio_window_ms": (args.sample_count / args.sample_rate) * 1000.0,
        }

    cycles = summary.get("cycles")
    if cycles and args.mac_ops:
        summary["mac_ops_per_cycle"] = args.mac_ops / cycles["avg"]

    if args.mac_ops:
        summary["compute_estimate"] = {
            "mac_ops_per_inference": args.mac_ops,
            "mac_ops_per_second":    args.mac_ops / avg_s,
            "gops":                  (args.mac_ops / avg_s) / 1e9,
        }

    if args.voltage and args.current_ma:
        power_w  = args.voltage * args.current_ma / 1000.0
        energy_j = power_w * avg_s
        summary["energy_estimate"] = {
            "voltage_v":               args.voltage,
            "current_ma":              args.current_ma,
            "power_w":                 power_w,
            "energy_per_inference_j":  energy_j,
            "energy_per_inference_uj": energy_j * 1e6,
        }
        if args.mac_ops:
            summary["energy_estimate"]["mac_ops_per_joule"] = args.mac_ops / energy_j


# ── Main ───────────────────────────────────────────────────────────────────────

def load_config(path):
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise RuntimeError(f"Config not found: {p}")
    return p, json.loads(p.read_text())


def main():
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    bargs, _ = bootstrap.parse_known_args()
    config_path, cfg = load_config(bargs.config)

    ap = argparse.ArgumentParser()
    ap.add_argument("--config",       default=str(config_path))
    ap.add_argument("--mode",         choices=("offline", "live"),
                    default=cfg.get("mode", "offline"))
    ap.add_argument("--project",      default=cfg.get("project"))
    ap.add_argument("--elf",          default=cfg.get("elf"))
    ap.add_argument("--port",         default=cfg.get("port", "/dev/ttyACM0"))
    ap.add_argument("--baud",         type=int,   default=cfg.get("baud", 115200))
    ap.add_argument("--timeout",      type=float, default=cfg.get("timeout", 120.0))
    ap.add_argument("--clock-mhz",    type=float, default=cfg.get("clock_mhz", 100.0))
    ap.add_argument("--mac-ops",      type=float, default=cfg.get("mac_ops", 3867916.0))
    ap.add_argument("--sample-rate",  type=float, default=cfg.get("sample_rate", 16000.0))
    ap.add_argument("--sample-count", type=float, default=cfg.get("sample_count", 16000.0))
    ap.add_argument("--voltage",      type=float, default=cfg.get("voltage"))
    ap.add_argument("--current-ma",   type=float, default=cfg.get("current_ma"))
    ap.add_argument("--board",        default=cfg.get("board", "FTHR_RevA"))
    ap.add_argument("--variant-name", default=cfg.get("variant_name", "ds_cnn_l_float32"))
    ap.add_argument("--maxim-path",   default=cfg.get("maxim_path"))
    ap.add_argument("--openocd-root", default=cfg.get("openocd_root"))
    ap.add_argument("--no-build",  action="store_true", default=bool(cfg.get("no_build", True)))
    ap.add_argument("--no-flash",  action="store_true", default=bool(cfg.get("no_flash", True)))
    args = ap.parse_args()

    project = Path(args.project).expanduser().resolve()
    elf     = Path(args.elf).expanduser().resolve() if args.elf else (project / "build" / "max78000.elf")

    measure_root = Path(__file__).resolve().parent.parent / "measurements"
    subdir       = "offline_measurements" if args.mode == "offline" else "live_measurements"
    prefix       = f"kws20_{args.variant_name.replace(' ','_')}_{args.mode}"
    out_dir      = measure_root / subdir / datetime.now().strftime(f"{prefix}_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "timestamp":    datetime.now().isoformat(),
        "mode":         args.mode,
        "variant_name": args.variant_name,
        "config":       str(config_path),
        "project":      str(project),
        "port":         args.port,
        "baud":         args.baud,
        "board":        args.board,
        "clock_mhz":    args.clock_mhz,
    }

    env = os.environ.copy()
    if args.maxim_path:
        env["MAXIM_PATH"] = str(Path(args.maxim_path).expanduser().resolve())

    # ── Build ──────────────────────────────────────────────────────────────────
    if not args.no_build:
        make_extra = [f"BOARD={args.board}", "COMPILER=GCC"]
        if args.maxim_path:
            make_extra.append(f"MAXIM_PATH={args.maxim_path}")
        build_text  = run_cmd(["make", "distclean"] + make_extra, project, env, out_dir / "distclean.log")
        build_text += run_cmd(["make", "-j"] + make_extra,        project, env, out_dir / "build.log")
        summary["memory"] = parse_memory_from_build(build_text)

    # ── ELF size ───────────────────────────────────────────────────────────────
    if elf.exists():
        summary["elf"] = str(elf)
        summary["elf_size_bytes"] = elf.stat().st_size
        if "memory" not in summary:
            summary["memory"] = run_size_on_elf(elf, out_dir)

    mem = summary.get("memory")
    if mem and "data_bytes" in mem and "bss_bytes" in mem:
        static = int(mem["data_bytes"]) + int(mem["bss_bytes"])
        summary["static_sram_usage"] = {
            "static_sram_bytes": static,
            "static_sram_kib":   static / 1024.0,
            "data_bytes":        int(mem["data_bytes"]),
            "bss_bytes":         int(mem["bss_bytes"]),
        }

    # ── Flash ──────────────────────────────────────────────────────────────────
    if not args.no_flash:
        if not elf.exists():
            raise RuntimeError(f"ELF not found: {elf}")
        if not args.openocd_root:
            raise RuntimeError("--openocd-root required when flashing")
        ocd_root = Path(args.openocd_root).expanduser().resolve()
        ocd_bin  = ocd_root / "bin/Linux_x86_64/openocd"
        run_cmd([
            str(ocd_bin),
            "-s", str(ocd_root),
            "-f", "interface/cmsis-dap.cfg",
            "-f", "max78000.cfg",
            "-c", f"transport select swd; program {elf} verify reset exit",
        ], project, env, out_dir / "flash.log")

    # ── UART ──────────────────────────────────────────────────────────────────
    print(f"\n[UART] Opening {args.port} @ {args.baud}")
    with serial.Serial(args.port, args.baud, timeout=0.2) as ser:
        time.sleep(1.0)
        ser.reset_input_buffer()
        uart_rows, inf_rows, model_info, acquisition = collect_uart(
            ser, args.baud, args.timeout, out_dir, args.mode
        )

    write_csv(out_dir / "uart_events.csv", uart_rows)
    write_csv(out_dir / "inference_events.csv", inf_rows)

    cnn_us_values = [float(r["cnn_us"]) for r in inf_rows if "cnn_us" in r]
    cycle_values  = [float(r["cycles"])  for r in inf_rows if "cycles"  in r]

    summary["counts"]         = {"num_uart_events": len(uart_rows), "num_inference_events": len(inf_rows)}
    summary["model_info"]     = model_info
    summary["cnn_latency_us"] = stats(cnn_us_values)
    summary["cycles"]         = stats(cycle_values)

    if acquisition:
        try:
            summary["sensor_acquisition"] = {
                "sample_rate_hz":  float(acquisition["sample_rate_hz"]),
                "sample_count":    float(acquisition["sample_count"]),
                "audio_window_ms": float(acquisition["audio_window_ms"]),
                "audio_window_s":  float(acquisition["audio_window_ms"]) / 1000.0,
                "fw_reported":     acquisition,
            }
        except (KeyError, ValueError):
            pass

    add_derived(summary, args)

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nDONE")
    print(f"Output: {out_dir}")
    print(json.dumps({
        "memory":              summary.get("memory"),
        "static_sram_usage":   summary.get("static_sram_usage"),
        "model_info":          summary.get("model_info"),
        "cnn_latency_us":      summary.get("cnn_latency_us"),
        "cnn_latency_ms_avg":  summary.get("cnn_latency_ms_avg"),
        "cycles":              summary.get("cycles"),
        "inferences_per_second": summary.get("inferences_per_second"),
        "mac_ops_per_cycle":   summary.get("mac_ops_per_cycle"),
        "compute_estimate":    summary.get("compute_estimate"),
        "energy_estimate":     summary.get("energy_estimate"),
        "counts":              summary.get("counts"),
    }, indent=2))


if __name__ == "__main__":
    main()
