#!/usr/bin/env python3
"""
Offline measurement script for DS-CNN-L on GAP9 board.

Runs N inferences via gapy (JTAG), collects cycle counts and predicted classes,
computes statistics, saves to a timestamped measurements directory.

MUST run inside the Deeploy Docker container after an initial build:
  cd /app/Deeploy/DeeployTest
  python deeployRunner_gap9.py -t Tests/Models/DSCNNL -s board   # build + first run
  cd /app/Deeploy
  python tools/kws20_measure_gap9_ds_cnn.py [--runs 50]

Outputs (saved to /app/Deeploy/measurements/offline_measurements/<timestamp>/):
  summary.json       -- statistics + metadata
  cycles_raw.csv     -- per-run cycles / latency / predicted class
  run_logs/          -- raw gapy stdout per run
"""

import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ── Defaults ───────────────────────────────────────────────────────────────────

DEFAULT_CONFIG_PATH = Path(__file__).with_name("gap9_ds_cnn_measure_config.json")

LABELS = [
    "down", "go", "left", "no", "off", "on",
    "right", "stop", "up", "yes", "silence", "unknown",
]

# Semihost adds an optional "[cluster, core]" prefix to every printf line.
_SH = r'(?:\[\d+[,\s]+\d+\]\s*)?'
RE_CYCLES = re.compile(_SH + r'Runtime:\s*(\d+)\s*cycles',           re.I)
RE_ERRORS = re.compile(_SH + r'Errors:\s*(\d+)\s+out\s+of\s+(\d+)', re.I)
RE_PRED   = re.compile(_SH + r'Predicted\s+class:\s*(\d+)\s*'
                              r'\(confidence\s+([0-9.eE+\-]+)\)',     re.I)
RE_L2MEM  = re.compile(
    r'(L2[\w_]*):\s+([\d.]+)\s+(KB|B)\s+([\d.]+)\s+(KB|MB)\s+([\d.]+)%')

GAP9_SDK_DEFAULT = "/app/install/gap9-sdk"
BINARY_SUBPATH   = "DeeployTest/TEST_GAP9/build_master/DSCNNL"
WORKDIR_SUBPATH  = "DeeployTest/TEST_GAP9/build_master/board_workdir"


# ── Helpers ────────────────────────────────────────────────────────────────────

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


def _to_bytes(val_str, unit_str):
    v = float(val_str)
    u = unit_str.upper()
    if u == "KB":
        return int(v * 1024)
    if u == "MB":
        return int(v * 1024 * 1024)
    return int(v)


def parse_l2_memory(text):
    """Parse GAP9 linker memory-region summary from cmake/gapy output."""
    mem = {}
    for m in RE_L2MEM.finditer(text):
        region = m.group(1)
        mem[region] = {
            "used_bytes":  _to_bytes(m.group(2), m.group(3)),
            "total_bytes": _to_bytes(m.group(4), m.group(5)),
            "used_pct":    float(m.group(6)),
        }
    return mem


def run_size_on_elf(elf_path, sdk_home):
    """Return section sizes dict using the LLVM or system 'size' tool."""
    candidates = [
        os.path.join(sdk_home, "utils/llvm/bin/llvm-size"),
        "/app/install/llvm/bin/llvm-size",
        "llvm-size",
        "size",
    ]
    for size_bin in candidates:
        try:
            proc = subprocess.run(
                [size_bin, str(elf_path)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, timeout=15,
            )
            if proc.returncode == 0 and proc.stdout:
                m = re.search(
                    r'^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([0-9a-fA-F]+)',
                    proc.stdout, re.M)
                if m:
                    return {
                        "text_bytes": int(m.group(1)),
                        "data_bytes": int(m.group(2)),
                        "bss_bytes":  int(m.group(3)),
                        "dec_bytes":  int(m.group(4)),
                        "size_tool":  size_bin,
                    }
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return {}


def run_inference(gapy_cmd, env, run_idx, log_dir):
    """Execute one gapy board run.

    Returns (cycles, errors_n, tested_n, pred_class, pred_conf) or None on timeout.
    """
    log_path = log_dir / f"run_{run_idx:03d}.log"
    try:
        proc = subprocess.run(
            gapy_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            timeout=180,
        )
        raw = proc.stdout
    except subprocess.TimeoutExpired:
        (log_dir / f"run_{run_idx:03d}_timeout.log").write_text("TIMEOUT\n")
        return None

    log_path.write_text(raw)

    m = RE_CYCLES.search(raw)
    if not m:
        return None  # no "Runtime:" found — treat as failed run

    cycles  = int(m.group(1))
    errors_n = tested_n = pred_class = pred_conf = None

    m = RE_ERRORS.search(raw)
    if m:
        errors_n = int(m.group(1))
        tested_n = int(m.group(2))

    m = RE_PRED.search(raw)
    if m:
        pred_class = int(m.group(1))
        pred_conf  = float(m.group(2))

    return cycles, errors_n, tested_n, pred_class, pred_conf


# ── Main ───────────────────────────────────────────────────────────────────────

def load_config(path):
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return p, {}
    return p, json.loads(p.read_text())


def main():
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    bargs, _ = bootstrap.parse_known_args()
    config_path, cfg = load_config(bargs.config)

    ap = argparse.ArgumentParser(
        description="Measure DS-CNN-L inference on GAP9 board (runs inside Docker container)")
    ap.add_argument("--config",          default=str(config_path))
    ap.add_argument("--runs",            type=int,   default=cfg.get("runs", 50),
                    help="Number of inference runs (default: 50)")
    ap.add_argument("--clock-hz",        type=float, default=cfg.get("clock_hz", 240_000_000.0))
    ap.add_argument("--mac-ops",         type=float, default=cfg.get("mac_ops",  3_824_768.0))
    ap.add_argument("--variant-name",    default=cfg.get("variant_name", "ds_cnn_l_float32"))
    ap.add_argument("--deeploy-root",    default=cfg.get("deeploy_root", "/app/Deeploy"))
    ap.add_argument("--sdk-home",        default=cfg.get("sdk_home",     GAP9_SDK_DEFAULT))
    ap.add_argument("--measurements-root", default=cfg.get("measurements_root"))
    ap.add_argument("--voltage",    type=float, default=cfg.get("voltage_v"),
                    help="Supply voltage in V (e.g. 1.8) — for energy estimate")
    ap.add_argument("--current-ma", type=float, default=cfg.get("current_ma"),
                    help="Average current in mA measured via PPK2 — for energy estimate")
    ap.add_argument("--board",       default=cfg.get("board", "GAP9_EVK"),
                    help="Board name for metadata (default: GAP9_EVK)")
    args = ap.parse_args()

    # ── Paths ──────────────────────────────────────────────────────────────────
    deeploy_root = Path(args.deeploy_root)
    sdk_home     = args.sdk_home or os.environ.get("GAP_SDK_HOME", GAP9_SDK_DEFAULT)
    binary_path  = deeploy_root / BINARY_SUBPATH
    workdir_path = deeploy_root / WORKDIR_SUBPATH
    workdir_path.mkdir(parents=True, exist_ok=True)

    if not binary_path.exists():
        print(f"ERROR: binary not found: {binary_path}")
        print("Run inside the container first:")
        print("  cd /app/Deeploy/DeeployTest")
        print("  python deeployRunner_gap9.py -t Tests/Models/DSCNNL -s board")
        sys.exit(1)

    # ── Output directory ───────────────────────────────────────────────────────
    if args.measurements_root:
        measure_root = Path(args.measurements_root).expanduser().resolve()
    else:
        measure_root = deeploy_root / "measurements"
    prefix  = f"kws20_{args.variant_name.replace(' ', '_')}_offline"
    out_dir = (measure_root / "offline_measurements"
               / datetime.now().strftime(f"{prefix}_%Y%m%d_%H%M%S"))
    run_logs_dir = out_dir / "run_logs"
    run_logs_dir.mkdir(parents=True, exist_ok=True)

    print(f"[GAP9 Measure] Output:  {out_dir}")
    print(f"[GAP9 Measure] Binary:  {binary_path}")
    print(f"[GAP9 Measure] Runs:    {args.runs}")
    print(f"[GAP9 Measure] Clock:   {args.clock_hz/1e6:.0f} MHz")

    # ── ELF size ───────────────────────────────────────────────────────────────
    elf_size = run_size_on_elf(binary_path, sdk_home)
    if elf_size:
        print(f"[GAP9 Measure] ELF:     text={elf_size['text_bytes']:,}  "
              f"data={elf_size['data_bytes']:,}  bss={elf_size['bss_bytes']:,}")

    # ── Try to find L2 memory from existing cmake linker output ───────────────
    l2_memory = {}
    for map_candidate in [
        deeploy_root / "DeeployTest/TEST_GAP9/build_master/DSCNNL.map",
        deeploy_root / "DeeployTest/TEST_GAP9/build_master/memory_report.txt",
    ]:
        if map_candidate.exists():
            try:
                text = map_candidate.read_text(errors="replace")
                l2_memory = parse_l2_memory(text)
                if l2_memory:
                    break
            except Exception:
                pass

    # ── gapy command (same as cmake board_DSCNNL target) ──────────────────────
    gapy_bin = os.path.join(sdk_home, "utils/gapy_v2/bin/gapy")
    gapy_cmd = [
        gapy_bin,
        "--target=gap9.evk",
        "--platform=board",
        "--target-property=boot.flash_device=mram",
        "--target-property=boot.mode=flash",
        f"--target-dir={sdk_home}/utils/gapy_v2/targets",
        f"--openocd-cable={sdk_home}/utils/openocd_tools/tcl/gapuino_ftdi.cfg",
        f"--openocd-script={sdk_home}/utils/openocd_tools/tcl/gap9revb.tcl",
        f"--openocd-tools={sdk_home}/utils/openocd_tools",
        f"--binary={binary_path}",
        f"--work-dir={workdir_path}",
        "run",
        "--py-stack",
    ]
    env = os.environ.copy()
    env.setdefault("GAP_SDK_HOME", sdk_home)

    # ── Inference loop ─────────────────────────────────────────────────────────
    rows              = []
    cycle_values      = []
    latency_us_values = []
    error_runs        = 0
    pred_class_counts = {}

    print(f"\n[GAP9 Measure] Starting {args.runs} inference runs...\n")

    for run_idx in range(1, args.runs + 1):
        print(f"  Run {run_idx:3d}/{args.runs} ...", end=" ", flush=True)
        result = run_inference(gapy_cmd, env, run_idx, run_logs_dir)

        if result is None:
            print("FAILED (timeout or no Runtime: line) — skipping")
            error_runs += 1
            continue

        cycles, errors_n, tested_n, pred_class, pred_conf = result
        lat_us = cycles / args.clock_hz * 1e6
        lat_ms = lat_us / 1000.0
        label  = LABELS[pred_class] if pred_class is not None and pred_class < len(LABELS) else "?"
        err_str = f"{errors_n}/{tested_n}" if errors_n is not None else "?"
        conf_str = f"{pred_conf:.4f}" if pred_conf is not None else "?"

        print(f"{cycles:>12,} cy  {lat_ms:6.1f} ms  "
              f"err={err_str}  class={pred_class}({label})  conf={conf_str}")

        cycle_values.append(cycles)
        latency_us_values.append(lat_us)
        if pred_class is not None:
            pred_class_counts[pred_class] = pred_class_counts.get(pred_class, 0) + 1

        rows.append({
            "run":        run_idx,
            "cycles":     cycles,
            "latency_us": round(lat_us, 2),
            "latency_ms": round(lat_ms, 3),
            "errors":     errors_n,
            "tested":     tested_n,
            "pred_class": pred_class,
            "pred_label": label,
            "pred_conf":  round(pred_conf, 4) if pred_conf is not None else None,
        })

    # ── Summary ────────────────────────────────────────────────────────────────
    n_ok = len(rows)
    print(f"\n[GAP9 Measure] {n_ok}/{args.runs} runs succeeded  "
          f"({error_runs} failed/timeout)")

    cycle_stats = stats(cycle_values)
    lat_stats   = stats(latency_us_values)

    # ── Memory sections (equivalent to MAX78000 "memory" key) ─────────────────
    # On GAP9 NOFLASH mode: text = code in L2, data = compiled weights in L2,
    # bss = runtime buffers/stack in L2. No separate flash — everything is L2.
    memory = elf_size if elf_size else None
    static_sram_usage = None
    if memory:
        static_bytes = memory.get("data_bytes", 0) + memory.get("bss_bytes", 0)
        static_sram_usage = {
            "static_sram_bytes": static_bytes,
            "static_sram_kib":   static_bytes / 1024.0,
            "data_bytes":        memory.get("data_bytes", 0),
            "bss_bytes":         memory.get("bss_bytes", 0),
            "note": "GAP9 NOFLASH: data section contains compiled model weights; equivalent metric to MCU SRAM usage",
        }

    # ── L2 total usage (GAP9-specific) ────────────────────────────────────────
    if not l2_memory and memory:
        total_l2 = memory.get("dec_bytes", 0)
        l2_memory = {
            "total_l2_bytes": total_l2,
            "total_l2_kib":   total_l2 / 1024.0,
            "code_bytes":     memory.get("text_bytes", 0),
            "data_bytes":     memory.get("data_bytes", 0),
            "bss_bytes":      memory.get("bss_bytes", 0),
            "source":         "estimated from ELF sections (llvm-size)",
        }

    summary = {
        "timestamp":    datetime.now().isoformat(),
        "mode":         "offline",
        "platform":     "GAP9",
        "board":        args.board,
        "variant_name": args.variant_name,
        "config":       str(config_path),
        "binary":       str(binary_path),
        "clock_mhz":    args.clock_hz / 1e6,
        "clock_hz":     args.clock_hz,
        "counts": {
            "num_inference_events": n_ok,
            "n_runs_requested":     args.runs,
            "n_runs_failed":        error_runs,
        },
        "memory":             memory,
        "static_sram_usage":  static_sram_usage,
        "l2_memory":          l2_memory,
        "cycles":             cycle_stats,
        "cnn_latency_us":     lat_stats,          # same key as MAX78000
        "latency_us":         lat_stats,          # GAP9 alias
        "cnn_latency_ms_avg": (lat_stats["avg"] / 1000.0) if lat_stats else None,
        "latency_ms_avg":     (lat_stats["avg"] / 1000.0) if lat_stats else None,
        "labels":             LABELS,
        "pred_class_distribution": {
            (LABELS[k] if k < len(LABELS) else str(k)): v
            for k, v in sorted(pred_class_counts.items())
        },
    }

    # Derived: throughput, efficiency, compute estimate, energy
    if cycle_stats and lat_stats:
        avg_cycles = cycle_stats["avg"]
        avg_s      = avg_cycles / args.clock_hz
        summary["inferences_per_second"] = 1.0 / avg_s

        if args.mac_ops:
            summary["mac_ops_per_cycle"] = args.mac_ops / avg_cycles
            summary["compute_estimate"]  = {
                "mac_ops_per_inference": args.mac_ops,
                "mac_ops_per_second":    args.mac_ops / avg_s,
                "gops":                  args.mac_ops / avg_s / 1e9,
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
                **({"mac_ops_per_joule": args.mac_ops / energy_j} if args.mac_ops else {}),
            }

    # ── Save files ─────────────────────────────────────────────────────────────
    write_csv(out_dir / "cycles_raw.csv", rows)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[GAP9 Measure] DONE")
    print(f"Output: {out_dir}")
    print(json.dumps({
        "memory":                  summary.get("memory"),
        "static_sram_usage":       summary.get("static_sram_usage"),
        "l2_memory":               summary.get("l2_memory"),
        "cycles":                  summary.get("cycles"),
        "cnn_latency_us":          summary.get("cnn_latency_us"),
        "cnn_latency_ms_avg":      summary.get("cnn_latency_ms_avg"),
        "inferences_per_second":   summary.get("inferences_per_second"),
        "mac_ops_per_cycle":       summary.get("mac_ops_per_cycle"),
        "compute_estimate":        summary.get("compute_estimate"),
        "energy_estimate":         summary.get("energy_estimate"),
        "pred_class_distribution": summary.get("pred_class_distribution"),
        "counts":                  summary.get("counts"),
    }, indent=2))


if __name__ == "__main__":
    main()
