#!/usr/bin/env python3
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


DEFAULT_CONFIG_PATH = Path(__file__).with_name("kws20_measure_max78000_config.json")

KEYWORDS = [
    "up", "down", "left", "right", "stop", "go",
    "yes", "no", "on", "off",
    "one", "two", "three", "four", "five",
    "six", "seven", "eight", "nine", "zero",
    "unknown", "_unknown_", "silence", "_silence_",
]


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
    nbytes = len(line.encode("utf-8", errors="replace"))
    return (nbytes * 10) / baud


def parse_size_output(text):
    out = {}
    m = re.search(r"^\s*([0-9]+)\s+([0-9]+)\s+([0-9]+)\s+([0-9]+)\s+([0-9a-fA-F]+)\s+.+$", text, re.M)
    if m:
        out["text_bytes"] = int(m.group(1))
        out["data_bytes"] = int(m.group(2))
        out["bss_bytes"] = int(m.group(3))
        out["dec_bytes"] = int(m.group(4))
        out["hex"] = m.group(5)
    return out


def run_size_on_elf(elf_path, out_dir):
    text = run_cmd(["arm-none-eabi-size", str(elf_path)], elf_path.parent, os.environ.copy(), out_dir / "size.log")
    return parse_size_output(text)


def parse_memory_from_build(text):
    out = parse_size_output(text)
    m = re.search(r"FLASH:\s+([0-9]+)\s+B\s+512\s+KB\s+([0-9.]+)%", text)
    if m:
        out["flash_bytes"] = int(m.group(1))
        out["flash_percent"] = float(m.group(2))
    m = re.search(r"SRAM:\s+([0-9]+)\s+B\s+128\s+KB\s+([0-9.]+)%", text)
    if m:
        out["sram_bytes"] = int(m.group(1))
        out["sram_percent"] = float(m.group(2))
    return out


def parse_cnn_time(line):
    m = re.search(r"CNN Time:\s*([0-9]+)\s*us", line)
    if m:
        return int(m.group(1))
    m = re.search(r"BENCH,.*cnn_us=([0-9]+)", line)
    if m:
        return int(m.group(1))
    return None


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


def parse_prediction(line):
    s = line.strip().lower()
    if "cnn time" in s or s.startswith("bench,"):
        return None

    likely = any(x in s for x in [
        "classification", "class", "keyword", "detected",
        "result", "word", "confidence", "prob", "probability"
    ])

    has_percent = bool(re.search(r"\b[0-9]+(?:\.[0-9]+)?\s*%", s))
    if not likely and not has_percent:
        return None

    found = []
    for kw in KEYWORDS:
        if re.search(rf"\b{re.escape(kw)}\b", s):
            found.append(kw.replace("_", ""))

    found = sorted(set(found))
    if len(found) != 1:
        return None

    confidence = None
    m = re.search(r"(confidence|prob|probability)\s*[:=]\s*([0-9.]+)", s)
    if m:
        confidence = float(m.group(2))
    else:
        m = re.search(r"\b([0-9]+(?:\.[0-9]+)?)\s*%", s)
        if m:
            confidence = float(m.group(1))

    return {
        "prediction": found[0],
        "confidence": confidence,
        "raw": line.strip(),
    }


def load_config(path):
    cfg_path = Path(path).expanduser().resolve()
    if not cfg_path.exists():
        raise RuntimeError(f"Config file not found: {cfg_path}")
    data = json.loads(cfg_path.read_text())
    if not isinstance(data, dict):
        raise RuntimeError(f"Config file must contain a JSON object: {cfg_path}")
    return cfg_path, data


def slugify_variant(value):
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unnamed"


def collect_uart_live(ser, baud, timeout_s, out_dir):
    raw_path = out_dir / "serial_raw.log"
    uart_rows = []
    inf_rows = []
    pred_rows = []
    model_info = None
    acquisition = None
    saw_bench_inference = False
    start = time.time()
    print(f"[UART] capturing for up to {timeout_s:.0f} s -- press Ctrl+C to stop early "
          f"(summary will still be written)")

    with open(raw_path, "w") as raw:
        try:
            while time.time() - start < timeout_s:
                data = ser.readline()
                if not data:
                    continue
                now = time.time()
                elapsed = now - start
                iso = datetime.fromtimestamp(now).isoformat(timespec="milliseconds")
                line = data.decode("utf-8", errors="replace").strip()
                print(line)
                raw.write(f"{iso},elapsed_s={elapsed:.6f},{line}\n")
                raw.flush()

                uart_rows.append({
                    "time_iso": iso,
                    "elapsed_s": elapsed,
                    "bytes": len(line.encode("utf-8", errors="replace")),
                    "uart_tx_time_s_est": estimate_uart_tx_time_s(line, baud),
                    "raw": line,
                })

                bench = parse_bench_fields(line)
                if bench:
                    event = bench.get("event")
                    if event == "model_info":
                        model_info = bench
                    elif event == "acquisition":
                        acquisition = bench
                    elif event == "inference":
                        saw_bench_inference = True
                        row = {
                            "time_iso": iso,
                            "elapsed_s": elapsed,
                            "raw": line,
                        }
                        for key in ("run", "cnn_us", "timer_ticks", "cycles", "pred_idx",
                                    "pred_reported_idx", "accepted",
                                    "confidence_x10", "sample_counter", "word_counter"):
                            if key in bench:
                                row[key] = bench[key]
                        inf_rows.append(row)
                        pred_idx = bench.get("pred_reported_idx", bench.get("pred_idx", ""))
                        pred_label = pred_idx
                        if pred_idx.lstrip("-").isdigit():
                            idx = int(pred_idx)
                            if 0 <= idx < len(KEYWORDS):
                                pred_label = KEYWORDS[idx]
                        pred = {
                            "prediction": pred_label,
                            "confidence": (float(bench["confidence_x10"]) / 10.0)
                            if "confidence_x10" in bench else None,
                            "raw": line.strip(),
                            "time_iso": iso,
                            "elapsed_s": elapsed,
                        }
                        pred_rows.append(pred)
                        continue

                cnn = parse_cnn_time(line)
                if cnn is not None and not saw_bench_inference:
                    inf_rows.append({
                        "time_iso": iso,
                        "elapsed_s": elapsed,
                        "cnn_us": cnn,
                        "raw": line,
                    })

                pred = parse_prediction(line)
                if pred and not saw_bench_inference:
                    pred.update({"time_iso": iso, "elapsed_s": elapsed})
                    pred_rows.append(pred)
        except KeyboardInterrupt:
            print(f"\n[UART] Ctrl+C received after {time.time() - start:.1f} s -- "
                  f"finishing up and writing summary...")

    return uart_rows, inf_rows, pred_rows, model_info, acquisition


def add_derived(summary, args):
    cnn = summary.get("cnn_latency_us")
    if not cnn:
        return

    avg_us = cnn["avg"]
    avg_s = avg_us * 1e-6
    summary["cnn_latency_ms_avg"] = avg_us / 1000.0
    summary["inferences_per_second_from_cnn_time"] = 1.0 / avg_s if avg_s > 0 else None

    if args.mode == "live":
        if "sensor_acquisition" not in summary:
            summary["sensor_acquisition"] = {
                "sample_rate_hz": args.sample_rate,
                "sample_count": args.sample_count,
                "audio_window_s": args.sample_count / args.sample_rate,
                "audio_window_ms": (args.sample_count / args.sample_rate) * 1000.0,
            }
        summary["estimated_end_to_end_lower_bound_ms"] = (
            summary["sensor_acquisition"]["audio_window_ms"] + (avg_us / 1000.0)
        )

    cycles = summary.get("cycles")
    if cycles:
        summary["mac_ops_per_cycle"] = None
        if args.mac_ops and cycles.get("avg"):
            summary["mac_ops_per_cycle"] = args.mac_ops / cycles["avg"]

    if args.mac_ops:
        summary["compute_estimate"] = {
            "mac_ops_per_inference": args.mac_ops,
            "mac_ops_per_second": args.mac_ops / avg_s,
            "mops_per_second": (args.mac_ops / avg_s) / 1e6,
        }

    if args.voltage and args.current_ma:
        power_w = args.voltage * args.current_ma / 1000.0
        energy_j = power_w * avg_s
        summary["energy_estimate"] = {
            "voltage_v": args.voltage,
            "current_ma": args.current_ma,
            "power_w": power_w,
            "energy_per_inference_j": energy_j,
            "energy_per_inference_uj": energy_j * 1e6,
        }
        if args.mac_ops:
            summary["energy_estimate"]["mac_ops_per_joule"] = args.mac_ops / energy_j
            summary["energy_estimate"]["mac_ops_per_second_per_watt"] = (args.mac_ops / avg_s) / power_w


def main():
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    bootstrap_args, _ = bootstrap.parse_known_args()
    config_path, config = load_config(bootstrap_args.config)

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(config_path))
    ap.add_argument("--mode", choices=("live", "offline"), default=config.get("mode", "live"))
    ap.add_argument("--project", default=config.get("project", str(Path(__file__).resolve().parent.parent / "kws20_demo")))
    ap.add_argument("--elf", default=config.get("elf"))
    ap.add_argument("--port", default=config.get("port", "/dev/ttyACM0"))
    ap.add_argument("--baud", type=int, default=config.get("baud", 115200))
    ap.add_argument("--timeout", type=float, default=config.get("timeout", 60.0))
    ap.add_argument("--clock-mhz", type=float, default=config.get("clock_mhz", 100.0))
    ap.add_argument("--mac-ops", type=float, default=config.get("mac_ops"))
    ap.add_argument("--sample-rate", type=float, default=config.get("sample_rate", 16000.0))
    ap.add_argument("--sample-count", type=float, default=config.get("sample_count", 16384.0))
    ap.add_argument("--voltage", type=float, default=config.get("voltage"))
    ap.add_argument("--current-ma", type=float, default=config.get("current_ma"))
    ap.add_argument("--board", default=config.get("board", "FTHR_RevA"))
    ap.add_argument("--variant-name", default=config.get("variant_name"))
    ap.add_argument("--maxim-path", default=config.get("maxim_path"))
    ap.add_argument("--openocd-root", default=config.get("openocd_root"))
    ap.add_argument("--no-build", action="store_true", default=bool(config.get("no_build", True)))
    ap.add_argument("--no-flash", action="store_true", default=bool(config.get("no_flash", True)))
    args = ap.parse_args()
    project = Path(args.project).expanduser().resolve()
    variant_name = args.variant_name or project.name
    variant_slug = slugify_variant(variant_name)
    measure_subdir = "live_measurements" if args.mode == "live" else "offline_measurements"
    prefix = f"kws20_{variant_slug}_{args.mode}"
    repo_measure_root = Path(__file__).resolve().parent.parent / "measurements" / measure_subdir
    out_dir = repo_measure_root / datetime.now().strftime(f"{prefix}_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "mode": args.mode,
        "variant_name": variant_name,
        "config": str(Path(args.config).expanduser().resolve()),
        "project": str(project),
        "port": args.port,
        "baud": args.baud,
        "board": args.board,
    }

    env = os.environ.copy()
    if args.maxim_path:
        env["MAXIM_PATH"] = str(Path(args.maxim_path).expanduser().resolve())

    if not args.no_build:
        build_text = run_cmd(["make", "distclean"], project, env, out_dir / "distclean.log")
        build_text += run_cmd(["make", "-j", f"BOARD={args.board}"], project, env, out_dir / "build.log")
        summary["memory"] = parse_memory_from_build(build_text)

    elf = Path(args.elf).expanduser().resolve() if args.elf else (project / "build" / "max78000.elf")
    if elf.exists():
        summary["elf"] = str(elf)
        summary["elf_size_bytes"] = elf.stat().st_size
        if "memory" not in summary:
            summary["memory"] = run_size_on_elf(elf, out_dir)

    memory = summary.get("memory")
    if memory and ("data_bytes" in memory) and ("bss_bytes" in memory):
        static_sram_bytes = int(memory["data_bytes"]) + int(memory["bss_bytes"])
        summary["static_sram_usage"] = {
            "static_sram_bytes": static_sram_bytes,
            "static_sram_kib": static_sram_bytes / 1024.0,
            "data_bytes": int(memory["data_bytes"]),
            "bss_bytes": int(memory["bss_bytes"]),
        }

    if not args.no_flash:
        if not elf.exists():
            raise RuntimeError(f"ELF not found: {elf}")
        if not args.openocd_root:
            raise RuntimeError("openocd_root required when flashing is enabled")
        ocd_root = Path(args.openocd_root).expanduser().resolve()
        ocd_bin = ocd_root / "bin/Linux_x86_64/openocd"
        run_cmd([
            str(ocd_bin),
            "-s", str(ocd_root),
            "-f", "interface/cmsis-dap.cfg",
            "-f", "max78000.cfg",
            "-c", f"transport select swd; program {elf} verify reset exit",
        ], project, env, out_dir / "flash.log")

    print(f"\n[UART] Opening {args.port} @ {args.baud}")
    with serial.Serial(args.port, args.baud, timeout=0.2) as ser:
        time.sleep(1.0)
        ser.reset_input_buffer()
        uart_rows, inf_rows, pred_rows, model_info, acquisition = collect_uart_live(
            ser, args.baud, args.timeout, out_dir
        )

    write_csv(out_dir / "uart_events.csv", uart_rows)
    write_csv(out_dir / "inference_events.csv", inf_rows)
    write_csv(out_dir / "prediction_events.csv", pred_rows)

    cnn_us_values = [float(r["cnn_us"]) for r in inf_rows if "cnn_us" in r]
    cycle_values = [float(r["cycles"]) for r in inf_rows if "cycles" in r]
    uart_tx_values = [r["uart_tx_time_s_est"] for r in uart_rows]
    pred_hist = {}
    for row in pred_rows:
        pred = row.get("prediction", "")
        if pred:
            pred_hist[pred] = pred_hist.get(pred, 0) + 1

    summary["counts"] = {
        "num_uart_events": len(uart_rows),
        "num_inference_events": len(inf_rows),
        "num_prediction_events": len(pred_rows),
    }
    summary["prediction_histogram"] = pred_hist
    summary["model_info"] = model_info
    summary["cnn_latency_us"] = stats(cnn_us_values)
    summary["cycles"] = stats(cycle_values)
    summary["uart_transmission_time_estimate_s"] = stats(uart_tx_values)

    if acquisition:
        try:
            summary["sensor_acquisition"] = {
                "sample_rate_hz": float(acquisition["sample_rate_hz"]),
                "sample_count": float(acquisition["sample_count"]),
                "audio_window_s": float(acquisition["audio_window_ms"]) / 1000.0,
                "audio_window_ms": float(acquisition["audio_window_ms"]),
                "fw_reported": acquisition,
            }
        except (KeyError, ValueError):
            pass

    add_derived(summary, args)

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nDONE")
    print(f"Output: {out_dir}")
    print(json.dumps({
        "memory": summary.get("memory"),
        "static_sram_usage": summary.get("static_sram_usage"),
        "elf_size_bytes": summary.get("elf_size_bytes"),
        "model_info": summary.get("model_info"),
        "cnn_latency_us": summary.get("cnn_latency_us"),
        "cycles": summary.get("cycles"),
        "sensor_acquisition": summary.get("sensor_acquisition"),
        "estimated_end_to_end_lower_bound_ms": summary.get("estimated_end_to_end_lower_bound_ms"),
        "mac_ops_per_cycle": summary.get("mac_ops_per_cycle"),
        "compute_estimate": summary.get("compute_estimate"),
        "energy_estimate": summary.get("energy_estimate"),
        "counts": summary.get("counts"),
        "prediction_histogram": summary.get("prediction_histogram"),
    }, indent=2))


if __name__ == "__main__":
    main()
