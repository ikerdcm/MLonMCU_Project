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

def parse_memory(build_text):
    out = {}
    m = re.search(r"FLASH:\s+([0-9]+)\s+B\s+512\s+KB\s+([0-9.]+)%", build_text)
    if m:
        out["flash_bytes"] = int(m.group(1))
        out["flash_percent"] = float(m.group(2))
    m = re.search(r"SRAM:\s+([0-9]+)\s+B\s+128\s+KB\s+([0-9.]+)%", build_text)
    if m:
        out["sram_bytes"] = int(m.group(1))
        out["sram_percent"] = float(m.group(2))
    m = re.search(
        r"\n\s*text\s+data\s+bss\s+dec\s+hex\s+filename\s*\n\s*"
        r"([0-9]+)\s+([0-9]+)\s+([0-9]+)\s+([0-9]+)\s+([0-9a-fA-F]+)",
        build_text,
    )
    if m:
        out["text_bytes"] = int(m.group(1))
        out["data_bytes"] = int(m.group(2))
        out["bss_bytes"] = int(m.group(3))
        out["dec_bytes"] = int(m.group(4))
        out["hex"] = m.group(5)
    return out

def parse_cnn_time(line):
    m = re.search(r"CNN Time:\s*([0-9]+)\s*us", line)
    if m:
        return int(m.group(1))
    m = re.search(r"BENCH,.*cnn_us=([0-9]+)", line)
    if m:
        return int(m.group(1))
    return None

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
            clean = kw.replace("_", "")
            found.append(clean)

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

def estimate_uart_tx_time_s(line, baud):
    # 8N1 UART: around 10 bits per byte.
    nbytes = len(line.encode("utf-8", errors="replace"))
    return (nbytes * 10) / baud

def accelerator_evidence(project):
    files = [project / "main.c", project / "cnn.c", project / "cnn.h"]
    patterns = [
        "cnn_init", "cnn_load_weights", "cnn_load_bias",
        "cnn_start", "cnn_unload", "MXC_CNN", "CNN_COMPLETE",
    ]
    evidence = {}
    for f in files:
        if not f.exists():
            continue
        text = f.read_text(errors="replace")
        evidence[str(f.name)] = {p: text.count(p) for p in patterns}
    return evidence

def analyze_checkpoint(path):
    path = Path(path).expanduser()
    if not path.exists():
        return {"available": False, "error": f"checkpoint not found: {path}"}
    try:
        import torch
    except Exception as e:
        return {"available": False, "error": f"torch import failed: {e}"}

    out = {"available": True, "path": str(path)}
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
        out["checkpoint_keys"] = list(ckpt.keys())
        extras = ckpt.get("extras", {})
        if isinstance(extras, dict):
            out["extras"] = {k: str(v) for k, v in extras.items()}
    elif isinstance(ckpt, dict):
        sd = ckpt
        out["checkpoint_keys"] = list(ckpt.keys())
    else:
        return {"available": False, "error": "unknown checkpoint format"}

    total_tensors = 0
    total_numel = 0
    total_bytes_real_dtype = 0
    dtype_hist = {}
    layer_rows = []

    for name, tensor in sd.items():
        if not hasattr(tensor, "numel"):
            continue
        n = int(tensor.numel())
        b = int(tensor.element_size() * tensor.numel())
        dtype = str(tensor.dtype)
        total_tensors += 1
        total_numel += n
        total_bytes_real_dtype += b
        dtype_hist[dtype] = dtype_hist.get(dtype, 0) + n
        layer_rows.append({
            "name": name,
            "shape": list(tensor.shape),
            "numel": n,
            "dtype": dtype,
            "bytes_real_dtype": b,
        })

    out["num_tensors"] = total_tensors
    out["total_tensor_elements"] = total_numel
    out["memory_bytes_real_dtype"] = total_bytes_real_dtype
    out["memory_bytes_if_int8"] = total_numel
    out["memory_bytes_if_int32"] = total_numel * 4
    out["dtype_histogram_by_elements"] = dtype_hist
    out["first_layers"] = layer_rows[:30]
    out["quantized_checkpoint_guess"] = "-q.pth" in path.name or "qat" in path.name.lower()
    return out

def load_manifest(path):
    rows = []
    with open(Path(path).expanduser(), newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "label": r["label"].strip().lower().replace("_", ""),
                "wav": str(Path(r["wav"]).expanduser()),
            })
    return rows

def make_confusion(trials):
    labels = sorted(set([r["expected"] for r in trials] + [r["predicted"] for r in trials if r["predicted"]]))
    matrix = {a: {b: 0 for b in labels} for a in labels}
    for r in trials:
        exp = r["expected"]
        pred = r["predicted"] if r["predicted"] else "none"
        if pred not in labels:
            for row in matrix.values():
                row[pred] = 0
            labels.append(pred)
            matrix[pred] = {b: 0 for b in labels}
        matrix[exp][pred] += 1
    rows = []
    for exp in sorted(matrix.keys()):
        row = {"expected": exp}
        row.update(matrix[exp])
        rows.append(row)
    return rows

def summarize_accuracy(trials):
    if not trials:
        return None
    total = len(trials)
    correct = sum(int(r["correct"]) for r in trials)
    per_label = {}
    for r in trials:
        lab = r["expected"]
        per_label.setdefault(lab, {"total": 0, "correct": 0})
        per_label[lab]["total"] += 1
        per_label[lab]["correct"] += int(r["correct"])
    for lab, d in per_label.items():
        d["accuracy"] = d["correct"] / d["total"] if d["total"] else 0.0

    false_alarm_trials = [
        r for r in trials
        if r["expected"] in ["silence", "unknown"] and r["predicted"] not in ["", "silence", "unknown"]
    ]
    miss_trials = [
        r for r in trials
        if r["expected"] not in ["silence", "unknown"] and r["predicted"] != r["expected"]
    ]

    return {
        "total_trials": total,
        "correct_trials": correct,
        "accuracy": correct / total if total else 0.0,
        "per_label": per_label,
        "false_alarm_count_on_silence_or_unknown": len(false_alarm_trials),
        "miss_count_on_keywords": len(miss_trials),
    }

def add_derived(summary, cnn_us_values, args):
    if not cnn_us_values:
        return
    s = stats(cnn_us_values)
    summary["cnn_latency_us"] = s
    avg_s = s["avg"] * 1e-6
    summary["cnn_latency_ms_avg"] = s["avg"] / 1000.0
    summary["inferences_per_second_from_cnn_time"] = 1.0 / avg_s if avg_s > 0 else None

    if args.sample_rate and args.sample_count:
        acq_s = args.sample_count / args.sample_rate
        summary["sensor_acquisition"] = {
            "sample_rate_hz": args.sample_rate,
            "sample_count": args.sample_count,
            "audio_window_s": acq_s,
            "audio_window_ms": acq_s * 1000.0,
        }
        summary["estimated_end_to_end_lower_bound_ms"] = acq_s * 1000.0 + s["avg"] / 1000.0

    if args.clock_mhz:
        cycles = avg_s * args.clock_mhz * 1e6
        summary["cycles_estimate"] = {
            "clock_mhz": args.clock_mhz,
            "cycles_per_inference_avg": cycles,
        }
        if args.mac_ops:
            summary["cycles_estimate"]["mac_ops_per_cycle"] = args.mac_ops / cycles

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

def collect_uart_live(ser, args, out_dir):
    raw_path = out_dir / "serial_raw.log"
    inf_rows = []
    pred_rows = []
    uart_rows = []
    start = time.time()

    with open(raw_path, "w") as raw:
        while time.time() - start < args.duration:
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
                "uart_tx_time_s_est": estimate_uart_tx_time_s(line, args.baud),
                "raw": line,
            })

            cnn = parse_cnn_time(line)
            if cnn is not None:
                inf_rows.append({
                    "time_iso": iso,
                    "elapsed_s": elapsed,
                    "cnn_us": cnn,
                    "raw": line,
                })

            pred = parse_prediction(line)
            if pred:
                pred.update({"time_iso": iso, "elapsed_s": elapsed})
                pred_rows.append(pred)

    return inf_rows, pred_rows, uart_rows

def collect_playback(ser, args, out_dir):
    manifest = load_manifest(args.manifest)
    if args.shuffle:
        import random
        random.shuffle(manifest)

    raw_path = out_dir / "serial_raw_playback.log"
    trials = []
    all_inf = []
    all_pred = []
    all_uart = []
    start_global = time.time()

    with open(raw_path, "w") as raw:
        time.sleep(1.0)
        ser.reset_input_buffer()

        for idx, item in enumerate(manifest, start=1):
            expected = item["label"]
            wav = item["wav"]
            print("\n========================================")
            print(f"Trial {idx}/{len(manifest)} expected={expected}")
            print(wav)

            time.sleep(args.pre_delay)
            play_start = time.time()
            play_start_iso = datetime.fromtimestamp(play_start).isoformat(timespec="milliseconds")

            if expected == "silence":
                time.sleep(args.silence_duration)
            else:
                subprocess.run([args.player, wav], check=True)

            trial_inf = []
            trial_pred = []
            first_prediction_latency_s = None

            response_start = time.time()
            while time.time() - response_start < args.response_window:
                data = ser.readline()
                if not data:
                    continue

                now = time.time()
                elapsed = now - start_global
                iso = datetime.fromtimestamp(now).isoformat(timespec="milliseconds")
                line = data.decode("utf-8", errors="replace").strip()

                print(line)
                raw.write(f"{iso},elapsed_s={elapsed:.6f},{line}\n")
                raw.flush()

                all_uart.append({
                    "time_iso": iso,
                    "elapsed_s": elapsed,
                    "bytes": len(line.encode("utf-8", errors="replace")),
                    "uart_tx_time_s_est": estimate_uart_tx_time_s(line, args.baud),
                    "raw": line,
                })

                cnn = parse_cnn_time(line)
                if cnn is not None:
                    row = {"time_iso": iso, "elapsed_s": elapsed, "cnn_us": cnn, "raw": line}
                    trial_inf.append(row)
                    all_inf.append(row)

                pred = parse_prediction(line)
                if pred:
                    if first_prediction_latency_s is None:
                        first_prediction_latency_s = now - play_start
                    pred.update({"time_iso": iso, "elapsed_s": elapsed})
                    trial_pred.append(pred)
                    all_pred.append(pred)

            predicted = trial_pred[-1]["prediction"] if trial_pred else ""
            correct = int(predicted == expected)
            cnn_vals = [r["cnn_us"] for r in trial_inf]

            trials.append({
                "trial": idx,
                "expected": expected,
                "predicted": predicted,
                "correct": correct,
                "wav": wav,
                "play_start_iso": play_start_iso,
                "num_inferences": len(trial_inf),
                "cnn_us_avg_in_trial": sum(cnn_vals) / len(cnn_vals) if cnn_vals else "",
                "host_e2e_latency_s_first_prediction": first_prediction_latency_s if first_prediction_latency_s is not None else "",
            })

            time.sleep(args.post_delay)

    return all_inf, all_pred, all_uart, trials

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--mode", choices=["live", "playback", "all"], default="live")
    ap.add_argument("--project", default=str(Path.home() / "max78000/ai8x-synthesis/sdk/Examples/MAX78000/CNN/kws20_demo"))
    ap.add_argument("--board", default="FTHR_RevA")
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--duration", type=int, default=120)
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--no-flash", action="store_true")

    ap.add_argument("--checkpoint", default=str(Path.home() / "max78000/ai8x-synthesis/trained/ai85-kws20_v3-qat8-q.pth.tar"))
    ap.add_argument("--mac-ops", type=float, default=None)
    ap.add_argument("--clock-mhz", type=float, default=100.0)
    ap.add_argument("--voltage", type=float, default=None)
    ap.add_argument("--current-ma", type=float, default=None)

    ap.add_argument("--sample-rate", type=float, default=16000.0)
    ap.add_argument("--sample-count", type=float, default=16384.0)

    ap.add_argument("--manifest", default=None)
    ap.add_argument("--response-window", type=float, default=2.5)
    ap.add_argument("--pre-delay", type=float, default=0.7)
    ap.add_argument("--post-delay", type=float, default=0.8)
    ap.add_argument("--silence-duration", type=float, default=1.0)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--player", default="aplay")

    args = ap.parse_args()

    project = Path(args.project).expanduser().resolve()
    out_dir = Path.home() / "max78000/measurements" / datetime.now().strftime("kws20_all_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    maxim_path = Path.home() / "max78000/ai8x-synthesis/sdk"
    ocd_root = Path.home() / "max78000/ai8x-synthesis/openocd"
    ocd_bin = ocd_root / "bin/Linux_x86_64/openocd"

    env = os.environ.copy()
    env["MAXIM_PATH"] = str(maxim_path)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "mode": args.mode,
        "project": str(project),
        "board": args.board,
        "port": args.port,
        "baud": args.baud,
        "metrics_coverage": {
            "hardware": [
                "flash_bytes", "sram_bytes", "elf_size_bytes",
                "cnn_latency_us", "estimated_audio_acquisition_time",
                "estimated_end_to_end_lower_bound",
                "uart_tx_time_estimate",
                "mac_ops_per_cycle_if_mac_ops_given",
                "energy_if_voltage_current_given",
            ],
            "network": [
                "checkpoint_tensor_count", "checkpoint_parameter_elements",
                "weight_memory_real_dtype", "weight_memory_if_int8",
                "quantized_checkpoint_guess",
                "mac_ops_if_given",
            ],
            "application": [
                "prediction_histogram", "playback_accuracy_if_manifest_given",
                "confusion_matrix", "false_alarm_count_on_silence_or_unknown",
                "host_e2e_latency_first_prediction",
            ],
        },
    }

    summary["accelerator_evidence"] = accelerator_evidence(project)
    summary["checkpoint_analysis"] = analyze_checkpoint(args.checkpoint)

    if not args.no_build:
        build_text = ""
        build_text += run_cmd(["make", "distclean"], project, env, out_dir / "distclean.log")
        build_text += run_cmd(["make", "-j", f"BOARD={args.board}"], project, env, out_dir / "build.log")
        summary["memory"] = parse_memory(build_text)

    elf = project / "build/max78000.elf"
    if elf.exists():
        summary["elf"] = str(elf)
        summary["elf_size_bytes"] = elf.stat().st_size

    if not args.no_flash:
        if not elf.exists():
            raise RuntimeError(f"ELF not found: {elf}")
        run_cmd([
            str(ocd_bin),
            "-s", str(ocd_root),
            "-f", "interface/cmsis-dap.cfg",
            "-f", "max78000.cfg",
            "-c", f"transport select swd; program {elf} verify reset exit",
        ], project, env, out_dir / "flash.log")

    all_inf = []
    all_pred = []
    all_uart = []
    trials = []

    print(f"\n[UART] Opening {args.port} @ {args.baud}")
    with serial.Serial(args.port, args.baud, timeout=0.2) as ser:
        if args.mode in ["live", "all"]:
            inf, pred, uart = collect_uart_live(ser, args, out_dir)
            all_inf.extend(inf)
            all_pred.extend(pred)
            all_uart.extend(uart)

        if args.mode in ["playback", "all"]:
            if not args.manifest:
                print("WARNING: playback skipped because --manifest was not given")
            else:
                inf, pred, uart, trials = collect_playback(ser, args, out_dir)
                all_inf.extend(inf)
                all_pred.extend(pred)
                all_uart.extend(uart)

    write_csv(out_dir / "inference_events.csv", all_inf)
    write_csv(out_dir / "prediction_events.csv", all_pred)
    write_csv(out_dir / "uart_events.csv", all_uart)
    write_csv(out_dir / "accuracy_trials.csv", trials)
    write_csv(out_dir / "confusion_matrix.csv", make_confusion(trials) if trials else [])

    cnn_values = [r["cnn_us"] for r in all_inf if "cnn_us" in r]
    add_derived(summary, cnn_values, args)

    summary["counts"] = {
        "num_inference_events": len(all_inf),
        "num_prediction_events": len(all_pred),
        "num_uart_events": len(all_uart),
        "num_accuracy_trials": len(trials),
    }

    pred_hist = {}
    for r in all_pred:
        p = r.get("prediction", "")
        if p:
            pred_hist[p] = pred_hist.get(p, 0) + 1
    summary["prediction_histogram"] = pred_hist

    uart_tx_values = [r["uart_tx_time_s_est"] for r in all_uart]
    summary["uart_transmission_time_estimate_s"] = stats(uart_tx_values)

    acc = summarize_accuracy(trials)
    if acc:
        summary["application_accuracy"] = acc
        e2e = [r["host_e2e_latency_s_first_prediction"] for r in trials if r["host_e2e_latency_s_first_prediction"] != ""]
        summary["host_e2e_latency_s_first_prediction"] = stats(e2e)

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nDONE")
    print(f"Output: {out_dir}")
    print("\nMain summary:")
    print(json.dumps({
        "memory": summary.get("memory"),
        "cnn_latency_us": summary.get("cnn_latency_us"),
        "application_accuracy": summary.get("application_accuracy"),
        "energy_estimate": summary.get("energy_estimate"),
        "compute_estimate": summary.get("compute_estimate"),
        "counts": summary.get("counts"),
    }, indent=2))

if __name__ == "__main__":
    main()
