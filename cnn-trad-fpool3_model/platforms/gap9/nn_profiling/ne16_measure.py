#!/usr/bin/env python3
"""
ne16_measure.py — Run N NE16 inferences on GAP9 board, collect cycle statistics,
and verify the "left" keyword prediction.

First run: compiles the AT binary (~5-6 min).
Subsequent runs: NNTool reuses the build in /tmp/nn_profiler (make sees nothing changed).

Usage (inside Docker container, after sourcing gap9 env):
    cd /app/nn_profiling
    python ne16_measure.py --config configs/ne16_measure_config.json
    python ne16_measure.py --runs 10          # quick test

Results saved to /app/gap9_measurements/offline_measurements/ (mounted from host).
"""

import argparse
import csv
import json
import logging
import os
import re
import statistics
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from logging_utils import setup_logging, load_yaml, format_bytes

log = logging.getLogger("nn_profiler")


# ──────────────────────────────────────────────────────────────────────────────
# Defaults (overridden by config file or CLI)
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "runs": 20,
    "clock_hz": 240_000_000.0,
    "mac_ops": 3_832_460.0,
    "variant_name": "ds_cnn_l_ne16_int8",
    "model_path": "/app/models/ds_cnn_l.onnx",
    "left_input_npz": "/app/nn_profiling/data/dscnnl_left_input.npz",
    "quant_config": "configs/quant_ne16.yml",
    "target_config": "configs/target_config_opt.yml",
    "build_dir": "/tmp/nn_profiler_ne16",
    "measurements_root": "/app/gap9_measurements",
    "board": "GAP9_EVK",
}


# ──────────────────────────────────────────────────────────────────────────────
# Load "left" keyword MFCC input
# ──────────────────────────────────────────────────────────────────────────────
def load_left_input(npz_path: str) -> np.ndarray:
    """Load float32 'left' MFCC features; returns NCHW array for NNTool."""
    data = np.load(npz_path)
    key = list(data.keys())[0]
    arr = data[key].astype(np.float32)          # (1, 49, 10, 1) NHWC
    if arr.ndim == 4 and arr.shape[-1] == 1:    # NHWC → NCHW
        arr = np.transpose(arr, (0, 3, 1, 2))   # (1, 1, 49, 10)
    log.info("Loaded 'left' input: shape=%s  range=[%.2f, %.2f]",
             arr.shape, arr.min(), arr.max())
    return arr


# ──────────────────────────────────────────────────────────────────────────────
# Verify "left" prediction using float32 onnxruntime (no board needed)
# ──────────────────────────────────────────────────────────────────────────────
def verify_left_onnx(model_path: str, left_input_nchw: np.ndarray) -> int:
    """Run float32 inference with onnxruntime; returns predicted class index."""
    try:
        import onnxruntime as ort
    except ImportError:
        log.warning("onnxruntime not installed — skipping Python-side verification")
        return -1

    # onnxruntime expects NHWC for this model (original input shape is NHWC)
    left_nhwc = np.transpose(left_input_nchw, (0, 2, 3, 1))  # (1,49,10,1)
    try:
        sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    except Exception as e:
        log.warning("onnxruntime could not load model (old opset?) — skipping Python-side verification: %s", e)
        return -1
    input_name = sess.get_inputs()[0].name
    output = sess.run(None, {input_name: left_nhwc})[0]
    pred = int(np.argmax(output[0]))
    log.info("Python float32 verification: predicted class %d (logit %.4f)", pred, float(output[0][pred]))
    return pred


# ──────────────────────────────────────────────────────────────────────────────
# Prepare NNGraph with real "left" calibration
# ──────────────────────────────────────────────────────────────────────────────
def prepare_ne16_graph(model_path: str, left_input_nchw: np.ndarray, quant_config: str):
    """Load, fuse, and quantize DS-CNN-L with NE16 using real 'left' input for calibration."""
    from nntool.api import NNGraph
    from nntool.api.utils import quantization_options

    log.info("Loading model: %s", model_path)
    G = NNGraph.load_graph(model_path)
    G.adjust_order()

    cfg = load_yaml(quant_config)
    fusions = cfg.get("fusions", ["scaled_match_group"])
    graph_options = cfg.get("graph_options", {})

    log.info("Applying fusions: %s", fusions)
    G.fusions(*fusions)

    q_opts = quantization_options(**graph_options)

    # Calibrate with the real 'left' MFCC input (range ~-52..18)
    # This is critical: fake random [-1,1] gives wrong quantization scales.
    def calib_iter():
        yield [left_input_nchw]

    log.info("Collecting calibration statistics from 'left' MFCC input")
    stats = G.collect_statistics(calib_iter())

    log.info("Quantizing (NE16, SQ8, channel-wise)")
    G.quantize(stats, graph_options=q_opts)
    log.info("  Quantization done")
    return G


# ──────────────────────────────────────────────────────────────────────────────
# Single board run
# ──────────────────────────────────────────────────────────────────────────────
def run_once(G, ms, left_input_nchw: np.ndarray, build_dir: str,
             get_memory: bool = False) -> dict:
    """
    Run one board inference via execute_on_target.

    Second and later calls reuse the build in build_dir because:
    - input data is identical (same left_input_nchw every run)
    - model/settings are identical
    → make sees nothing changed → skips compilation → only gapy runs.
    """
    result = G.execute_on_target(
        directory=build_dir,
        input_tensors=[left_input_nchw],
        settings=ms,
        platform="board",
        performance=True,
        at_log=True,
        memory=get_memory,
    )
    perf = result.performance if hasattr(result, "performance") else []
    total_cycles = None
    if isinstance(perf, list):
        for row in perf:
            if isinstance(row, (list, tuple)) and len(row) >= 2 and row[0] == "Total":
                total_cycles = int(row[1])
                break

    mem_info = result.basic_mem_infos if (get_memory and hasattr(result, "basic_mem_infos")) else None
    return {"total_cycles": total_cycles, "performance": perf, "memory": mem_info}


# ──────────────────────────────────────────────────────────────────────────────
# Statistics
# ──────────────────────────────────────────────────────────────────────────────
def compute_stats(cycles_list: list, clock_hz: float, mac_ops: float) -> dict:
    valid = [c for c in cycles_list if c is not None]
    if not valid:
        return {}
    sorted_c = sorted(valid)
    n = len(valid)
    return {
        "count": n,
        "min_cycles": sorted_c[0],
        "max_cycles": sorted_c[-1],
        "mean_cycles": statistics.mean(valid),
        "median_cycles": statistics.median(valid),
        "std_cycles": statistics.stdev(valid) if n > 1 else 0.0,
        "p95_cycles": sorted_c[int(0.95 * n)],
        "mean_latency_ms": statistics.mean(valid) / clock_hz * 1000,
        "mean_macs_per_cycle": mac_ops / statistics.mean(valid),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Save results
# ──────────────────────────────────────────────────────────────────────────────
def save_results(cfg: dict, stats: dict, all_cycles: list,
                 first_run: dict, pred_class: int,
                 measurements_root: str):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    variant = cfg.get("variant_name", "ds_cnn_l_ne16_int8")
    out_dir = os.path.join(measurements_root, "offline_measurements",
                           f"{variant}_offline_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    # summary.json
    summary = {
        "timestamp": datetime.now().isoformat(),
        "variant": variant,
        "board": cfg.get("board", "GAP9_EVK"),
        "clock_hz": cfg.get("clock_hz"),
        "mac_ops": cfg.get("mac_ops"),
        "runs": cfg.get("runs"),
        "python_predicted_class": pred_class,
        "stats": stats,
    }
    if first_run.get("memory"):
        summary["memory"] = {
            k: (v if isinstance(v, (int, float, str, dict)) else str(v))
            for k, v in first_run["memory"].items()
        }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # cycles_raw.csv
    with open(os.path.join(out_dir, "cycles_raw.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "cycles", "latency_us"])
        writer.writeheader()
        clock_hz = cfg.get("clock_hz", 240e6)
        for i, c in enumerate(all_cycles):
            writer.writerow({
                "run": i + 1,
                "cycles": c if c is not None else "",
                "latency_us": f"{c / clock_hz * 1e6:.1f}" if c else "",
            })

    log.info("Results saved to %s", out_dir)
    return out_dir


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="NE16 board measurement: 20 inferences + 'left' verification")
    parser.add_argument("--config", metavar="JSON", help="Path to config JSON (overrides defaults)")
    parser.add_argument("--runs", type=int, help="Number of board inferences")
    parser.add_argument("--model", metavar="ONNX", help="Path to DS-CNN-L ONNX model")
    parser.add_argument("--left-input", metavar="NPZ", help="Path to float32 DSCNNL inputs.npz")
    args = parser.parse_args()

    # Merge: defaults → config file → CLI overrides
    cfg = dict(DEFAULT_CONFIG)
    if args.config:
        cfg.update(load_yaml(args.config) if args.config.endswith((".yml", ".yaml"))
                   else json.load(open(args.config)))
    if args.runs:
        cfg["runs"] = args.runs
    if args.model:
        cfg["model_path"] = args.model
    if args.left_input:
        cfg["left_input_npz"] = args.left_input

    setup_logging()

    n_runs = cfg["runs"]
    clock_hz = cfg["clock_hz"]
    mac_ops = cfg["mac_ops"]
    model_path = cfg["model_path"]
    left_input_npz = cfg["left_input_npz"]
    build_dir = cfg["build_dir"]
    measurements_root = cfg["measurements_root"]

    log.info("=" * 60)
    log.info("NE16 Board Measurement  —  %d runs", n_runs)
    log.info("=" * 60)

    # ── Load 'left' input ────────────────────────────────────────────────────
    left_input_nchw = load_left_input(left_input_npz)

    # ── Python-side float32 verification ────────────────────────────────────
    pred_class = verify_left_onnx(model_path, left_input_nchw)
    if pred_class == 2:
        log.info("  ✓ Float32 Python: predicted class 2 = 'Left'")
    elif pred_class >= 0:
        log.warning("  ! Float32 Python: predicted class %d (expected 2 = 'Left')", pred_class)

    # ── Prepare NNGraph ──────────────────────────────────────────────────────
    G = prepare_ne16_graph(model_path, left_input_nchw, cfg["quant_config"])

    # ── Build model_settings ─────────────────────────────────────────────────
    from nntool.api.utils import model_settings
    target_cfg = load_yaml(cfg["target_config"])
    ms = model_settings(**target_cfg.get("settings", {}))

    # ── Board runs ───────────────────────────────────────────────────────────
    all_cycles = []
    first_run_data = {}

    for run_idx in range(n_runs):
        log.info("── Run %d / %d ──────────────────────────────", run_idx + 1, n_runs)
        get_mem = (run_idx == 0)
        run_data = run_once(G, ms, left_input_nchw, build_dir, get_memory=get_mem)

        cycles = run_data["total_cycles"]
        all_cycles.append(cycles)

        if run_idx == 0:
            first_run_data = run_data

        if cycles is not None:
            lat_ms = cycles / clock_hz * 1000
            log.info("  Cycles: %d  (%.3f ms)", cycles, lat_ms)
        else:
            log.warning("  Could not extract total cycles from result")

    # ── Statistics ───────────────────────────────────────────────────────────
    stats = compute_stats(all_cycles, clock_hz, mac_ops)
    if not stats:
        log.error("No valid cycle counts collected — board runs all failed")
        return

    log.info("")
    log.info("── RESULTS ──────────────────────────────────────────────────")
    log.info("  Runs           : %d", stats.get("count", 0))
    log.info("  Min cycles     : %s", f"{stats['min_cycles']:,}")
    log.info("  Max cycles     : %s", f"{stats['max_cycles']:,}")
    log.info("  Mean cycles    : %s", f"{stats['mean_cycles']:,.0f}")
    log.info("  Median cycles  : %s", f"{stats['median_cycles']:,}")
    log.info("  Std cycles     : %s", f"{stats['std_cycles']:,.0f}")
    log.info("  Mean latency   : %.3f ms  @ %.0f MHz",
             stats['mean_latency_ms'], clock_hz / 1e6)
    log.info("  MACs/cycle     : %.2f", stats['mean_macs_per_cycle'])
    log.info("────────────────────────────────────────────────────────────")

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs(measurements_root, exist_ok=True)
    out_dir = save_results(cfg, stats, all_cycles, first_run_data,
                           pred_class, measurements_root)
    log.info("Done. Results: %s", out_dir)


if __name__ == "__main__":
    main()
