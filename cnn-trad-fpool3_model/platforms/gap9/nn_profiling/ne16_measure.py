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

Power measurement with GPIO trigger + sleep:
    python ne16_measure.py --config configs/ne16_power_config.json

Results saved to /app/gap9_measurements/offline_measurements/ (mounted from host).
"""

import argparse
import csv
import glob
import json
import logging
import os
import re
import statistics
import subprocess
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
    "sleep_us": 0,
}


# ──────────────────────────────────────────────────────────────────────────────
# Load "left" keyword MFCC input
# ──────────────────────────────────────────────────────────────────────────────
def load_left_input(npz_path: str) -> np.ndarray:
    data = np.load(npz_path)
    key = list(data.keys())[0]
    arr = data[key].astype(np.float32)
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = np.transpose(arr, (0, 3, 1, 2))
    log.info("Loaded 'left' input: shape=%s  range=[%.2f, %.2f]",
             arr.shape, arr.min(), arr.max())
    return arr


# ──────────────────────────────────────────────────────────────────────────────
# Verify "left" prediction using float32 onnxruntime
# ──────────────────────────────────────────────────────────────────────────────
def verify_left_onnx(model_path: str, left_input_nchw: np.ndarray) -> int:
    try:
        import onnxruntime as ort
    except ImportError:
        log.warning("onnxruntime not installed — skipping Python-side verification")
        return -1
    left_nhwc = np.transpose(left_input_nchw, (0, 2, 3, 1))
    try:
        sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    except Exception as e:
        log.warning("onnxruntime could not load model — skipping verification: %s", e)
        return -1
    input_name = sess.get_inputs()[0].name
    output = sess.run(None, {input_name: left_nhwc})[0]
    pred = int(np.argmax(output[0]))
    log.info("Python float32 verification: predicted class %d (logit %.4f)",
             pred, float(output[0][pred]))
    return pred


# ──────────────────────────────────────────────────────────────────────────────
# Prepare NNGraph
# ──────────────────────────────────────────────────────────────────────────────
def prepare_ne16_graph(model_path: str, left_input_nchw: np.ndarray, quant_config: str):
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

    def calib_iter():
        yield [left_input_nchw]

    log.info("Collecting calibration statistics from 'left' MFCC input")
    stats = G.collect_statistics(calib_iter())

    log.info("Quantizing (NE16, SQ8, channel-wise)")
    G.quantize(stats, graph_options=q_opts)
    log.info("  Quantization done")
    return G


# ──────────────────────────────────────────────────────────────────────────────
# Sleep patch helpers
# ──────────────────────────────────────────────────────────────────────────────
def _find_test_c(build_dir: str) -> str | None:
    """Find the AT-generated test harness C file (may not be called test.c)."""
    result = subprocess.run(
        ["find", build_dir, "-name", "*.c", "-not", "-path", "*/BUILD/*"],
        capture_output=True, text=True
    )
    candidates = [f for f in result.stdout.strip().splitlines() if f]
    log.debug("C files in build_dir (outside BUILD/): %s", candidates)

    # Prefer the file that actually contains pi_cluster_send_task
    for path in candidates:
        try:
            if "pi_cluster_send_task" in open(path).read():
                return path
        except OSError:
            pass

    # Fallback: any .c that is not an AT kernel file
    for path in candidates:
        name = os.path.basename(path)
        if not any(kw in name for kw in ("_kernels", "_model", "_basic_kernels")):
            return path

    return None


def patch_test_c_for_sleep(build_dir: str, sleep_us: int, gpio_pin: int = None,
                           **kwargs) -> bool:
    """
    Patch the AT-generated test harness C file:
    1. Insert pi_time_wait_us(sleep_us) before/after cluster dispatch.
       If kwargs['n_loops'] > 1: wrap in a for-loop so binary runs N inferences
       autonomously (flash once, no repeated cmake builds).
    2. If gpio_pin given: prepend #define GPIO_MEAS_PIN so macros activate.
    """
    test_c = _find_test_c(build_dir)
    if not test_c:
        log.warning("Test harness C file not found in %s", build_dir)
        return False
    log.info("Patching %s", test_c)
    content = open(test_c).read()

    # ── 1. Sleep + optional loop around cluster dispatch ────────────────────
    n_loops = kwargs.get("n_loops", 1)
    pattern = r'([ \t]*pi_cluster_send_task(?:_to_cl)?\s*\([^;]+;\s*\n)'
    m = re.search(pattern, content)
    if not m:
        log.warning("pi_cluster_send_task* not found — cannot add sleep")
        return False
    orig = m.group(1)
    indent = re.match(r'([ \t]*)', orig).group(1)
    inner = (
        f"{indent}  pi_time_wait_us({sleep_us});\n"
        f"{indent}  {orig.lstrip()}"
        f"{indent}  pi_time_wait_us({sleep_us});\n"
    )
    if n_loops > 1:
        sleep_patched = (
            f"{indent}for (int _meas_i = 0; _meas_i < {n_loops}; _meas_i++) {{\n"
            f"{inner}"
            f"{indent}}}\n"
        )
        log.info("  Loop: %d × (sleep %d µs → inference → sleep %d µs) in binary",
                 n_loops, sleep_us, sleep_us)
    else:
        sleep_patched = (
            f"{indent}pi_time_wait_us({sleep_us});\n"
            f"{orig}"
            f"{indent}pi_time_wait_us({sleep_us});\n"
        )
        log.info("  Sleep: %d µs before/after pi_cluster_send_task*", sleep_us)
    content = content.replace(orig, sleep_patched, 1)

    # ── 2. Force GPIO_MEAS_PIN define so GPIO_HIGH/LOW macros activate ───────
    if gpio_pin is not None and "GPIO_MEAS_PIN" not in content:
        define = f"#ifndef GPIO_MEAS_PIN\n#define GPIO_MEAS_PIN PI_GPIO_A{gpio_pin}\n#endif\n"
        content = define + content
        log.info("  GPIO: prepended #define GPIO_MEAS_PIN PI_GPIO_A%d", gpio_pin)
    elif gpio_pin is not None:
        log.info("  GPIO: GPIO_MEAS_PIN already present in file")

    open(test_c, 'w').write(content)
    return True


def _find_build_subdir(build_dir: str) -> str | None:
    # NNTool 5.x uses lowercase 'build/', older used 'BUILD/<target>'
    for pattern in [
        os.path.join(build_dir, "build"),
        os.path.join(build_dir, "BUILD", "*"),
    ]:
        candidates = glob.glob(pattern)
        if candidates:
            sub = candidates[0]
            if os.path.isdir(sub):
                return sub
    # Fallback: find any directory with a Makefile
    result = subprocess.run(
        ["find", build_dir, "-maxdepth", "3", "-name", "Makefile"],
        capture_output=True, text=True
    )
    makefiles = [f for f in result.stdout.strip().splitlines() if f]
    if makefiles:
        return os.path.dirname(makefiles[0])
    return None


def recompile_build(build_dir: str) -> bool:
    sub = _find_build_subdir(build_dir)
    if not sub:
        log.warning("No BUILD/* in %s", build_dir)
        return False
    log.info("Recompiling: make -C %s", sub)
    proc = subprocess.run(["make", "-C", sub])
    ok = proc.returncode == 0
    if ok:
        log.info("  Recompile OK")
    else:
        log.warning("  make failed (rc=%d)", proc.returncode)
    return ok


def run_board_direct(build_dir: str, timeout: int = 300) -> int | None:
    """Run pre-compiled binary directly via cmake --target run, parse cycle count."""
    sub = _find_build_subdir(build_dir)
    if not sub:
        log.warning("Cannot find build subdir in %s", build_dir)
        return None
    proc = subprocess.run(
        ["cmake", "--build", sub, "--target", "run"],
        capture_output=True, text=True, timeout=timeout
    )
    output = proc.stdout + "\n" + proc.stderr
    # Parse "Total" line from AT performance table
    # Handles both "Total   188646" and "* Total   188646" formats
    for line in output.splitlines():
        parts = line.split()
        for i, p in enumerate(parts):
            if p in ("Total", "total") and i + 1 < len(parts):
                try:
                    return int(parts[i + 1].replace(",", ""))
                except ValueError:
                    pass
    log.debug("run_board_direct output:\n%s", output[:2000])
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Single board run via NNTool (generates project, compiles, runs)
# ──────────────────────────────────────────────────────────────────────────────
def run_once(G, ms, left_input_nchw: np.ndarray, build_dir: str,
             get_memory: bool = False, eot_kwargs: dict = None) -> dict:
    kwargs = dict(eot_kwargs or {})
    result = G.execute_on_target(
        directory=build_dir,
        input_tensors=[left_input_nchw],
        settings=ms,
        platform="board",
        performance=True,
        at_log=True,
        memory=get_memory,
        **kwargs,
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
                 first_run: dict, pred_class: int, measurements_root: str):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    variant = cfg.get("variant_name", "ds_cnn_l_ne16_int8")
    out_dir = os.path.join(measurements_root, "offline_measurements",
                           f"{variant}_offline_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "variant": variant,
        "board": cfg.get("board", "GAP9_EVK"),
        "clock_hz": cfg.get("clock_hz"),
        "mac_ops": cfg.get("mac_ops"),
        "runs": cfg.get("runs"),
        "sleep_us": cfg.get("sleep_us", 0),
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
    parser = argparse.ArgumentParser(description="NE16 board measurement")
    parser.add_argument("--config", metavar="JSON")
    parser.add_argument("--runs", type=int)
    parser.add_argument("--model", metavar="ONNX")
    parser.add_argument("--left-input", metavar="NPZ")
    args = parser.parse_args()

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

    n_runs          = cfg["runs"]
    clock_hz        = cfg["clock_hz"]
    mac_ops         = cfg["mac_ops"]
    model_path      = cfg["model_path"]
    left_input_npz  = cfg["left_input_npz"]
    build_dir       = cfg["build_dir"]
    measurements_root = cfg["measurements_root"]
    sleep_us        = int(cfg.get("sleep_us", 0))

    log.info("=" * 60)
    log.info("NE16 Board Measurement  —  %d runs", n_runs)
    log.info("=" * 60)

    left_input_nchw = load_left_input(left_input_npz)

    pred_class = verify_left_onnx(model_path, left_input_nchw)
    if pred_class == 2:
        log.info("  ✓ Float32 Python: predicted class 2 = 'Left'")
    elif pred_class >= 0:
        log.warning("  ! Float32 Python: predicted class %d (expected 2)", pred_class)

    G = prepare_ne16_graph(model_path, left_input_nchw, cfg["quant_config"])

    from nntool.api.utils import model_settings
    target_cfg = load_yaml(cfg["target_config"])
    ms = model_settings(**target_cfg.get("settings", {}))
    eot_kwargs = target_cfg.get("execute_on_target_kwargs", {})

    if eot_kwargs:
        log.info("execute_on_target kwargs: %s", eot_kwargs)
    if sleep_us:
        log.info("Sleep: %d µs — will patch test.c after first compile", sleep_us)

    all_cycles = []
    first_run_data = {}
    # After first compile + sleep patch, use run_board_direct for remaining runs
    use_direct = False

    for run_idx in range(n_runs):
        log.info("── Run %d / %d ──────────────────────────────", run_idx + 1, n_runs)

        if use_direct:
            # Binary already patched with sleep+GPIO — flash and run directly
            cycles = run_board_direct(build_dir)
            run_data = {"total_cycles": cycles, "performance": [], "memory": None}
        else:
            # NNTool path: generates AT project, compiles, runs
            get_mem = (run_idx == 0)
            run_data = run_once(G, ms, left_input_nchw, build_dir,
                                get_memory=get_mem, eot_kwargs=eot_kwargs)
            if run_idx == 0:
                first_run_data = run_data
                # After first compile: patch test.c for sleep if requested
                if sleep_us > 0:
                    gpio_pin = eot_kwargs.get("gpio_meas") if eot_kwargs else None
                    n_loops = cfg.get("n_loops", 1)
                    if patch_test_c_for_sleep(build_dir, sleep_us,
                                              gpio_pin=gpio_pin, n_loops=n_loops):
                        if recompile_build(build_dir):
                            use_direct = True
                            log.info("Patch active — remaining runs use direct board flash")

        cycles = run_data["total_cycles"]
        all_cycles.append(cycles)
        if run_idx == 0 and not first_run_data:
            first_run_data = run_data

        if cycles is not None:
            log.info("  Cycles: %d  (%.3f ms)", cycles, cycles / clock_hz * 1000)
        else:
            log.warning("  Could not extract total cycles from result")

    stats = compute_stats(all_cycles, clock_hz, mac_ops)
    if not stats:
        log.error("No valid cycle counts collected — board runs all failed")
        return

    log.info("")
    log.info("── RESULTS ──────────────────────────────────────────────────")
    log.info("  Runs           : %d", stats["count"])
    log.info("  Min cycles     : %s", f"{stats['min_cycles']:,}")
    log.info("  Max cycles     : %s", f"{stats['max_cycles']:,}")
    log.info("  Mean cycles    : %s", f"{stats['mean_cycles']:,.0f}")
    log.info("  Median cycles  : %s", f"{stats['median_cycles']:,}")
    log.info("  Std cycles     : %s", f"{stats['std_cycles']:,.0f}")
    log.info("  Mean latency   : %.3f ms  @ %.0f MHz",
             stats['mean_latency_ms'], clock_hz / 1e6)
    log.info("  MACs/cycle     : %.2f", stats['mean_macs_per_cycle'])
    log.info("────────────────────────────────────────────────────────────")

    os.makedirs(measurements_root, exist_ok=True)
    out_dir = save_results(cfg, stats, all_cycles, first_run_data,
                           pred_class, measurements_root)
    log.info("Done. Results: %s", out_dir)


if __name__ == "__main__":
    main()
