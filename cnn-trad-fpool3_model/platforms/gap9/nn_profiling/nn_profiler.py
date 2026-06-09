#!/usr/bin/env python3
"""
nn_profiler.py — Load, quantize, and profile a neural network on GAP9 via NNTool.

Usage examples:
    # Quick run with a preset quantization scheme
    python nn_profiler.py model.onnx --scheme ne16 --platform gvsoc

    # Full run with YAML config files
    python nn_profiler.py model.tflite --config quant_config.yml --target-config target_config.yml --platform board

    # FP16 on GVSOC
    python nn_profiler.py model.onnx --scheme fp16 --platform gvsoc
"""

import argparse
import logging
import os
import shutil
import sys
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from prepare_graph import *
from logging_utils import setup_logging, load_yaml, format_bytes

log = logging.getLogger("nn_profiler")


# ──────────────────────────────────────────────────────────────────────────────
# Experiment directory
# ──────────────────────────────────────────────────────────────────────────────
def create_experiment_dir(model_path: str, scheme_name: str, base_dir: str = "experiments", exp_label: str = None) -> str:
    """Create a timestamped experiment directory and return its path."""
    model_stem = Path(model_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if exp_label:
        dirname = f"{model_stem}_{scheme_name}_{exp_label}_{timestamp}"
    else:
        dirname = f"{model_stem}_{scheme_name}_{timestamp}"
    exp_dir = os.path.join(base_dir, dirname)
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


def save_experiment_configs(
        model_path: str,
        platform: str,
        scheme: str,
        load_tflite_quant: bool,
        exp_dir: str,
        graph_cfg: dict,
        target_cfg: dict
    ):
    """Save all configuration used for this experiment."""
    # Save the resolved graph config
    with open(os.path.join(exp_dir, "graph_config.yml"), "w") as f:
        yaml.dump(graph_cfg, f, default_flow_style=False, sort_keys=False)

    # Save the resolved target config
    with open(os.path.join(exp_dir, "target_config.yml"), "w") as f:
        yaml.dump(target_cfg, f, default_flow_style=False, sort_keys=False)

    # Save CLI args
    cli_info = {
        "model": model_path,
        "platform": platform,
        "scheme": scheme,
        "config": graph_cfg,
        "target_config": target_cfg,
        "load_tflite_quant": load_tflite_quant,
    }
    with open(os.path.join(exp_dir, "cli_args.yml"), "w") as f:
        yaml.dump(cli_info, f, default_flow_style=False, sort_keys=False)


def save_memory_plots(result, exp_dir: str):
    """Save plot_memory_boxes figures to the experiment directory."""
    try:
        figs = result.plot_memory_boxes()
        if not figs:
            log.warning("No memory box figures returned")
            return
        areas = ["L1", "L2", "L3", "ConstL3"]
        for i, fig in enumerate(figs):
            label = areas[i] if i < len(areas) else f"mem_{i}"
            path = os.path.join(exp_dir, f"mem_boxes_{label}.png")
            fig.savefig(path, bbox_inches="tight", dpi=150)
            plt.close(fig)
            log.info("Saved memory plot: %s", path)
    except Exception as e:
        log.warning("Could not generate memory box plots: %s", e)

def report_results(result, exp_dir: str = None):
    """Print profiling results with colored formatting and save to experiment dir."""
    BOLD = "\033[1m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    MAGENTA = "\033[35m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    sep = f"{DIM}{'─' * 80}{RESET}"

    # ── Performance table ────────────────────────────────────────────────
    if hasattr(result, "performance") and result.performance:
        perf = result.performance
        log.info(sep)
        log.info("  %s%s⚡ PERFORMANCE SUMMARY%s", BOLD, CYAN, RESET)
        log.info(sep)

        if isinstance(perf, list):
            # Header
            headers = ["Node", "Cycles", "Ops", "Ops/Cyc", "% Cycles", "% Ops"]
            widths = [35, 12, 12, 10, 10, 10]
            header_line = "  ".join(f"{BOLD}{h:<{w}}{RESET}" for h, w in zip(headers, widths))
            log.info("  %s", header_line)
            log.info("  %s%s%s", DIM, "─" * sum(w + 2 for w in widths), RESET)

            for row in perf:
                if not isinstance(row, (list, tuple)) or len(row) < 6:
                    continue
                name, cycles, ops, ops_cyc, pct_cyc, pct_ops = row[:6]
                if name == "Total":
                    log.info("  %s%s%s", DIM, "─" * sum(w + 2 for w in widths), RESET)
                    ops_cyc_s = f"{ops_cyc:.2f}" if not (ops_cyc != ops_cyc) else "—"  # NaN check
                    line = (
                        f"  {BOLD}{YELLOW}{name:<35}{RESET}  "
                        f"{BOLD}{cycles:<12,}{RESET}  "
                        f"{BOLD}{ops:<12,}{RESET}  "
                        f"{BOLD}{ops_cyc_s:<10}{RESET}  "
                        f"{BOLD}{'—':<10}{RESET}  "
                        f"{BOLD}{'—':<10}{RESET}"
                    )
                elif name == "IO_Wait":
                    continue
                else:
                    ops_cyc_s = f"{ops_cyc:.2f}" if not (ops_cyc != ops_cyc) else "—"
                    pct_cyc_s = f"{pct_cyc:.1f}%"
                    pct_ops_s = f"{pct_ops:.1f}%"
                    line = (
                        f"  {GREEN}{name:<35}{RESET}  "
                        f"{cycles:<12,}  "
                        f"{ops:<12,}  "
                        f"{ops_cyc_s:<10}  "
                        f"{pct_cyc_s:<10}  "
                        f"{pct_ops_s:<10}"
                    )
                log.info("%s", line)
        elif isinstance(perf, dict):
            for key, val in perf.items():
                log.info("  %s%s%s: %s", GREEN, key, RESET, val)
        else:
            log.info("  %s", perf)

    # ── Memory info ──────────────────────────────────────────────────────
    if hasattr(result, "basic_mem_infos") and result.basic_mem_infos:
        log.info(sep)
        log.info("  %s%s🧠 MEMORY INFO%s", BOLD, MAGENTA, RESET)
        log.info(sep)
        mem = result.basic_mem_infos
        if isinstance(mem, dict):
            # Pre-extract L2 given for static/dyn bar rendering
            l2_given = 0
            if "L2" in mem and isinstance(mem["L2"], dict):
                l2_given = mem["L2"].get("given", 0)
            l2_static = mem.get("L2 Static Memory Usage", 0)
            l2_dyn = mem.get("L2 Dyn Memory Usage", 0)

            BLUE = "\033[34m"
            RED = "\033[31m"

            for k, v in mem.items():
                # Render L2 Static / Dyn as stacked bars against L2 given
                if k in ("L2 Static Memory Usage", "L2 Dyn Memory Usage") and l2_given:
                    bar_len = 20
                    static_pct = (l2_static / l2_given * 100) if l2_given else 0
                    dyn_pct = (l2_dyn / l2_given * 100) if l2_given else 0
                    static_blocks = int(bar_len * static_pct / 100)
                    dyn_blocks = int(bar_len * dyn_pct / 100)
                    empty_blocks = bar_len - static_blocks - dyn_blocks

                    if k == "L2 Static Memory Usage":
                        pct = static_pct
                        bar = f"{BLUE}{'█' * static_blocks}{DIM}{'░' * (bar_len - static_blocks)}{RESET}"
                        log.info("  %s%-30s%s %s %5.1f%%  (%s / %s)", CYAN, k, RESET, bar, pct, format_bytes(l2_static), format_bytes(l2_given))
                    else:
                        pct = dyn_pct
                        bar = f"{MAGENTA}{'█' * dyn_blocks}{DIM}{'░' * (bar_len - dyn_blocks)}{RESET}"
                        log.info("  %s%-30s%s %s %5.1f%%  (%s / %s)", CYAN, k, RESET, bar, pct, format_bytes(l2_dyn), format_bytes(l2_given))
                elif isinstance(v, dict) and "given" in v and "used" in v:
                    given, used = v["given"], v["used"]
                    pct = (used / given * 100) if given else 0
                    bar_len = 20
                    filled = int(bar_len * pct / 100)
                    bar_color = GREEN if pct < 70 else (YELLOW if pct < 90 else "\033[31m")
                    bar = f"{bar_color}{'█' * filled}{DIM}{'░' * (bar_len - filled)}{RESET}"
                    log.info("  %s%-30s%s %s %5.1f%%  (%s / %s)", CYAN, k, RESET, bar, pct, format_bytes(used), format_bytes(given))
                else:
                    log.info("  %s%-30s%s %s", CYAN, k, RESET, v)
        else:
            log.info("  %s", mem)

    log.info(sep)

    # ── Save results to experiment dir ───────────────────────────────────
    if exp_dir:
        results_data = {}
        if hasattr(result, "performance") and result.performance:
            results_data["performance"] = result.performance
        if hasattr(result, "basic_mem_infos") and result.basic_mem_infos:
            results_data["memory"] = {
                k: (v if isinstance(v, (int, float, str, dict)) else str(v))
                for k, v in result.basic_mem_infos.items()
            }
        results_path = os.path.join(exp_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(results_data, f, indent=2, default=str)
        log.info("Results saved to %s", results_path)
    
    save_memory_plots(result, exp_dir)

def run_experiment(
        model_path: str,
        graph_config: str,
        target_config: str,
        exp_dir: str,
        scheme: str = "ne16",
        load_tflite_quant: bool = False,
        platform: str = "gvsoc",
        exp_label: str = None,
):

    log.info("=" * 60)
    log.info("NN Profiler started")
    log.info("=" * 60)

    if not graph_config and not scheme:
        log.warning("No quantization scheme specified, defaulting to ne16")

    # Save configs

    # ── Build quantization config ────────────────────────────────────────────
    if graph_config:
        g_cfg = load_yaml(graph_config)
        scheme_label = Path(graph_config).stem
    elif scheme:
        g_cfg = dict(PRESETS[scheme])
        scheme_label = scheme
    else:
        g_cfg = dict(PRESETS["ne16"])
        scheme_label = "ne16"

    # ── Create experiment directory ─────────────────────────────────────────
    exp_dir = create_experiment_dir(model_path, scheme_label, base_dir=exp_dir, exp_label=exp_label)
    setup_logging(exp_dir)
    log.info("Experiment directory: %s", exp_dir)

    graph_options = g_cfg.get("graph_options", {})
    node_options = g_cfg.get("node_options", None)
    fusions = g_cfg.get("fusions", None)

    target_cfg = load_yaml(target_config) if target_config else {}
    save_experiment_configs(
        model_path,
        platform,
        scheme_label,
        load_tflite_quant,
        exp_dir,
        g_cfg,
        target_cfg
    )

    # ── Build target settings ───────────────────────────────────────────────
    target_settings = target_cfg.get("settings", {})
    eot_kwargs = target_cfg.get("execute_on_target_kwargs", {})
    io_settings = target_cfg.get("io_settings", {})

    log.info("Configs saved to experiment directory")

    # ── Prepare graph (load + fusions + fake quantize) ──────────────────────
    G = prepare_graph(
        model_path,
        load_tflite_quant=load_tflite_quant,
        graph_options=graph_options,
        node_options=node_options,
        fusions=fusions,
    )
    # ── Execute on target ───────────────────────────────────────────────────
    result = execute_on_target(
        G,
        settings=target_settings,
        platform=platform,
        io_settings=io_settings,
        **eot_kwargs
    )

    # ── Print results ───────────────────────────────────────────────────────
    report_results(result, exp_dir=exp_dir)

    # ── Save memory box plots ───────────────────────────────────────────────
    save_memory_plots(result, exp_dir)
    log.info("Experiment complete: %s", exp_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Load, quantize, and profile an NN model on GAP9 via NNTool.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("model", help="Path to the NN model file (.onnx, .tflite, .json)")
    parser.add_argument("--load-tflite-quant", action="store_true", help="Load TFLite quantization parameters")

    # Quantization source: either preset or config file
    quant_group = parser.add_mutually_exclusive_group()
    quant_group.add_argument(
        "--scheme",
        choices=list(PRESETS.keys()),
        help="Use a built-in quantization preset",
    )
    quant_group.add_argument(
        "--config",
        metavar="YAML",
        help="YAML config file for quantization (graph_options, node_options, fusions, etc.)",
    )

    # Target / execution config
    parser.add_argument(
        "--target-config",
        metavar="YAML",
        default="configs/target_config_opt.yml",
        help="YAML config file with model_settings for execute_on_target",
    )

    # Platform
    parser.add_argument(
        "--platform",
        choices=["gvsoc", "board", "emul"],
        default="gvsoc",
        help="Execution platform (default: gvsoc)",
    )

    parser.add_argument(
        "--exp-dir",
        metavar="DIR",
        default="experiments",
        help="Base directory for experiment logs (default: experiments)",
    )

    parser.add_argument(
        "--exp-label",
        metavar="LABEL",
        help="Label for the experiment (default: derived from directory name)",
    )

    args = parser.parse_args()
    run_experiment(
        model_path=args.model,
        graph_config=args.config,
        target_config=args.target_config,
        exp_dir=args.exp_dir,
        scheme=args.scheme,
        load_tflite_quant=args.load_tflite_quant,
        platform=args.platform,
        exp_label=args.exp_label,
    )

if __name__ == "__main__":
    main()
