#!/usr/bin/env python3
"""
Cross-platform comparison: STM32U5 vs MAX78000 — ai85kws20netv3 model.
Generates tables (CSV) and plots (PNG) in the same directory.
"""

import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

ROOT = Path("/home/pascal/Documents/ml_on_mcu/MLonMCU_Project")
OUT  = ROOT / "final_results" / "stm32u5_vs_max78000_ai85kws20netv3"
OUT.mkdir(parents=True, exist_ok=True)

# ── Variants ──────────────────────────────────────────────────────────────────
VARIANTS = [
    {
        "id":       "u5_v0",
        "label":    "STM32U5\nv0 float32",
        "short":    "U5-v0",
        "platform": "STM32U5",
        "version":  "v0",
        "color":    "#1d4ed8",
        "marker":   "o",
        "path":     ROOT / "Experiments/General Profiling/U5/ai85kws20netv3_model_v0/online/on.json",
    },
    {
        "id":       "u5_v1",
        "label":    "STM32U5\nv1 INT8",
        "short":    "U5-v1",
        "platform": "STM32U5",
        "version":  "v1",
        "color":    "#0891b2",
        "marker":   "o",
        "path":     ROOT / "Experiments/General Profiling/U5/ai85kws20netv3_model_v1/online/on.json",
    },
    {
        "id":       "u5_v2",
        "label":    "STM32U5\nv2 pruned",
        "short":    "U5-v2",
        "platform": "STM32U5",
        "version":  "v2",
        "color":    "#0e7490",
        "marker":   "o",
        "path":     ROOT / "Experiments/General Profiling/U5/ai85kws20netv3_model_v2/online/on.json",
    },
    {
        "id":       "mx_v0",
        "label":    "MAX78000\nv0 float32",
        "short":    "MX-v0",
        "platform": "MAX78000",
        "version":  "v0",
        "color":    "#b45309",
        "marker":   "s",
        "path":     ROOT / "Experiments/General Profiling/MAX78000/ai85kws20netv3_model_v0/online/on.json",
    },
    {
        "id":       "mx_v1",
        "label":    "MAX78000\nv1 INT8 HW",
        "short":    "MX-v1",
        "platform": "MAX78000",
        "version":  "v1",
        "color":    "#d97706",
        "marker":   "s",
        "path":     ROOT / "Experiments/General Profiling/MAX78000/ai85kws20netv3_model_v1/online/on.json",
    },
]

def g(d, key, default=None):
    parts = key.split(".")
    v = d
    for p in parts:
        if not isinstance(v, dict) or p not in v:
            return default
        v = v[p]
    return v

# ── Load ──────────────────────────────────────────────────────────────────────
for v in VARIANTS:
    with open(v["path"]) as f:
        v["data"] = json.load(f)

def val(v, key):
    r = g(v["data"], key)
    if isinstance(r, str):
        try: return float(r)
        except: return None
    return r

# ── Helpers ───────────────────────────────────────────────────────────────────
def fmt_val(v):
    if v is None: return "—"
    if abs(v) >= 1e9:  return f"{v/1e9:.3f}G"
    if abs(v) >= 1e6:  return f"{v/1e6:.3f}M"
    if abs(v) >= 1e3:  return f"{v/1e3:.2f}k"
    if abs(v) < 0.001: return f"{v:.2e}"
    return f"{v:.4g}"

def log_bar(ax, values, labels, colors, title, ylabel, unit=""):
    # replace 0/None with NaN so log scale doesn't explode
    safe = [v if (v is not None and v > 0) else float("nan") for v in values]
    x = np.arange(len(safe))
    bars = ax.bar(x, safe, color=colors, edgecolor="white", linewidth=0.8, zorder=3)
    values = safe
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.set_ylabel(f"{ylabel} [{unit}]" if unit else ylabel, fontsize=8)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines[["top","right"]].set_visible(False)
    real = [v for v in values if not (v != v)]  # filter NaN
    best = min(real) if real else None
    for bar, v in zip(bars, values):
        if v != v: continue  # skip NaN
        clr = "#dc2626" if v == best else "black"
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.4,
                fmt_val(v), ha="center", va="bottom", fontsize=7.5,
                color=clr, fontweight="bold" if v == best else "normal")

def lin_bar(ax, values, labels, colors, title, ylabel, unit="", highlight_min=True):
    x = np.arange(len(values))
    bars = ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.set_ylabel(f"{ylabel} [{unit}]" if unit else ylabel, fontsize=8)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines[["top","right"]].set_visible(False)
    best = min(values) if highlight_min else max(values)
    for bar, v in zip(bars, values):
        clr = "#dc2626" if v == best else "black"
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.02,
                fmt_val(v), ha="center", va="bottom", fontsize=7.5,
                color=clr, fontweight="bold" if v == best else "normal")

SHORT   = [v["short"]  for v in VARIANTS]
LABELS  = [v["label"]  for v in VARIANTS]
COLORS  = [v["color"]  for v in VARIANTS]

# ── CSV Table ─────────────────────────────────────────────────────────────────
METRICS_TABLE = [
    ("Platform",                  lambda v: v["platform"],                                       ""),
    ("Version",                   lambda v: v["version"],                                        ""),
    ("Latency avg (ms)",          lambda v: val(v,"cnn_latency_ms_avg"),                         "ms"),
    ("Latency min (ms)",          lambda v: (val(v,"cnn_latency_us.min") or 0)/1000,             "ms"),
    ("Latency max (ms)",          lambda v: (val(v,"cnn_latency_us.max") or 0)/1000,             "ms"),
    ("Throughput (inf/s)",        lambda v: val(v,"inferences_per_second_from_cnn_time"),        "inf/s"),
    ("E2E lower bound (ms)",      lambda v: val(v,"estimated_end_to_end_lower_bound_ms"),        "ms"),
    ("Flash text (KB)",           lambda v: (val(v,"memory.text_bytes") or 0)/1024,             "KB"),
    ("Static SRAM (KB)",          lambda v: (val(v,"static_sram_usage.static_sram_bytes") or 0)/1024, "KB"),
    ("ELF size (KB)",             lambda v: (val(v,"elf_size_bytes") or 0)/1024,                "KB"),
    ("Weights (KB)",              lambda v: (val(v,"model_info.weights_bytes") or 0)/1024,       "KB"),
    ("Activations (KB)",          lambda v: (val(v,"model_info.activations_bytes") or 0)/1024,   "KB"),
    ("MAC ops/inf (M)",           lambda v: (val(v,"compute_estimate.mac_ops_per_inference") or 0)/1e6, "M"),
    ("MOPS/s",                    lambda v: val(v,"compute_estimate.mops_per_second"),           "MOPS/s"),
    ("Current (mA)",              lambda v: val(v,"energy_estimate.current_ma"),                 "mA"),
    ("Power (mW)",                lambda v: (val(v,"energy_estimate.power_w") or 0)*1000,        "mW"),
    ("Energy/inf (µJ)",           lambda v: val(v,"energy_estimate.energy_per_inference_uj"),    "µJ"),
    ("MAC ops/J (M)",             lambda v: (val(v,"energy_estimate.mac_ops_per_joule") or 0)/1e6, "M MAC/J"),
]

with open(OUT/"comparison_table.csv","w",newline="") as f:
    w = csv.writer(f)
    w.writerow(["Metric","Unit"] + SHORT)
    for name, fn, unit in METRICS_TABLE:
        row = [name, unit]
        for v in VARIANTS:
            try: row.append(fn(v))
            except: row.append("—")
        w.writerow(row)
print("✓ comparison_table.csv")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 1 — Latency & Throughput (log scale — 215× range)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("STM32U5 vs MAX78000 — Latency & Throughput  (ai85kws20netv3)",
             fontsize=12, fontweight="bold")

lats  = [val(v,"cnn_latency_ms_avg") for v in VARIANTS]
tputs = [val(v,"inferences_per_second_from_cnn_time") for v in VARIANTS]
e2e   = [val(v,"estimated_end_to_end_lower_bound_ms") for v in VARIANTS]

log_bar(axes[0], lats,  LABELS, COLORS, "Inference Latency (avg)", "Latency", "ms")
log_bar(axes[1], tputs, LABELS, COLORS, "Throughput", "Inferences/s", "inf/s")
axes[1].invert_yaxis() if False else None  # higher=better, no invert
# for throughput highlight max
best_tp = max(tputs)
for bar, v in zip(axes[1].patches, tputs):
    clr = "#dc2626" if v == best_tp else "black"
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.15,
                 fmt_val(v), ha="center", va="bottom", fontsize=7.5,
                 color=clr, fontweight="bold" if v == best_tp else "normal")
log_bar(axes[2], e2e,   LABELS, COLORS, "End-to-End Lower Bound", "Time", "ms")

for ax in axes:
    ax.axvline(2.5, color="grey", linestyle="--", linewidth=1, alpha=0.5)

fig.tight_layout()
fig.savefig(OUT/"fig1_latency_throughput.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("✓ fig1_latency_throughput.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 2 — Energy (log scale — 111× range)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("STM32U5 vs MAX78000 — Power & Energy  (ai85kws20netv3)",
             fontsize=12, fontweight="bold")

energies = [val(v,"energy_estimate.energy_per_inference_uj") for v in VARIANTS]
powers   = [(val(v,"energy_estimate.power_w") or 0)*1000 for v in VARIANTS]
mac_j    = [(val(v,"energy_estimate.mac_ops_per_joule") or 0)/1e6 for v in VARIANTS]

log_bar(axes[0], energies, LABELS, COLORS, "Energy per Inference", "Energy", "µJ")
lin_bar(axes[1], powers,   LABELS, COLORS, "Power Draw", "Power", "mW")
lin_bar(axes[2], mac_j,    LABELS, COLORS, "Compute Efficiency", "M MAC/J", highlight_min=False)

for ax in axes:
    ax.axvline(2.5, color="grey", linestyle="--", linewidth=1, alpha=0.5)

fig.tight_layout()
fig.savefig(OUT/"fig2_power_energy.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("✓ fig2_power_energy.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 3 — Memory footprint (log scale)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
fig.suptitle("STM32U5 vs MAX78000 — Memory Footprint  (ai85kws20netv3)",
             fontsize=12, fontweight="bold")

flash   = [(val(v,"memory.text_bytes") or 0)/1024    for v in VARIANTS]
sram    = [(val(v,"static_sram_usage.static_sram_bytes") or 0)/1024 for v in VARIANTS]
weights = [(val(v,"model_info.weights_bytes") or 0)/1024 for v in VARIANTS]
elf     = [(val(v,"elf_size_bytes") or 0)/1024       for v in VARIANTS]

log_bar(axes[0], flash,   LABELS, COLORS, "Flash (text segment)", "Size", "KB")
log_bar(axes[1], sram,    LABELS, COLORS, "Static SRAM", "Size", "KB")
log_bar(axes[2], weights, LABELS, COLORS, "Model Weights", "Size", "KB")
log_bar(axes[3], elf,     LABELS, COLORS, "ELF Size", "Size", "KB")

for ax in axes:
    ax.axvline(2.5, color="grey", linestyle="--", linewidth=1, alpha=0.5)

fig.tight_layout()
fig.savefig(OUT/"fig3_memory.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("✓ fig3_memory.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 4 — Grouped comparison: v0 float32 vs v1 INT8 side-by-side per platform
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Platform × Version Grouped Comparison  (ai85kws20netv3)",
             fontsize=13, fontweight="bold")

# Use only versions present on both platforms for fair comparison
CROSS = [
    ("STM32U5 v0", VARIANTS[0]),
    ("STM32U5 v1", VARIANTS[1]),
    ("STM32U5 v2", VARIANTS[2]),
    ("MAX78000 v0", VARIANTS[3]),
    ("MAX78000 v1", VARIANTS[4]),
]
xlabels = [c[0] for c in CROSS]
xcolors  = [c[1]["color"] for c in CROSS]

def grouped_bar(ax, data_list, title, ylabel, unit="", log=False, higher_better=False):
    safe = [v if (v is not None and v > 0) else float("nan") for v in data_list]
    x = np.arange(len(safe))
    bars = ax.bar(x, safe, color=xcolors, edgecolor="white", linewidth=0.8, zorder=3)
    data_list = safe
    if log: ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=7.5, rotation=15, ha="right")
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.set_ylabel(f"{ylabel} [{unit}]" if unit else ylabel, fontsize=8)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines[["top","right"]].set_visible(False)
    real = [v for v in data_list if v == v and v is not None]
    best = (max(real) if higher_better else min(real)) if real else None
    for bar, v in zip(bars, data_list):
        if v != v: continue
        clr = "#dc2626" if v == best else "black"
        y = bar.get_height()*(1.4 if log else 1.02)
        ax.text(bar.get_x()+bar.get_width()/2, y, fmt_val(v),
                ha="center", va="bottom", fontsize=7,
                color=clr, fontweight="bold" if v == best else "normal")
    ax.axvline(2.5, color="grey", linestyle="--", linewidth=1.2, alpha=0.6)

grouped_bar(axes[0,0],
    [val(v[1],"cnn_latency_ms_avg") for v in CROSS],
    "Inference Latency (avg)", "ms", log=True)
grouped_bar(axes[0,1],
    [val(v[1],"energy_estimate.energy_per_inference_uj") for v in CROSS],
    "Energy per Inference", "µJ", log=True)
grouped_bar(axes[0,2],
    [(val(v[1],"energy_estimate.mac_ops_per_joule") or 0)/1e6 for v in CROSS],
    "Compute Efficiency", "M MAC/J", higher_better=True)
grouped_bar(axes[1,0],
    [(val(v[1],"memory.text_bytes") or 0)/1024 for v in CROSS],
    "Flash (text)", "KB", log=True)
grouped_bar(axes[1,1],
    [(val(v[1],"static_sram_usage.static_sram_bytes") or 0)/1024 for v in CROSS],
    "Static SRAM", "KB", log=True)
grouped_bar(axes[1,2],
    [(val(v[1],"model_info.weights_bytes") or 0)/1024 for v in CROSS],
    "Model Weights", "KB", log=True)

fig.tight_layout()
fig.savefig(OUT/"fig4_grouped_comparison.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("✓ fig4_grouped_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 5 — Radar chart (normalized, outer=better)
# ─────────────────────────────────────────────────────────────────────────────
RADAR = [
    ("Latency\n(lower=better)",    "cnn_latency_ms_avg",                            True),
    ("Energy/inf\n(lower=better)", "energy_estimate.energy_per_inference_uj",       True),
    ("Flash\n(lower=better)",      "memory.text_bytes",                             True),
    ("SRAM\n(lower=better)",       "static_sram_usage.static_sram_bytes",           True),
    ("Weights\n(lower=better)",    "model_info.weights_bytes",                      True),
    ("Throughput\n(higher=better)","inferences_per_second_from_cnn_time",           False),
    ("MAC/J\n(higher=better)",     "energy_estimate.mac_ops_per_joule",             False),
]
N = len(RADAR)
angles = [n / N * 2 * math.pi for n in range(N)] + [0]

fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
ax.set_theta_offset(math.pi/2)
ax.set_theta_direction(-1)
ax.set_xticks(angles[:-1])
ax.set_xticklabels([r[0] for r in RADAR], fontsize=9)
ax.set_ylim(0, 1)
ax.set_yticks([0.25,0.5,0.75,1.0])
ax.set_yticklabels(["25%","50%","75%","100%"], fontsize=7)
ax.grid(color="grey", alpha=0.3)

for v in VARIANTS:
    normalized = []
    for _, key, lower_better in RADAR:
        all_vals = [val(vv, key) for vv in VARIANTS if val(vv, key) is not None]
        vv = val(v, key)
        if vv is None or not all_vals:
            normalized.append(0.5); continue
        mn, mx = min(all_vals), max(all_vals)
        if mx == mn:
            normalized.append(1.0)
        elif lower_better:
            normalized.append(1.0 - (vv-mn)/(mx-mn))
        else:
            normalized.append((vv-mn)/(mx-mn))
    normalized += normalized[:1]
    ax.plot(angles, normalized, "o-", linewidth=2, color=v["color"],
            label=v["short"], markersize=5)
    ax.fill(angles, normalized, alpha=0.08, color=v["color"])

ax.set_title("Multi-metric Radar — STM32U5 vs MAX78000\n(outer = better)",
             fontsize=12, fontweight="bold", pad=25)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
fig.tight_layout()
fig.savefig(OUT/"fig5_radar.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("✓ fig5_radar.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 6 — Scatter: Latency vs Energy (both log) — all variants
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
for v in VARIANTS:
    lat = val(v,"cnn_latency_ms_avg")
    eng = val(v,"energy_estimate.energy_per_inference_uj")
    ax.scatter(lat, eng, color=v["color"], marker=v["marker"],
               s=120, zorder=5, label=v["short"])
    ax.annotate(v["short"], (lat, eng),
                textcoords="offset points", xytext=(8, 4), fontsize=8,
                color=v["color"], fontweight="bold")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Inference Latency (ms)", fontsize=10)
ax.set_ylabel("Energy per Inference (µJ)", fontsize=10)
ax.set_title("Latency vs Energy Trade-off\n(lower-left = better)",
             fontsize=11, fontweight="bold")
ax.grid(True, alpha=0.3, which="both")
ax.spines[["top","right"]].set_visible(False)
ax.legend(fontsize=9, loc="upper left")
# annotate ideal direction
ax.annotate("", xy=(ax.get_xlim()[0]*1.5, ax.get_ylim()[0]*1.5),
            xytext=(ax.get_xlim()[0]*5, ax.get_ylim()[0]*8),
            arrowprops=dict(arrowstyle="->", color="grey", lw=1.5))
ax.text(ax.get_xlim()[0]*2.5, ax.get_ylim()[0]*2, "better", fontsize=8,
        color="grey", rotation=45)
fig.tight_layout()
fig.savefig(OUT/"fig6_scatter_latency_energy.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("✓ fig6_scatter_latency_energy.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 7 — Speedup & energy reduction relative to U5-v0 baseline
# ─────────────────────────────────────────────────────────────────────────────
base_lat = val(VARIANTS[0],"cnn_latency_ms_avg")
base_eng = val(VARIANTS[0],"energy_estimate.energy_per_inference_uj")
base_fla = val(VARIANTS[0],"memory.text_bytes")

speedups  = [base_lat / val(v,"cnn_latency_ms_avg") for v in VARIANTS]
esavings  = [base_eng / val(v,"energy_estimate.energy_per_inference_uj") for v in VARIANTS]
fsavings  = [base_fla / (val(v,"memory.text_bytes") or 1) for v in VARIANTS]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Improvement relative to STM32U5-v0 baseline  (ai85kws20netv3)",
             fontsize=12, fontweight="bold")

for ax, data, title, ylabel in [
    (axes[0], speedups, "Latency speedup vs U5-v0", "× faster"),
    (axes[1], esavings, "Energy reduction vs U5-v0", "× less energy"),
    (axes[2], fsavings, "Flash reduction vs U5-v0",  "× smaller"),
]:
    x = np.arange(len(VARIANTS))
    bars = ax.bar(x, data, color=COLORS, edgecolor="white", linewidth=0.8, zorder=3)
    ax.axhline(1, color="black", linestyle="--", linewidth=1, label="baseline")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT, fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines[["top","right"]].set_visible(False)
    ax.axvline(2.5, color="grey", linestyle="--", linewidth=1, alpha=0.5)
    best = max(data)
    for bar, v in zip(bars, data):
        clr = "#dc2626" if v == best else "black"
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.2,
                f"{v:.1f}×", ha="center", va="bottom", fontsize=8,
                color=clr, fontweight="bold" if v == best else "normal")

fig.tight_layout()
fig.savefig(OUT/"fig7_improvement_vs_u5v0.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("✓ fig7_improvement_vs_u5v0.png")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 8 — Equivalent versions side-by-side: U5-v0 vs MX-v0, U5-v1 vs MX-v1
# ─────────────────────────────────────────────────────────────────────────────
PAIRS = [
    ("v0 float32", VARIANTS[0], VARIANTS[3]),
    ("v1 INT8",    VARIANTS[1], VARIANTS[4]),
]
METRICS_PAIR = [
    ("Latency (ms)",    "cnn_latency_ms_avg",                            True,  True),
    ("Energy (µJ)",     "energy_estimate.energy_per_inference_uj",       True,  True),
    ("Power (mW)",      "energy_estimate.power_w",                       True,  True),
    ("Flash (KB)",      "memory.text_bytes",                             True,  False),
    ("SRAM (KB)",       "static_sram_usage.static_sram_bytes",           True,  False),
    ("Weights (KB)",    "model_info.weights_bytes",                      False, False),
]
fig, axes = plt.subplots(len(PAIRS), len(METRICS_PAIR),
                          figsize=(len(METRICS_PAIR)*3, len(PAIRS)*3.5))
fig.suptitle("Equivalent Version Pairs: STM32U5 vs MAX78000",
             fontsize=12, fontweight="bold")

for ri, (pair_label, u5v, mxv) in enumerate(PAIRS):
    for ci, (metric_label, key, use_log, scale_to_kb) in enumerate(METRICS_PAIR):
        ax = axes[ri, ci]
        raw = [val(u5v, key), val(mxv, key)]
        # scale
        if scale_to_kb and "bytes" in key:
            data = [r/1024 if r else 0 for r in raw]
        elif key == "energy_estimate.power_w":
            data = [(r or 0)*1000 for r in raw]
        else:
            data = [r or 0 for r in raw]
        bars = ax.bar([0,1], data,
                      color=[u5v["color"], mxv["color"]],
                      edgecolor="white", linewidth=0.8, zorder=3)
        if use_log and all(d > 0 for d in data):
            ax.set_yscale("log")
        ax.set_xticks([0,1])
        ax.set_xticklabels([u5v["short"], mxv["short"]], fontsize=8)
        ax.set_title(f"{pair_label}\n{metric_label}", fontsize=8, fontweight="bold")
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.spines[["top","right"]].set_visible(False)
        best = min(data)
        for bar, v in zip(bars, data):
            clr = "#dc2626" if v == best else "black"
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()*(1.2 if use_log else 1.02),
                    fmt_val(v), ha="center", va="bottom", fontsize=7.5,
                    color=clr, fontweight="bold" if v == best else "normal")

fig.tight_layout()
fig.savefig(OUT/"fig8_pair_comparison.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("✓ fig8_pair_comparison.png")

print(f"\nAll outputs saved to: {OUT}")
