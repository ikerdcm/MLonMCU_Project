import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

COLORS = {"u5":"#1F77B4","max":"#FF7F0E","coral":"#D62728","gap9":"#2CA02C"}

plt.rcParams.update({
    "font.size": 67,
    "axes.titlesize": 82,
    "axes.labelsize": 74,
    "xtick.labelsize": 62,
    "ytick.labelsize": 58,
    "legend.fontsize": 58,
    "figure.facecolor": "white",
})

CAPACITY = {
    "u5":    {"flash": 4096,  "sram": 2560},
    "max":   {"flash": 512,   "sram": 128},
    "coral": {"flash": 4096,  "sram": 8192},
    "gap9":  {"flash": 1536,  "sram": 1536},
}

# (mcu, config_label, flash_used_kib, sram_used_kib | None)
DSCNN = [
    ("max",   "int8-accel",    139.5, 38.0),
    ("max",   "fp32-cpu",      208.8, 99.7),
    ("u5",    "int8-cpu",      150.5, 82.8),
    ("u5",    "fp32-cpu",      220.8, 125.8),
    ("coral", "int8-accel",    144.6, 62.0),
    ("coral", "fp32-cpu",      144.5, None),
    ("coral", "prune f64b4",   120.6, 46.5),
    ("coral", "prune f32b6",   108.6, 39.0),
    ("coral", "prune f32b4",   92.6,  30.5),
    ("coral", "distill f64b4", 120.6, 46.5),
    ("coral", "distill f32b4", 92.6,  30.5),
    ("gap9",  "fp32-cluster",  46.48, 134.08),
    ("gap9",  "int8-cluster",  46.69, 32.86),
    ("gap9",  "int8-ne16",     36.38, 20.02),
]

AI85 = [
    ("max",  "int8-cpu",       397.8, 66.7),
    ("max",  "int8-accel",     381.1, 37.7),
    ("max",  "int8-accel-w90", 357.5, 37.8),
    ("u5",   "fp32-cpu",       724.1, 238.1),
    ("u5",   "int8-cpu",       250.1, 180.8),
    ("u5",   "int8-cpu-w90",   226.7, 176.8),
]


def build_y_positions(entries):
    BH, INNER, OUTER, SEP = 0.9, 0.08, 0.18, 0.45
    STEP = 2*BH + INNER + OUTER
    ys, y, prev = [], 0.0, None
    for mcu, *_ in entries:
        if prev is not None and mcu != prev:
            y += SEP
        ys.append(y)
        y += STEP
        prev = mcu
    return ys, STEP, BH, INNER


def draw_panel(ax, entries, title):
    ys, STEP, BH, INNER = build_y_positions(entries)

    for i, (mcu, label, flash_used, sram_used) in enumerate(entries):
        c = COLORS[mcu]
        cap = CAPACITY[mcu]
        y_fl = ys[i] + BH/2 + INNER/2
        y_sr = ys[i] - BH/2 - INNER/2

        pct_fl = flash_used / cap["flash"] * 100
        ax.barh(y_fl, 100, BH, color="#E8E8E8", left=0, zorder=2)
        ax.barh(y_fl, pct_fl, BH, color=c, left=0, zorder=3, alpha=0.88)
        ax.text(102, y_fl, f"{pct_fl:.0f}%",
                va="center", ha="left", fontsize=52, color=c, fontweight="bold")

        if sram_used is not None:
            pct_sr = sram_used / cap["sram"] * 100
            ax.barh(y_sr, 100, BH, color="#E8E8E8", left=0, zorder=2)
            ax.barh(y_sr, pct_sr, BH, color=c, left=0, zorder=3,
                    alpha=0.42, hatch="///", edgecolor=c, linewidth=0.6)
            ax.text(102, y_sr, f"{pct_sr:.0f}%",
                    va="center", ha="left", fontsize=52, color=c)
        else:
            ax.text(102, y_sr, "N/A",
                    va="center", ha="left", fontsize=52, color="#999", style="italic")

        if i > 0 and entries[i][0] != entries[i-1][0]:
            mid = (ys[i] + ys[i-1] + STEP) / 2
            ax.axhline(mid, color="#CCCCCC", lw=3.0, zorder=1)

    ax.axvline(100, color="#333", lw=4.0, linestyle="--", alpha=0.7, zorder=5)
    ax.set_yticks(ys)
    ax.set_yticklabels([e[1] for e in entries])
    ax.set_xlim(0, 130)
    ax.set_ylim(-STEP/2, max(ys) + STEP)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_xlabel("Fill Level (%)", labelpad=30)
    ax.set_title(title, pad=40)
    ax.grid(axis="x", linestyle="--", alpha=0.40, linewidth=3.0, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=20)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(71, 40),
                                gridspec_kw={"width_ratios": [1, 1]})
draw_panel(ax1, DSCNN, "DS-CNN")
draw_panel(ax2, AI85,  "AI85KWS20NetV3")

mcu_h = [mpatches.Patch(color=COLORS[k], label=n)
         for k, n in [("u5","STM32U5"),("max","MAX78000"),("coral","Coral"),("gap9","GAP9")]]
style_h = [
    mpatches.Patch(color="#888", alpha=0.88,             label="Flash  (solid)"),
    mpatches.Patch(facecolor="#888", alpha=0.42, hatch="///",
                   edgecolor="#888",                      label="SRAM  (hatched)"),
    mpatches.Patch(color="#E8E8E8",                       label="Free headroom"),
]
fig.legend(handles=mcu_h + style_h, loc="upper center", ncol=4,
           framealpha=0.92, bbox_to_anchor=(0.5, 1.02))


fig.tight_layout(pad=3.0)
fig.savefig(Path(__file__).with_suffix(".png"), dpi=150, bbox_inches="tight")
print("Saved:", Path(__file__).with_suffix(".png"))
