import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
from pathlib import Path

COLORS = {"u5":"#1F77B4","max":"#FF7F0E","coral":"#D62728","gap9":"#2CA02C"}

plt.rcParams.update({
    "font.size": 67,
    "axes.titlesize": 82,
    "axes.labelsize": 74,
    "xtick.labelsize": 62,
    "ytick.labelsize": 62,
    "legend.fontsize": 58,
    "figure.facecolor": "white",
})

# (mcu, config_label, mcu_acc%, energy_uJ, flash_kib, model, above)
# Color = MCU, shape = model — label is config only
# above=True → label above dot, False → label below dot
DATA = [
    # far left — isolated low accuracy
    ("max",  "int8-accel",      67.36,  23.62,     139.5, "DS-CNN", False),
    # far left — high accuracy
    ("gap9", "int8-ne16",       91.23,  34.32,     36.38, "DS-CNN", True),
    # MAX AI85 cluster (~250-285 µJ)
    ("max",  "int8-accel-w90",  83.47,  250.94,    357.5, "AI85",   False),
    ("max",  "int8-accel",      91.25,  283.39,    381.1, "AI85",   True),
    # GAP9 int8-cluster
    ("gap9", "int8-cluster",    91.45,  480.11,    46.69, "DS-CNN", False),
    # dense Coral pruning cluster (~1535-1932 µJ)
    ("coral","prune f32b4",     62.50,  1535.30,   92.6,  "DS-CNN", False),
    ("coral","distill f32b4",   77.78,  1584.47,   92.6,  "DS-CNN", True),
    ("coral","prune f64b4",     88.89,  1603.45,   120.6, "DS-CNN", False),
    ("coral","distill f64b4",   90.28,  1610.40,   120.6, "DS-CNN", True),
    ("coral","prune f32b6",     72.22,  1795.69,   108.6, "DS-CNN", False),
    ("coral","int8-accel",      91.67,  1932.03,   144.6, "DS-CNN", True),
    # GAP9 fp32-cluster
    ("gap9", "fp32-cluster",    91.62,  3893.85,   46.48, "DS-CNN", True),
    # U5 cluster (~11k-32k µJ)
    ("u5",   "int8-cpu-w90",    84.27,  11475.73,  226.7, "AI85",   False),
    ("u5",   "int8-cpu",        90.01,  12751.97,  250.1, "AI85",   True),
    ("u5",   "int8-cpu",        91.67,  13466.62,  150.5, "DS-CNN", False),
    ("u5",   "fp32-cpu",        93.75,  18905.89,  220.8, "DS-CNN", True),
    ("u5",   "fp32-cpu",        91.23,  31394.71,  724.1, "AI85",   False),
    # MAX right cluster (~50k-97k µJ)
    ("max",  "int8-cpu",        89.43,  49541.78,  397.8, "AI85",   False),
    ("max",  "fp32-cpu",        93.75,  97246.96,  208.8, "DS-CNN", True),
    # Coral fp32-cpu — rightmost
    ("coral","fp32-cpu",        90.97,  316970.53, 144.5, "DS-CNN", False),
]

BUBBLE  = 40.0
MARKER  = {"DS-CNN": "o", "AI85": "^"}
Y_OFF   = 2.5   # percentage-point offset above/below dot centre

fig, ax = plt.subplots(figsize=(71, 40))

for mcu, label, acc, energy, flash, model, above in DATA:
    c = COLORS[mcu]
    ax.scatter(energy, acc, s=flash * BUBBLE, color=c, alpha=0.82,
               edgecolors=c, linewidths=2.5, marker=MARKER[model], zorder=4)

    if above:
        ax.text(energy, acc + Y_OFF, label,
                fontsize=56, color=c, fontweight="bold",
                ha="center", va="bottom", zorder=10)
    else:
        ax.text(energy, acc - Y_OFF, label,
                fontsize=56, color=c, fontweight="bold",
                ha="center", va="top", zorder=10)

ax.set_xscale("log")
ax.set_xticks([25, 500, 5000, 50000, 300000])
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _:
    f"{int(v/1000)}k" if v >= 1000 else f"{v:.0f}"))
ax.set_yticks([65, 80, 95])
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))

ax.set_xlabel("Energy per Inference  (µJ, log scale)", labelpad=16)
ax.set_ylabel("On-Device MCU Accuracy (%)", labelpad=16)
ax.set_title("Accuracy vs Inference Energy\n"
             "(bubble size ∝ on-device model footprint in KiB)", pad=22)
ax.set_ylim(54, 102)
ax.grid(True, which="both", linestyle="--", alpha=0.65, linewidth=2.0, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

color_h = [mpatches.Patch(color=COLORS[k], label=n)
           for k, n in [("u5","STM32U5"),("max","MAX78000"),
                         ("coral","Coral"),("gap9","GAP9")]]
marker_h = [
    mlines.Line2D([], [], color="#555", marker="o", markersize=36,
                  linestyle="None", label="DS-CNN"),
    mlines.Line2D([], [], color="#555", marker="^", markersize=36,
                  linestyle="None", label="AI85KWS20NetV3"),
]
ref_h = [ax.scatter([], [], s=r * BUBBLE, color="gray", alpha=0.55,
                    label=f"{r} KiB")
         for r in [100, 300, 600]]

leg1 = ax.legend(handles=color_h + marker_h, title="MCU / Model",
                 loc="lower right", framealpha=0.92)
ax.legend(handles=ref_h, title="Model footprint",
          loc="upper left", framealpha=0.92)
ax.add_artist(leg1)

fig.tight_layout(pad=2.5)
fig.savefig(Path(__file__).with_suffix(".png"), dpi=150, bbox_inches="tight")
print("Saved:", Path(__file__).with_suffix(".png"))
