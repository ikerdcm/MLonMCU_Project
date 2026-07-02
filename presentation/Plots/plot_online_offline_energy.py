import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

COLORS = {"u5":"#1F77B4","max":"#FF7F0E","coral":"#D62728","gap9":"#2CA02C"}

plt.rcParams.update({
    "font.size": 44,
    "axes.titlesize": 54,
    "axes.labelsize": 48,
    "xtick.labelsize": 40,
    "ytick.labelsize": 40,
    "legend.fontsize": 38,
    "figure.facecolor": "white",
})

# (label, mcu, offline_uJ, online_uJ | None | "gap9_only")
ENTRIES = [
    ("U5\nfp32-cpu",     "u5",    16159.0,   16161.0),
    ("U5\nint8-cpu",     "u5",    4610.0,    4927.0),
    ("Coral\nfp32-cpu",  "coral", 316970.53, None),
    ("Coral\nint8-accel","coral", 1932.03,   None),
    ("MAX\nfp32-cpu",    "max",   97246.96,  None),
    ("MAX\nint8-accel",  "max",   23.62,     None),
    ("GAP9\nfp32-clust", "gap9",  3893.85,   "gap9_only"),
    ("GAP9\nint8-clust", "gap9",  480.11,    "gap9_only"),
    ("GAP9\nint8-ne16",  "gap9",  34.32,     "gap9_only"),
]

BW = 0.34
GAP = 0.55
positions = np.arange(len(ENTRIES)) * (1.0 + GAP)

fig, ax = plt.subplots(figsize=(26, 16))

PENDING_Y = 6.0

for i, (label, mcu, offline, online) in enumerate(ENTRIES):
    c = COLORS[mcu]
    xoff = positions[i] - BW / 2
    xon  = positions[i] + BW / 2

    ax.bar(xoff, offline, BW, color=c, alpha=0.92, zorder=3, linewidth=0)

    if online is None:
        ax.bar(xon, PENDING_Y, BW, color="lightgray", hatch="////",
               edgecolor="#888888", linewidth=1.5, zorder=3)
        ax.text(xon, PENDING_Y * 2.4, "pending", ha="center", va="bottom",
                fontsize=32, color="#888888", rotation=90, style="italic")
    elif online == "gap9_only":
        ax.text(xon, PENDING_Y * 2.4, "offline\nonly", ha="center", va="bottom",
                fontsize=30, color="#666666", style="italic")
    else:
        ax.bar(xon, online, BW, color=c, alpha=0.38, edgecolor=c,
               linewidth=2.5, zorder=3)
        delta = (online - offline) / offline * 100
        sign = "+" if delta >= 0 else ""
        y_top = max(offline, online) * 1.30
        ax.text(positions[i], y_top, f"{sign}{delta:.1f}%",
                ha="center", va="bottom", fontsize=34, color=c, fontweight="bold")

ax.set_yscale("log")
ax.set_ylim(bottom=3, top=3e6)
ax.set_ylabel("Energy / Inference  (µJ, log scale)", labelpad=14)
ax.set_title("Online vs Offline Inference Energy", pad=20)
ax.set_xticks(positions)
ax.set_xticklabels([e[0] for e in ENTRIES], rotation=0, ha="center")
ax.grid(axis="y", which="both", linestyle="--", alpha=0.35, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

mcu_p = [mpatches.Patch(color=COLORS[k], label=n)
         for k, n in [("u5","STM32U5"),("max","MAX78000"),("coral","Coral"),("gap9","GAP9")]]
style_p = [
    mpatches.Patch(color="#888888", alpha=0.92,   label="Offline"),
    mpatches.Patch(color="#888888", alpha=0.38,   label="Online"),
    mpatches.Patch(facecolor="lightgray", hatch="////",
                   edgecolor="#888888",            label="Pending"),
]
ax.legend(handles=mcu_p + style_p, loc="upper right", framealpha=0.92, ncol=2)
ax.annotate("GAP9 has no online mode.",
            xy=(0.01,0.02), xycoords="axes fraction",
            fontsize=32, color="#555", style="italic")

fig.tight_layout(pad=2.0)
fig.savefig(Path(__file__).with_suffix(".png"), dpi=150, bbox_inches="tight")
print("Saved:", Path(__file__).with_suffix(".png"))
