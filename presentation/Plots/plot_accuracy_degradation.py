import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

COLORS = {"u5":"#1F77B4","max":"#FF7F0E","coral":"#D62728","gap9":"#2CA02C","pc":"#333333"}

plt.rcParams.update({
    "font.size": 67,
    "axes.titlesize": 82,
    "axes.labelsize": 74,
    "xtick.labelsize": 62,
    "ytick.labelsize": 62,
    "legend.fontsize": 58,
    "figure.facecolor": "white",
})

DSCNN_GROUPS = [
    ("fp32-cpu",     [("pc",92.4,""),("coral",90.97,""),("max",93.75,""),("u5",93.75,""),("gap9",91.62,"")]),
    ("int8-cpu",     [("pc",92.0,""),("u5",91.67,"")]),
    ("int8-accel",   [("pc",92.0,""),("coral",91.67,""),("max",67.36,"")]),
    ("int8-cluster", [("pc",92.0,""),("gap9",91.45,"")]),
    ("int8-ne16",    [("pc",92.0,""),("gap9",91.23,"")]),
    ("prune\nf64b4", [("pc",89.7,""),("coral",88.89,"")]),
    ("prune\nf32b6", [("pc",76.7,""),("coral",72.22,"")]),
    ("prune\nf32b4", [("pc",65.1,""),("coral",62.50,"")]),
    ("distill\nf64b4",[("pc",90.8,""),("coral",90.28,"")]),
    ("distill\nf32b4",[("pc",76.5,""),("coral",77.78,"")]),
]

AI85_GROUPS = [
    ("fp32-cpu",   [("pc",91.17,""),("u5",91.23,"")]),
    ("int8-cpu",   [("pc",90.80,""),("max",89.43,""),("u5",90.01,"")]),
    ("int8-accel", [("pc",90.80,""),("max",91.25,"")]),
    ("w90 pruned", [("pc",85.10,""),("max",83.47,""),("u5",84.27,"")]),
]


def draw_groups(ax, groups, bw=0.45, gap=0.95, y_floor=55):
    x = 0.0
    ticks, xlabels, stars = [], [], []
    for g_label, members in groups:
        n = len(members)
        offsets = np.linspace(-(n-1)*bw/2, (n-1)*bw/2, n)
        for off, (mcu, acc, note) in zip(offsets, members):
            ax.bar(x+off, acc-y_floor, bw, bottom=y_floor,
                   color=COLORS[mcu], alpha=0.88 if mcu != "pc" else 1.0,
                   zorder=3, linewidth=0)
            if note:
                stars.append((x+off, acc, note))
        ticks.append(x); xlabels.append(g_label)
        x += 1.0 + gap
    ax.set_xticks(ticks)
    ax.set_xticklabels(xlabels, rotation=35, ha="right")
    for xp, acc, note in stars:
        ax.text(xp, acc+0.7, note, ha="center", va="bottom",
                fontsize=62, fontweight="bold")
    ax.set_xlim(-0.8, x - gap + 0.5)


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(71, 48),
                                gridspec_kw={"height_ratios":[3,2]})

ax1.set_title("DS-CNN: PC Model Accuracy vs On-Device Accuracy", pad=16)
draw_groups(ax1, DSCNN_GROUPS, y_floor=58)
ax1.set_ylabel("Accuracy (%)"); ax1.set_ylim(58, 100)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v:.0f}%"))
ax1.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

ax2.set_title("AI85KWS20NetV3: PC Model Accuracy vs On-Device Accuracy", pad=16)
draw_groups(ax2, AI85_GROUPS, y_floor=80)
ax2.set_ylabel("Accuracy (%)"); ax2.set_ylim(80, 96)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v:.0f}%"))
ax2.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

legend_handles = [mpatches.Patch(color=COLORS[k], label=n)
                  for k,n in [("pc","PC (model)"),("u5","STM32U5"),
                               ("max","MAX78000"),("coral","Coral"),("gap9","GAP9")]]
fig.legend(handles=legend_handles, loc="upper center", ncol=5,
           framealpha=0.92, bbox_to_anchor=(0.5, 1.01))

fig.tight_layout(pad=2.5)
fig.savefig(Path(__file__).with_suffix(".png"), dpi=150, bbox_inches="tight")
print("Saved:", Path(__file__).with_suffix(".png"))
