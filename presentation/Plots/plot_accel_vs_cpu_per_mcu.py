import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
from pathlib import Path

COLORS = {"u5":"#1F77B4","max":"#FF7F0E","coral":"#D62728","gap9":"#2CA02C"}
GRAY   = "#C0C0C0"

plt.rcParams.update({
    "font.size": 67,
    "axes.titlesize": 82,
    "axes.labelsize": 74,
    "xtick.labelsize": 62,
    "ytick.labelsize": 62,
    "legend.fontsize": 58,
    "figure.facecolor": "white",
})

# (mcu, config_label, mcu_acc%, flash_kib, hw_accel, model, above)
DATA = [
    ("gap9", "int8-ne16",      91.23, 36.38, True,  "DS-CNN", False),
    ("gap9", "fp32-cluster",   91.62, 46.48, False, "DS-CNN", True),
    ("gap9", "int8-cluster",   91.45, 46.69, False, "DS-CNN", False),
    ("coral","prune f32b4",    62.50, 92.6,  True,  "DS-CNN", False),
    ("coral","distill f32b4",  77.78, 92.6,  True,  "DS-CNN", True),
    ("coral","prune f32b6",    72.22, 108.6, True,  "DS-CNN", False),
    ("coral","prune f64b4",    88.89, 120.6, True,  "DS-CNN", False),
    ("coral","distill f64b4",  90.28, 120.6, True,  "DS-CNN", True),
    ("max",  "int8-accel",     67.36, 139.5, True,  "DS-CNN", False),
    ("coral","fp32-cpu",       90.97, 144.5, False, "DS-CNN", False),
    ("coral","int8-accel",     91.67, 144.6, True,  "DS-CNN", True),
    ("u5",   "int8-cpu",       91.67, 150.5, False, "DS-CNN", False),
    ("max",  "fp32-cpu",       93.75, 208.8, False, "DS-CNN", True),
    ("u5",   "fp32-cpu",       93.75, 220.8, False, "DS-CNN", False),
    ("u5",   "int8-cpu-w90",   84.27, 226.7, False, "AI85",   False),
    ("u5",   "int8-cpu",       90.01, 250.1, False, "AI85",   True),
    ("max",  "int8-accel-w90", 83.47, 357.5, True,  "AI85",   False),
    ("max",  "int8-accel",     91.25, 381.1, True,  "AI85",   True),
    ("max",  "int8-cpu",       89.43, 397.8, False, "AI85",   False),
    ("u5",   "fp32-cpu",       91.23, 724.1, False, "AI85",   True),
]

MSIZE  = 6000
MARKER = {"DS-CNN": "o", "AI85": "^"}
Y_OFF  = 2.5

MCU_NAMES = {
    "u5":    "STM32U5",
    "max":   "MAX78000",
    "coral": "Coral",
    "gap9":  "GAP9",
}

OUT_DIR = Path(__file__).parent


def draw_plot(highlight_mcu: str) -> None:
    fig, ax = plt.subplots(figsize=(71, 40))

    # Gray dots first (behind) so highlighted ones render on top
    for mcu, label, acc, flash, accel, model, above in DATA:
        if mcu == highlight_mcu:
            continue
        c = GRAY
        if accel:
            ax.scatter(flash, acc, s=MSIZE, color=c, marker=MARKER[model],
                       zorder=3, alpha=0.60, linewidths=0)
        else:
            ax.scatter(flash, acc, s=MSIZE, facecolors="white", edgecolors=c,
                       marker=MARKER[model], zorder=3, linewidths=8.0, alpha=0.70)
        if above:
            ax.text(flash, acc + Y_OFF, label,
                    fontsize=56, color=c, fontweight="bold",
                    ha="center", va="bottom", zorder=5)
        else:
            ax.text(flash, acc - Y_OFF, label,
                    fontsize=56, color=c, fontweight="bold",
                    ha="center", va="top", zorder=5)

    # Highlighted MCU dots on top
    for mcu, label, acc, flash, accel, model, above in DATA:
        if mcu != highlight_mcu:
            continue
        c = COLORS[mcu]
        if accel:
            ax.scatter(flash, acc, s=MSIZE, color=c, marker=MARKER[model],
                       zorder=4, alpha=0.88, linewidths=0)
        else:
            ax.scatter(flash, acc, s=MSIZE, facecolors="white", edgecolors=c,
                       marker=MARKER[model], zorder=4, linewidths=8.0, alpha=0.95)
        if above:
            ax.text(flash, acc + Y_OFF, label,
                    fontsize=56, color=c, fontweight="bold",
                    ha="center", va="bottom", zorder=10)
        else:
            ax.text(flash, acc - Y_OFF, label,
                    fontsize=56, color=c, fontweight="bold",
                    ha="center", va="top", zorder=10)

    ax.set_xscale("log")
    x_ticks = [40, 75, 125, 200, 350, 600]
    ax.set_xticks(x_ticks)
    ax.set_yticks([50, 70, 90])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax.set_xlabel("On-Device Model Footprint  (KiB, log scale)\n"
                  "Flash for STM32U5 / MAX78000 / Coral  —  L2 SRAM for GAP9 (no Flash)",
                  labelpad=16)
    ax.set_ylabel("On-Device MCU Accuracy (%)", labelpad=16)
    ax.set_title(f"HW-Accelerated vs CPU-Only  —  {MCU_NAMES[highlight_mcu]} highlighted",
                 pad=22)
    ax.set_ylim(44, 104)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.grid(True, linestyle="--", alpha=0.65, linewidth=2.0, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    mcu_h = [mpatches.Patch(color=COLORS[k] if k == highlight_mcu else GRAY,
                             label=n)
             for k, n in [("u5","STM32U5"),("max","MAX78000"),
                           ("coral","Coral"),("gap9","GAP9")]]
    model_h = [
        mlines.Line2D([], [], color="#555", marker="o", markersize=36,
                      linestyle="None", label="DS-CNN"),
        mlines.Line2D([], [], color="#555", marker="^", markersize=36,
                      linestyle="None", label="AI85KWS20NetV3"),
    ]
    fill_h = [
        mlines.Line2D([], [], color="#555", marker="o", markersize=36,
                      linestyle="None", markerfacecolor="#555",
                      label="HW NN accelerator"),
        mlines.Line2D([], [], color="#555", marker="o", markersize=36,
                      linestyle="None", markerfacecolor="white",
                      markeredgewidth=8.0, label="CPU-only"),
    ]

    leg1 = ax.legend(handles=mcu_h + model_h, title="MCU / Model",
                     loc="lower right", framealpha=0.92)
    ax.legend(handles=fill_h, loc="upper right", framealpha=0.92)
    ax.add_artist(leg1)

    fig.tight_layout(pad=2.5)
    out = OUT_DIR / f"plot_accel_vs_cpu_{highlight_mcu}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out)


for mcu in ["u5", "max", "coral", "gap9"]:
    draw_plot(mcu)
