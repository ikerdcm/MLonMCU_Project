"""
Interactive analyser for the Experiments/ tree.

Layout expected:
    Experiments/
        General Profiling/<platform>/<model>/<online|offline>/<word>.json
        PWR Consumption/  <platform>/<model>/<online|offline>/<word>.csv

Two top-level modes:
    * Inspect  -> drill down to one (platform, model, mode, word) and run an analysis
    * Compare  -> pick a dimension (word / mode / platform / model), pick 2+ items
                  along it with the other dimensions pinned, and run an analysis

Dependencies: rich, questionary, pandas, matplotlib, numpy.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd
import matplotlib.pyplot as plt

import questionary
from questionary import Choice
from rich.console import Console
from rich.tree import Tree
from rich.panel import Panel
from rich.table import Table


# --------------------------------------------------------------------------- #
# Paths & discovery
# --------------------------------------------------------------------------- #

ROOT = Path(__file__).resolve().parent
GP_DIR = ROOT / "General Profiling"
PWR_DIR = ROOT / "PWR Consumption"

MODES = ("online", "offline")

console = Console()


@dataclass
class WordEntry:
    word: str
    json_path: Path | None = None
    csv_path: Path | None = None

    @property
    def has_json(self) -> bool:
        return self.json_path is not None and self.json_path.exists()

    @property
    def has_csv(self) -> bool:
        return self.csv_path is not None and self.csv_path.exists()


# nested dict: platform -> model -> mode -> word -> WordEntry
Catalog = dict[str, dict[str, dict[str, dict[str, WordEntry]]]]


def discover() -> Catalog:
    """Walk both GP and PWR subtrees and merge them into a single catalog."""
    catalog: Catalog = {}

    def ensure(platform: str, model: str, mode: str, word: str) -> WordEntry:
        return (
            catalog.setdefault(platform, {})
            .setdefault(model, {})
            .setdefault(mode, {})
            .setdefault(word, WordEntry(word=word))
        )

    if GP_DIR.is_dir():
        for json_path in GP_DIR.glob("*/*/*/*.json"):
            platform = json_path.parts[-4]
            model = json_path.parts[-3]
            mode = json_path.parts[-2]
            if mode not in MODES:
                continue
            word = json_path.stem
            entry = ensure(platform, model, mode, word)
            entry.json_path = json_path

    if PWR_DIR.is_dir():
        for csv_path in PWR_DIR.glob("*/*/*/*.csv"):
            platform = csv_path.parts[-4]
            model = csv_path.parts[-3]
            mode = csv_path.parts[-2]
            if mode not in MODES:
                continue
            word = csv_path.stem
            entry = ensure(platform, model, mode, word)
            entry.csv_path = csv_path

    return catalog


# --------------------------------------------------------------------------- #
# Selection / context model
# --------------------------------------------------------------------------- #

@dataclass
class Selection:
    """A single (platform, model, mode, word) tuple resolving to a WordEntry."""
    platform: str
    model: str
    mode: str
    word: str
    entry: WordEntry

    def label(self) -> str:
        return f"{self.platform} / {self.model} / {self.mode} / {self.word}"


@dataclass
class CompareSet:
    """A list of Selections plus the axis along which they differ."""
    axis: str  # "word" | "mode" | "platform" | "model"
    selections: list[Selection] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Tree rendering helpers
# --------------------------------------------------------------------------- #

def render_tree(catalog: Catalog, highlight: dict[str, str] | None = None) -> None:
    """Pretty-print the full catalog as a rich Tree, optionally highlighting a path."""
    highlight = highlight or {}
    tree = Tree("[bold cyan]Experiments[/]")
    for platform in sorted(catalog):
        p_label = f"[bold]{platform}[/]"
        if highlight.get("platform") == platform:
            p_label = f"[bold green]>>> {platform}[/]"
        p_node = tree.add(p_label)
        for model in sorted(catalog[platform]):
            m_label = model
            if highlight.get("platform") == platform and highlight.get("model") == model:
                m_label = f"[green]>>> {model}[/]"
            m_node = p_node.add(m_label)
            for mode in MODES:
                if mode not in catalog[platform][model]:
                    continue
                mo_label = mode
                if (
                    highlight.get("platform") == platform
                    and highlight.get("model") == model
                    and highlight.get("mode") == mode
                ):
                    mo_label = f"[green]>>> {mode}[/]"
                mo_node = m_node.add(mo_label)
                for word in sorted(catalog[platform][model][mode]):
                    e = catalog[platform][model][mode][word]
                    flags = []
                    flags.append("csv" if e.has_csv else "[dim]-csv[/]")
                    flags.append("json" if e.has_json else "[dim]-json[/]")
                    w_label = f"{word}  [dim]({', '.join(flags)})[/]"
                    if (
                        highlight.get("platform") == platform
                        and highlight.get("model") == model
                        and highlight.get("mode") == mode
                        and highlight.get("word") == word
                    ):
                        w_label = f"[green]>>> {word}[/]  [dim]({', '.join(flags)})[/]"
                    mo_node.add(w_label)
    console.print(tree)


# --------------------------------------------------------------------------- #
# Interactive pickers
# --------------------------------------------------------------------------- #

def _ask_select(message: str, options: Iterable[str]) -> str | None:
    options = list(options)
    if not options:
        console.print(f"[red]No options available for: {message}[/]")
        return None
    return questionary.select(message, choices=options).ask()


def _ask_checkbox(message: str, options: Iterable[str], min_items: int = 2) -> list[str] | None:
    options = list(options)
    if len(options) < min_items:
        console.print(f"[red]Need at least {min_items} options to compare ({len(options)} available).[/]")
        return None
    while True:
        picks = questionary.checkbox(
            f"{message}  (space=toggle, enter=confirm; pick >= {min_items})",
            choices=options,
        ).ask()
        if picks is None:
            return None
        if len(picks) >= min_items:
            return picks
        console.print(f"[yellow]Pick at least {min_items} items.[/]")


def pick_inspect(catalog: Catalog) -> Selection | None:
    """Tree-driven drill-down to a single Selection."""
    highlight: dict[str, str] = {}

    console.clear()
    console.print(Panel.fit("[bold]Inspect[/] — drill down to one experiment", style="cyan"))
    render_tree(catalog, highlight)

    platform = _ask_select("Platform:", sorted(catalog))
    if not platform:
        return None
    highlight["platform"] = platform

    console.clear()
    render_tree(catalog, highlight)
    model = _ask_select("Model:", sorted(catalog[platform]))
    if not model:
        return None
    highlight["model"] = model

    console.clear()
    render_tree(catalog, highlight)
    available_modes = [m for m in MODES if m in catalog[platform][model]]
    mode = _ask_select("Mode:", available_modes)
    if not mode:
        return None
    highlight["mode"] = mode

    console.clear()
    render_tree(catalog, highlight)
    word = _ask_select("Word:", sorted(catalog[platform][model][mode]))
    if not word:
        return None
    highlight["word"] = word

    entry = catalog[platform][model][mode][word]
    console.clear()
    render_tree(catalog, highlight)
    return Selection(platform, model, mode, word, entry)


def pick_compare(catalog: Catalog) -> CompareSet | None:
    """Compare flow: pick axis, pin the rest, multi-select along the axis."""
    console.clear()
    console.print(Panel.fit("[bold]Compare[/] — choose a dimension to vary", style="cyan"))

    axis = _ask_select(
        "Compare across:",
        ["word", "mode (online vs offline)", "platform", "model"],
    )
    if not axis:
        return None
    axis = axis.split()[0]  # "mode (online vs offline)" -> "mode"

    cs = CompareSet(axis=axis)

    if axis == "word":
        # pin platform, model, mode; multi-select words
        render_tree(catalog)
        platform = _ask_select("Platform:", sorted(catalog))
        if not platform:
            return None
        model = _ask_select("Model:", sorted(catalog[platform]))
        if not model:
            return None
        modes_avail = [m for m in MODES if m in catalog[platform][model]]
        mode = _ask_select("Mode:", modes_avail)
        if not mode:
            return None
        words = sorted(catalog[platform][model][mode])
        picks = _ask_checkbox("Words to compare:", words)
        if not picks:
            return None
        for w in picks:
            cs.selections.append(
                Selection(platform, model, mode, w, catalog[platform][model][mode][w])
            )

    elif axis == "mode":
        render_tree(catalog)
        platform = _ask_select("Platform:", sorted(catalog))
        if not platform:
            return None
        model = _ask_select("Model:", sorted(catalog[platform]))
        if not model:
            return None
        modes_avail = [m for m in MODES if m in catalog[platform][model]]
        if len(modes_avail) < 2:
            console.print("[red]Both online and offline are required for mode comparison.[/]")
            return None
        # need a word that exists in both modes
        common_words = set(catalog[platform][model][modes_avail[0]])
        for m in modes_avail[1:]:
            common_words &= set(catalog[platform][model][m])
        if not common_words:
            console.print("[red]No word is available across both modes for this model.[/]")
            return None
        word = _ask_select("Word (must exist in both modes):", sorted(common_words))
        if not word:
            return None
        for m in modes_avail:
            cs.selections.append(
                Selection(platform, model, m, word, catalog[platform][model][m][word])
            )

    elif axis == "platform":
        render_tree(catalog)
        # need a model+mode+word that exists across the picked platforms
        platforms_all = sorted(catalog)
        if len(platforms_all) < 2:
            console.print("[red]Need at least 2 platforms.[/]")
            return None
        platforms = _ask_checkbox("Platforms to compare:", platforms_all)
        if not platforms:
            return None
        # intersect models
        common_models = set(catalog[platforms[0]])
        for p in platforms[1:]:
            common_models &= set(catalog[p])
        if not common_models:
            console.print("[red]No common model across the picked platforms.[/]")
            return None
        model = _ask_select("Model (common to all):", sorted(common_models))
        if not model:
            return None
        # intersect modes
        common_modes = set(MODES)
        for p in platforms:
            common_modes &= set(catalog[p][model])
        if not common_modes:
            console.print("[red]No common mode across the picked platforms for this model.[/]")
            return None
        mode = _ask_select("Mode:", sorted(common_modes))
        if not mode:
            return None
        # intersect words
        common_words = set(catalog[platforms[0]][model][mode])
        for p in platforms[1:]:
            common_words &= set(catalog[p][model][mode])
        if not common_words:
            console.print("[red]No common word across the picked platforms.[/]")
            return None
        word = _ask_select("Word:", sorted(common_words))
        if not word:
            return None
        for p in platforms:
            cs.selections.append(
                Selection(p, model, mode, word, catalog[p][model][mode][word])
            )

    elif axis == "model":
        render_tree(catalog)
        platform = _ask_select("Platform:", sorted(catalog))
        if not platform:
            return None
        models_all = sorted(catalog[platform])
        if len(models_all) < 2:
            console.print("[red]Need at least 2 models on this platform.[/]")
            return None
        models = _ask_checkbox("Models to compare:", models_all)
        if not models:
            return None
        common_modes = set(MODES)
        for m in models:
            common_modes &= set(catalog[platform][m])
        if not common_modes:
            console.print("[red]No common mode across the picked models.[/]")
            return None
        mode = _ask_select("Mode:", sorted(common_modes))
        if not mode:
            return None
        common_words = set(catalog[platform][models[0]][mode])
        for m in models[1:]:
            common_words &= set(catalog[platform][m][mode])
        if not common_words:
            console.print("[red]No common word across the picked models.[/]")
            return None
        word = _ask_select("Word:", sorted(common_words))
        if not word:
            return None
        for m in models:
            cs.selections.append(
                Selection(platform, m, mode, word, catalog[platform][m][mode][word])
            )

    if len(cs.selections) < 2:
        console.print("[red]Comparison needs at least 2 items.[/]")
        return None
    return cs


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_csv(path: Path) -> pd.DataFrame:
    """Read a power-consumption CSV with Timestamp(ms),Current(uA)."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Analyses
# --------------------------------------------------------------------------- #

def _label_for(sel: Selection, axis: str | None) -> str:
    if axis == "word":
        return sel.word
    if axis == "mode":
        return sel.mode
    if axis == "platform":
        return sel.platform
    if axis == "model":
        return sel.model
    return sel.label()


def analysis_plot_power(items: list[Selection], axis: str | None) -> None:
    """Plot the power-consumption signal (Current(uA) vs Timestamp(ms))."""
    plottable = [s for s in items if s.entry.has_csv]
    if not plottable:
        console.print("[red]No CSV power traces available for the selected items.[/]")
        return

    _, ax = plt.subplots(figsize=(11, 5))
    for sel in plottable:
        console.print(f"  [dim]reading[/] {sel.entry.csv_path}")
        df = load_csv(sel.entry.csv_path)
        ax.plot(df["Timestamp(ms)"], df["Current(uA)"],
                label=_label_for(sel, axis), linewidth=0.7)

    if len(plottable) == 1:
        title = plottable[0].label()
    else:
        title = f"Power signal — comparing by {axis}"
    ax.set_title(title)
    ax.set_xlabel("Timestamp (ms)")
    ax.set_ylabel("Current (uA)")
    ax.grid(True, alpha=0.3)
    if len(plottable) > 1:
        ax.legend()
    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------------- #
# Analysis helpers
# --------------------------------------------------------------------------- #

def _get(d: dict, *path, default=None):
    """Safe nested-dict access: _get(d, 'a', 'b') -> d['a']['b'] or default."""
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _scaled(v, factor: float):
    return v * factor if isinstance(v, (int, float)) else None


def _format_num(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, str):
        return v
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        if v != v:  # NaN
            return "—"
        if abs(v) >= 10000:
            return f"{v:,.0f}"
        if abs(v) >= 100:
            return f"{v:,.1f}"
        if abs(v) >= 1:
            return f"{v:.3f}"
        return f"{v:.4g}"
    return str(v)


def _load_json_safe(sel: Selection) -> dict:
    if not sel.entry.has_json:
        return {}
    try:
        return load_json(sel.entry.json_path)
    except Exception as e:
        console.print(f"[red]Failed to read {sel.entry.json_path}: {e}[/]")
        return {}


def _build_table(title: str, items: list[Selection], axis: str | None,
                 rows: list[tuple[str, list]]) -> Table:
    table = Table(title=title, show_header=True, header_style="bold cyan",
                  title_style="bold")
    table.add_column("Metric", style="cyan", no_wrap=True)
    for s in items:
        table.add_column(_label_for(s, axis), justify="right")
    for name, vals in rows:
        table.add_row(name, *[_format_num(v) for v in vals])
    return table


def _maybe_barplot(items: list[Selection], axis: str | None, title: str,
                   values: list, ylabel: str) -> None:
    """Bar chart for compare mode; no-op when only one item."""
    if len(items) < 2:
        return
    labels = [_label_for(s, axis) for s in items]
    pairs = [
        (l, float(v)) for l, v in zip(labels, values)
        if isinstance(v, (int, float)) and not (isinstance(v, float) and v != v)
    ]
    if len(pairs) < 2:
        return
    _, ax = plt.subplots(figsize=(max(6, 1.8 * len(pairs)), 4))
    labs, vals = zip(*pairs)
    bars = ax.bar(labs, vals)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                _format_num(v), ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------------- #
# Concrete analyses
# --------------------------------------------------------------------------- #

def analysis_power_stats(items: list[Selection], axis: str | None) -> None:
    """Aggregate stats on the Current(uA) trace from the CSV."""
    stats: list[dict | None] = []
    for sel in items:
        if not sel.entry.has_csv:
            stats.append(None)
            continue
        df = load_csv(sel.entry.csv_path)
        c = df["Current(uA)"].astype(float)
        t = df["Timestamp(ms)"].astype(float)
        median = float(c.median())
        std = float(c.std())
        thr = median + std
        stats.append({
            "samples": int(len(c)),
            "duration (ms)": float(t.iloc[-1] - t.iloc[0]),
            "mean (uA)": float(c.mean()),
            "median (uA)": median,
            "std (uA)": std,
            "min (uA)": float(c.min()),
            "p95 (uA)": float(c.quantile(0.95)),
            "max / peak (uA)": float(c.max()),
            "RMS (uA)": float(((c ** 2).mean()) ** 0.5),
            "active duty (% > med+std)": float((c > thr).mean() * 100.0),
        })

    if all(s is None for s in stats):
        console.print("[red]No CSV power traces available for the selected items.[/]")
        return

    metrics = [
        "samples", "duration (ms)", "mean (uA)", "median (uA)", "std (uA)",
        "min (uA)", "p95 (uA)", "max / peak (uA)", "RMS (uA)",
        "active duty (% > med+std)",
    ]
    rows = [(m, [s[m] if s else None for s in stats]) for m in metrics]
    console.print(_build_table("Power statistics (from CSV)", items, axis, rows))
    _maybe_barplot(items, axis, "Mean current",
                   [s["mean (uA)"] if s else None for s in stats], "uA")
    _maybe_barplot(items, axis, "Peak current",
                   [s["max / peak (uA)"] if s else None for s in stats], "uA")


def analysis_energy(items: list[Selection], axis: str | None) -> None:
    """Energy per inference from the JSON energy_estimate block."""
    data = [_load_json_safe(s) for s in items]
    rows = [
        ("voltage (V)",            [_get(d, "energy_estimate", "voltage_v") for d in data]),
        ("current (mA)",           [_get(d, "energy_estimate", "current_ma") for d in data]),
        ("power (W)",              [_get(d, "energy_estimate", "power_w") for d in data]),
        ("power (mW)",             [_scaled(_get(d, "energy_estimate", "power_w"), 1000) for d in data]),
        ("energy / inference (J)", [_get(d, "energy_estimate", "energy_per_inference_j") for d in data]),
        ("energy / inference (mJ)",[_scaled(_get(d, "energy_estimate", "energy_per_inference_j"), 1000) for d in data]),
        ("energy / inference (uJ)",[_get(d, "energy_estimate", "energy_per_inference_uj") for d in data]),
        ("MAC ops / Joule",        [_get(d, "energy_estimate", "mac_ops_per_joule") for d in data]),
    ]
    console.print(_build_table("Energy per inference (from JSON)", items, axis, rows))
    _maybe_barplot(items, axis, "Energy per inference",
                   [_get(d, "energy_estimate", "energy_per_inference_uj") for d in data], "uJ")


def analysis_latency(items: list[Selection], axis: str | None) -> None:
    """Inference latency stats from cnn_latency_us."""
    data = [_load_json_safe(s) for s in items]
    rows = [
        ("count",          [_get(d, "cnn_latency_us", "count") for d in data]),
        ("min (ms)",       [_scaled(_get(d, "cnn_latency_us", "min"), 1e-3) for d in data]),
        ("avg (ms)",       [_scaled(_get(d, "cnn_latency_us", "avg"), 1e-3) for d in data]),
        ("median (ms)",    [_scaled(_get(d, "cnn_latency_us", "median"), 1e-3) for d in data]),
        ("p95 (ms)",       [_scaled(_get(d, "cnn_latency_us", "p95"), 1e-3) for d in data]),
        ("max (ms)",       [_scaled(_get(d, "cnn_latency_us", "max"), 1e-3) for d in data]),
        ("std (ms)",       [_scaled(_get(d, "cnn_latency_us", "std"), 1e-3) for d in data]),
        ("avg (us)",       [_get(d, "cnn_latency_us", "avg") for d in data]),
        ("end-to-end lower bound (ms)",
         [_get(d, "estimated_end_to_end_lower_bound_ms") for d in data]),
    ]
    console.print(_build_table("Inference latency (from JSON)", items, axis, rows))
    _maybe_barplot(items, axis, "Average inference latency",
                   [_scaled(_get(d, "cnn_latency_us", "avg"), 1e-3) for d in data], "ms")


def analysis_cycles(items: list[Selection], axis: str | None) -> None:
    """CPU cycles per inference + clock & MAC density."""
    data = [_load_json_safe(s) for s in items]
    rows = [
        ("clock (MHz)",          [_get(d, "cycles_estimate", "clock_mhz") for d in data]),
        ("cycles count",         [_get(d, "cycles", "count") for d in data]),
        ("cycles avg",           [_get(d, "cycles", "avg") for d in data]),
        ("cycles median",        [_get(d, "cycles", "median") for d in data]),
        ("cycles min",           [_get(d, "cycles", "min") for d in data]),
        ("cycles max",           [_get(d, "cycles", "max") for d in data]),
        ("cycles p95",           [_get(d, "cycles", "p95") for d in data]),
        ("cycles std",           [_get(d, "cycles", "std") for d in data]),
        ("cycles per inference (avg)",
         [_get(d, "cycles_estimate", "cycles_per_inference_avg") for d in data]),
        ("MAC ops per cycle",    [_get(d, "cycles_estimate", "mac_ops_per_cycle") for d in data]),
    ]
    console.print(_build_table("CPU cycles (from JSON)", items, axis, rows))
    _maybe_barplot(items, axis, "Cycles per inference (avg)",
                   [_get(d, "cycles_estimate", "cycles_per_inference_avg") for d in data],
                   "cycles")


def analysis_throughput(items: list[Selection], axis: str | None) -> None:
    """Throughput in inferences/s and MAC ops/s."""
    data = [_load_json_safe(s) for s in items]
    rows = [
        ("inferences / s (from cnn time)",
         [_get(d, "inferences_per_second_from_cnn_time") for d in data]),
        ("end-to-end lower bound (ms)",
         [_get(d, "estimated_end_to_end_lower_bound_ms") for d in data]),
        ("MAC ops / inference",
         [_get(d, "compute_estimate", "mac_ops_per_inference") for d in data]),
        ("MAC ops / second",
         [_get(d, "compute_estimate", "mac_ops_per_second") for d in data]),
        ("MOPS (M MAC ops / s)",
         [_get(d, "compute_estimate", "mops_per_second") for d in data]),
    ]
    console.print(_build_table("Throughput (from JSON)", items, axis, rows))
    _maybe_barplot(items, axis, "Inferences per second",
                   [_get(d, "inferences_per_second_from_cnn_time") for d in data],
                   "inf/s")
    _maybe_barplot(items, axis, "MOPS",
                   [_get(d, "compute_estimate", "mops_per_second") for d in data],
                   "M MAC ops / s")


def analysis_efficiency(items: list[Selection], axis: str | None) -> None:
    """Compute efficiency: MAC density per cycle and per Joule."""
    data = [_load_json_safe(s) for s in items]
    rows = [
        ("MAC ops / inference",
         [_get(d, "compute_estimate", "mac_ops_per_inference") for d in data]),
        ("MAC ops / cycle",
         [_get(d, "cycles_estimate", "mac_ops_per_cycle") for d in data]),
        ("MAC ops / Joule",
         [_get(d, "energy_estimate", "mac_ops_per_joule") for d in data]),
        ("MAC ops / s / W",
         [_get(d, "energy_estimate", "mac_ops_per_second_per_watt") for d in data]),
        ("MOPS (M MAC ops / s)",
         [_get(d, "compute_estimate", "mops_per_second") for d in data]),
    ]
    console.print(_build_table("Compute efficiency (from JSON)", items, axis, rows))
    _maybe_barplot(items, axis, "MAC ops per Joule",
                   [_get(d, "energy_estimate", "mac_ops_per_joule") for d in data],
                   "MAC / J")


def analysis_memory(items: list[Selection], axis: str | None) -> None:
    """Memory footprint: text / data / bss / static SRAM, plus elf size."""
    data = [_load_json_safe(s) for s in items]

    def kib(v):
        return _scaled(v, 1 / 1024.0)

    rows = [
        ("elf (bytes)",         [_get(d, "elf_size_bytes") for d in data]),
        ("elf (KiB)",           [kib(_get(d, "elf_size_bytes")) for d in data]),
        ("text (bytes)",        [_get(d, "memory", "text_bytes") for d in data]),
        ("data (bytes)",        [_get(d, "memory", "data_bytes") for d in data]),
        ("bss (bytes)",         [_get(d, "memory", "bss_bytes") for d in data]),
        ("dec / total (bytes)", [_get(d, "memory", "dec_bytes") for d in data]),
        ("text (KiB)",          [kib(_get(d, "memory", "text_bytes")) for d in data]),
        ("static SRAM (bytes)", [_get(d, "static_sram_usage", "static_sram_bytes") for d in data]),
        ("static SRAM (KiB)",   [_get(d, "static_sram_usage", "static_sram_kib") for d in data]),
    ]
    console.print(_build_table("Memory footprint (from JSON)", items, axis, rows))
    _maybe_barplot(items, axis, "Static SRAM",
                   [_get(d, "static_sram_usage", "static_sram_kib") for d in data], "KiB")
    _maybe_barplot(items, axis, "Flash text size",
                   [kib(_get(d, "memory", "text_bytes")) for d in data], "KiB")


def analysis_uart(items: list[Selection], axis: str | None) -> None:
    """UART transmission overhead: event counts and time-per-event."""
    data = [_load_json_safe(s) for s in items]
    rows = [
        ("UART events",        [_get(d, "counts", "num_uart_events") for d in data]),
        ("inference events",   [_get(d, "counts", "num_inference_events") for d in data]),
        ("done seen",          [_get(d, "counts", "done_seen") for d in data]),
        ("uart time count",    [_get(d, "uart_transmission_time_estimate_s", "count") for d in data]),
        ("uart time avg (ms)", [_scaled(_get(d, "uart_transmission_time_estimate_s", "avg"), 1000) for d in data]),
        ("uart time median (ms)",
         [_scaled(_get(d, "uart_transmission_time_estimate_s", "median"), 1000) for d in data]),
        ("uart time p95 (ms)", [_scaled(_get(d, "uart_transmission_time_estimate_s", "p95"), 1000) for d in data]),
        ("uart time max (ms)", [_scaled(_get(d, "uart_transmission_time_estimate_s", "max"), 1000) for d in data]),
        ("uart time min (ms)", [_scaled(_get(d, "uart_transmission_time_estimate_s", "min"), 1000) for d in data]),
        ("total uart time (s, count*avg)",
         [(_get(d, "uart_transmission_time_estimate_s", "count") or 0)
          * (_get(d, "uart_transmission_time_estimate_s", "avg") or 0) for d in data]),
    ]
    console.print(_build_table("UART overhead (from JSON)", items, axis, rows))
    _maybe_barplot(items, axis, "UART events",
                   [_get(d, "counts", "num_uart_events") for d in data], "events")


def analysis_audio(items: list[Selection], axis: str | None) -> None:
    """Sensor / audio acquisition window."""
    data = [_load_json_safe(s) for s in items]
    rows = [
        ("sample rate (Hz)",    [_get(d, "sensor_acquisition", "sample_rate_hz") for d in data]),
        ("sample count",        [_get(d, "sensor_acquisition", "sample_count") for d in data]),
        ("audio window (ms)",   [_get(d, "sensor_acquisition", "audio_window_ms") for d in data]),
        ("audio window (s)",    [_get(d, "sensor_acquisition", "audio_window_s") for d in data]),
        ("mode (online/offline)", [_get(d, "mode") for d in data]),
    ]
    console.print(_build_table("Sensor / audio acquisition (from JSON)", items, axis, rows))


def analysis_summary(items: list[Selection], axis: str | None) -> None:
    """Run every implemented analysis in sequence (skipping the time-series plot)."""
    console.print(Panel.fit("[bold]Full summary[/]", style="cyan"))
    for key in ("memory", "audio", "latency", "cycles", "throughput",
                "efficiency", "energy", "uart", "power_stats"):
        label, fn = ANALYSES[key]
        console.rule(f"[bold]{label}[/]")
        fn(items, axis)


# Map of analysis-menu options. Callable signature: (items, axis).
ANALYSES: dict[str, tuple[str, callable]] = {
    "plot_power":  ("Plot power signal (Current vs time)", analysis_plot_power),
    "power_stats": ("Power statistics (mean / peak / RMS current, duty cycle)", analysis_power_stats),
    "energy":      ("Energy per inference (J, uJ, mJ)", analysis_energy),
    "latency":     ("Inference latency (cnn_latency_us: avg / median / p95 / std)", analysis_latency),
    "cycles":      ("CPU cycles per inference (cycles, MHz, cycles/MAC)", analysis_cycles),
    "throughput":  ("Throughput (inferences/s, MAC ops/s, MOPS)", analysis_throughput),
    "efficiency":  ("Compute efficiency (MACs/cycle, MAC ops per Joule)", analysis_efficiency),
    "memory":      ("Memory footprint (text / data / bss / static SRAM)", analysis_memory),
    "uart":        ("UART transmission overhead (event count, time estimate)", analysis_uart),
    "audio":       ("Sensor / audio acquisition (sample rate, window length)", analysis_audio),
    "summary":     ("Full summary report (all of the above, side-by-side)", analysis_summary),
}


def run_analysis_menu(items: list[Selection], axis: str | None) -> None:
    while True:
        choices = [
            Choice(title=label, value=key) for key, (label, _) in ANALYSES.items()
        ]
        choices.append(Choice(title="<- back", value="__back__"))
        pick = questionary.select("Analysis:", choices=choices).ask()
        if pick in (None, "__back__"):
            return
        _, fn = ANALYSES[pick]
        fn(items, axis)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main_menu(catalog: Catalog) -> None:
    while True:
        console.clear()
        console.print(Panel.fit(
            "[bold]Experiments — Data analysis[/]\n"
            f"[dim]{ROOT}[/]",
            style="cyan",
        ))
        render_tree(catalog)
        choice = questionary.select(
            "What would you like to do?",
            choices=[
                Choice(title="Inspect — drill down to one experiment", value="inspect"),
                Choice(title="Compare — vary one dimension across 2+ experiments", value="compare"),
                Choice(title="Quit", value="quit"),
            ],
        ).ask()

        if choice in (None, "quit"):
            return
        if choice == "inspect":
            sel = pick_inspect(catalog)
            if sel is None:
                continue
            run_analysis_menu([sel], axis=None)
        elif choice == "compare":
            cs = pick_compare(catalog)
            if cs is None:
                continue
            run_analysis_menu(cs.selections, axis=cs.axis)


def main() -> int:
    catalog = discover()
    if not catalog:
        console.print(f"[red]No experiments found under {ROOT}.[/]")
        return 1
    try:
        main_menu(catalog)
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
