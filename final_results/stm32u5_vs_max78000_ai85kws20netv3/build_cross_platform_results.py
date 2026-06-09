#!/usr/bin/env python3
"""
Cross-platform comparison: STM32U5 vs MAX78000 — ai85kws20netv3 model.
Same SVG + HTML dashboard structure as the per-platform build scripts.
"""

import csv
import html
import json
import math
import re
from pathlib import Path

ROOT   = Path("/home/pascal/Documents/ml_on_mcu/MLonMCU_Project")
OUT    = ROOT / "final_results" / "stm32u5_vs_max78000_ai85kws20netv3"
TABLES = OUT / "tables"
PLOTS  = OUT / "plots"

VARIANTS = [
    {
        "id":       "u5_v0",
        "label":    "U5 v0 float32",
        "short":    "U5-v0",
        "platform": "STM32U5",
        "color":    "#1d4ed8",
        "path":     ROOT / "Experiments/General Profiling/U5/ai85kws20netv3_model_v0/online/on.json",
        "pwr_dir":  ROOT / "Experiments/PWR Consumption/U5/ai85kws20netv3_model_v0/online/on_peak_analysis",
        "xcube_project": ROOT / "ai85kws20netv3_model/platforms/stm32u5/u5_keyword_spotting",
    },
    {
        "id":       "u5_v1",
        "label":    "U5 v1 INT8",
        "short":    "U5-v1",
        "platform": "STM32U5",
        "color":    "#0891b2",
        "path":     ROOT / "Experiments/General Profiling/U5/ai85kws20netv3_model_v1/online/on.json",
        "pwr_dir":  ROOT / "Experiments/PWR Consumption/U5/ai85kws20netv3_model_v1/online/on_peak_analysis",
        "xcube_project": ROOT / "ai85kws20netv3_model/platforms/stm32u5/u5_keyword_spotting_quantized",
    },
    {
        "id":       "u5_v2",
        "label":    "U5 v2 pruned",
        "short":    "U5-v2",
        "platform": "STM32U5",
        "color":    "#0e7490",
        "path":     ROOT / "Experiments/General Profiling/U5/ai85kws20netv3_model_v2/online/on.json",
        "pwr_dir":  ROOT / "Experiments/PWR Consumption/U5/ai85kws20netv3_model_v2/online/on_peak_analysis",
        "xcube_project": ROOT / "ai85kws20netv3_model/platforms/stm32u5/u5_keyword_spotting_pruned_quantized",
    },
    {
        "id":       "mx_v0",
        "label":    "MAX78000 v0",
        "short":    "MX-v0",
        "platform": "MAX78000",
        "color":    "#b45309",
        "path":     ROOT / "Experiments/General Profiling/MAX78000/ai85kws20netv3_model_v0/online/on.json",
        "pwr_dir":  ROOT / "Experiments/PWR Consumption/MAX78000/ai85kws20netv3_model_v0/online/on_peak_analysis",
        "xcube_project": None,
    },
    {
        "id":       "mx_v1",
        "label":    "MAX78000 v1",
        "short":    "MX-v1",
        "platform": "MAX78000",
        "color":    "#d97706",
        "path":     ROOT / "Experiments/General Profiling/MAX78000/ai85kws20netv3_model_v1/online/on.json",
        "pwr_dir":  ROOT / "Experiments/PWR Consumption/MAX78000/ai85kws20netv3_model_v1/online/on_peak_analysis",
        "xcube_project": None,
    },
    {
        "id":       "mx_v0_0",
        "label":    "MAX78000 cpu",
        "short":    "MX-cpu",
        "platform": "MAX78000",
        "color":    "#c2410c",
        "path":     ROOT / "Experiments/General Profiling/MAX78000/ai85kws20netv3_model_v0_0/online/on.json",
        "pwr_dir":  ROOT / "Experiments/PWR Consumption/MAX78000/ai85kws20netv3_model_v0_0/five_offline_peak_analysis",
        "xcube_project": None,
    },
]

KEY_METRICS = [
    ("cnn_latency_ms_avg",                          "CNN latency avg",         "ms",      "lower"),
    ("cycles.avg",                                  "Cycles avg",              "cycles",  "lower"),
    ("inferences_per_second_from_cnn_time",         "Throughput",              "inf/s",   "higher"),
    ("estimated_end_to_end_lower_bound_ms",         "End-to-end lower bound",  "ms",      "lower"),
    ("memory.text_bytes",                           "Text",                    "B",       "lower"),
    ("memory.data_bytes",                           "Data",                    "B",       "lower"),
    ("memory.bss_bytes",                            "BSS",                     "B",       "lower"),
    ("static_sram_usage.static_sram_bytes",         "Static SRAM",             "B",       "lower"),
    ("elf_size_bytes",                              "ELF size",                "B",       "lower"),
    ("model_info.weights_bytes",                    "AI weights",              "B",       "lower"),
    ("model_info.activations_bytes",                "AI activations",          "B",       "lower"),
    ("compute_estimate.mac_ops_per_inference",      "MAC ops",                 "MAC",     "lower"),
    ("compute_estimate.mops_per_second",            "MOPS/s",                  "MOPS/s",  "higher"),
    ("energy_estimate.current_ma",                  "Current",                 "mA",      "lower"),
    ("energy_estimate.power_w",                     "Power",                   "W",       "lower"),
    ("energy_estimate.energy_per_inference_uj",     "Energy/inference",        "uJ",      "lower"),
    ("energy_estimate.mac_ops_per_joule",           "MAC/J",                   "MAC/J",   "higher"),
]

RADAR_METRICS = [
    ("cnn_latency_ms_avg",                       "Latency",    "lower"),
    ("estimated_end_to_end_lower_bound_ms",       "End-to-end", "lower"),
    ("static_sram_usage.static_sram_bytes",       "SRAM",       "lower"),
    ("memory.text_bytes",                         "Text",       "lower"),
    ("elf_size_bytes",                            "ELF",        "lower"),
    ("energy_estimate.energy_per_inference_uj",   "Energy",     "lower"),
    ("inferences_per_second_from_cnn_time",        "Throughput", "higher"),
    ("energy_estimate.mac_ops_per_joule",          "MAC/J",      "higher"),
]

CROSS_PLATFORM_COMPARISONS = [
    ("mx_v0_vs_u5_v0",  "mx_v0",   "u5_v0"),
    ("mx_v1_vs_u5_v1",  "mx_v1",   "u5_v1"),
    ("mx_v1_vs_u5_v2",  "mx_v1",   "u5_v2"),
    ("mx_cpu_vs_u5_v0", "mx_v0_0", "u5_v0"),
    ("mx_cpu_vs_u5_v1", "mx_v0_0", "u5_v1"),
]

def ensure_dirs():
    TABLES.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)

def parse_num(value):
    if isinstance(value, bool) or value is None: return None
    if isinstance(value, (int, float)): return float(value)
    if isinstance(value, str):
        try: return float(value)
        except ValueError: return None
    return None

def flatten(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            child = f"{prefix}.{key}" if prefix else key
            out.update(flatten(value, child))
    else:
        num = parse_num(obj)
        if num is not None:
            out[prefix] = num
    return out

def parse_size_to_bytes(value, unit):
    number = float(value.replace(",", ""))
    return number * 1024.0 if unit.lower().startswith("k") else number

def parse_xcube_ai_report(report_path):
    if not report_path or not report_path.exists(): return {}
    text = report_path.read_text(errors="replace")
    info = {}
    m = re.search(r"weights \(ro\)\s*:\s*([0-9,]+)\s*B", text)
    if m: info["model_info.weights_bytes"] = float(m.group(1).replace(",", ""))
    m = re.search(r"activations \(rw\)\s*:\s*([0-9,]+)\s*B", text)
    if m: info["model_info.activations_bytes"] = float(m.group(1).replace(",", ""))
    m = re.search(r"macc\s*:\s*([0-9,]+)", text)
    if m: info["model_info.macc_from_report"] = float(m.group(1).replace(",", ""))
    return info

def fill_report_fallback(data, flat_rows, xcube_project):
    if not xcube_project: return
    report_path = xcube_project / "X-CUBE-AI" / "App" / "network_generate_report.txt"
    for key, value in parse_xcube_ai_report(report_path).items():
        flat_rows.setdefault(key, value)

def fmt_value(value, unit=""):
    if value is None or math.isnan(value): return ""
    abs_v = abs(value)
    if unit in ("B", "cycles", "MAC"): return f"{value:,.0f}"
    if abs_v >= 1_000_000: return f"{value:,.0f}"
    if abs_v >= 100: return f"{value:,.2f}"
    if abs_v >= 10:  return f"{value:,.3f}"
    if abs_v >= 1:   return f"{value:,.4f}"
    return f"{value:.6g}"

def pct(value):
    if value is None or math.isnan(value): return ""
    return f"{value:+.2f}%"

def metric_direction(key):
    higher_tokens = ["per_second", "mops", "mac_ops_per_cycle", "mac_ops_per_joule", "inferences_per_second", "clock_mhz"]
    lower_tokens  = ["bytes", "latency", "cycles", "time", "current", "power", "energy", "sram", "bss", "data", "text", "uart", "end_to_end"]
    low_key = key.lower()
    if any(t in low_key for t in higher_tokens): return "higher"
    if any(t in low_key for t in lower_tokens):  return "lower"
    return "higher"

def score(values, direction):
    present = [v for v in values if v is not None and not math.isnan(v)]
    if not present: return [None] * len(values)
    lo, hi = min(present), max(present)
    if hi == lo: return [1.0 if v is not None else None for v in values]
    result = []
    for value in values:
        if value is None or math.isnan(value): result.append(None); continue
        raw = (value - lo) / (hi - lo)
        result.append(1.0 - raw if direction == "lower" else raw)
    return result

def best_ratio_score(values, direction):
    present = [v for v in values if v is not None and not math.isnan(v) and v > 0]
    if not present: return [None] * len(values)
    best = min(present) if direction == "lower" else max(present)
    result = []
    for value in values:
        if value is None or math.isnan(value) or value <= 0: result.append(None)
        elif direction == "lower": result.append(best / value)
        else: result.append(value / best)
    return result

def svg_header(width, height):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial,Helvetica,sans-serif;fill:#111827}",
        ".title{font-size:24px;font-weight:700}",
        ".subtitle{font-size:13px;fill:#475569}",
        ".axis{stroke:#94a3b8;stroke-width:1}",
        ".grid{stroke:#cbd5e1;stroke-width:1;stroke-dasharray:3 4}",
        ".label{font-size:12px;fill:#334155}",
        ".small{font-size:10px;fill:#475569}",
        ".tick{font-size:10px;fill:#64748b}",
        ".platform_u5{font-size:11px;fill:#1d4ed8;font-style:italic}",
        ".platform_mx{font-size:11px;fill:#b45309;font-style:italic}",
        "</style>",
        '<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>',
    ]

def save_svg(path, width, height, body):
    content = svg_header(width, height) + body + ["</svg>"]
    path.write_text("\n".join(content))

def line(x1, y1, x2, y2, cls="axis", color=None, width=None, dash=None):
    attrs = [f'x1="{x1:.2f}"', f'y1="{y1:.2f}"', f'x2="{x2:.2f}"', f'y2="{y2:.2f}"']
    attrs.append(f'stroke="{color}"' if color else f'class="{cls}"')
    if width: attrs.append(f'stroke-width="{width}"')
    if dash:  attrs.append(f'stroke-dasharray="{dash}"')
    return "<line " + " ".join(attrs) + "/>"

def rect(x, y, w, h, fill, stroke="none", opacity=1.0, rx=0):
    return f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" rx="{rx}" fill="{fill}" stroke="{stroke}" opacity="{opacity}"/>'

def text(x, y, value, cls="label", anchor="start", rotate=None):
    transform = f' transform="rotate({rotate} {x:.2f} {y:.2f})"' if rotate else ""
    return f'<text x="{x:.2f}" y="{y:.2f}" class="{cls}" text-anchor="{anchor}"{transform}>{html.escape(str(value))}</text>'

def polyline(points, color, width=2.5, fill="none", opacity=1.0):
    coords = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f'<polyline points="{coords}" fill="{fill}" stroke="{color}" stroke-width="{width}" opacity="{opacity}"/>'

def polygon(points, color, opacity=0.16, stroke_width=2.5):
    coords = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f'<polygon points="{coords}" fill="{color}" fill-opacity="{opacity}" stroke="{color}" stroke-width="{stroke_width}"/>'

def title_block(title, subtitle=None):
    body = [text(32, 36, title, "title")]
    if subtitle: body.append(text(32, 58, subtitle, "subtitle"))
    return body

def spread_y_positions(desired, min_gap, lo, hi):
    if not desired: return []
    indexed = sorted(enumerate(desired), key=lambda item: item[1])
    ys = [min(max(y, lo), hi) for _, y in indexed]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + min_gap)
    overflow = ys[-1] - hi
    if overflow > 0: ys = [y - overflow for y in ys]
    for i in range(len(ys) - 2, -1, -1):
        ys[i] = min(ys[i], ys[i + 1] - min_gap)
    underflow = lo - ys[0]
    if underflow > 0: ys = [y + underflow for y in ys]
    result = [0.0] * len(desired)
    for (idx, _), y in zip(indexed, ys):
        result[idx] = y
    return result


# ── Bar grid (supports log scale for cross-platform ranges) ──────────────────

def bar_grid(metrics, rows, path, title, subtitle, cols=3, log_scale=False):
    panel_w = 360
    panel_h = 235
    margin_x = 38
    margin_y = 84
    width = margin_x * 2 + cols * panel_w
    nrows = math.ceil(len(metrics) / cols)
    height = margin_y + nrows * panel_h + 34
    body = title_block(title, subtitle)

    # Platform separator annotation
    # We have 3 STM32U5 variants (indices 0-2) then 2 MAX78000 (indices 3-4)
    gap = 18
    bar_w_approx = 30  # rough estimate

    for idx, (key, label, unit, direction) in enumerate(metrics):
        col = idx % cols
        row = idx // cols
        x0 = margin_x + col * panel_w
        y0 = margin_y + row * panel_h
        chart_x = x0 + 54
        chart_y = y0 + 32
        chart_w = panel_w - 94
        chart_h = panel_h - 82
        values = [rows[v["id"]].get(key) for v in VARIANTS]
        present = [v for v in values if v is not None and v > 0]

        body.append(text(x0, y0 + 10, label, "label"))
        body.append(text(x0, y0 + 27, unit, "small"))
        if not present:
            body.append(text(chart_x, chart_y + chart_h / 2, "n/a", "subtitle", "middle"))
            continue

        use_log = log_scale and (max(present) / min(present) > 10)

        if use_log:
            ymin_log = math.floor(math.log10(min(present)))
            ymax_log = math.ceil(math.log10(max(present)))
            if ymax_log == ymin_log: ymax_log += 1
        else:
            ymax = max(present) * 1.15
            if ymax <= 0: ymax = 1.0

        body.append(line(chart_x, chart_y, chart_x, chart_y + chart_h))
        body.append(line(chart_x, chart_y + chart_h, chart_x + chart_w, chart_y + chart_h))

        if use_log:
            for power in range(ymin_log, ymax_log + 1):
                ty = chart_y + chart_h - chart_h * (power - ymin_log) / (ymax_log - ymin_log)
                body.append(line(chart_x, ty, chart_x + chart_w, ty, "grid"))
                body.append(text(chart_x - 4, ty + 4, f"1e{power}", "tick", "end"))
        else:
            for tick in range(1, 4):
                ty = chart_y + chart_h - chart_h * tick / 4
                body.append(line(chart_x, ty, chart_x + chart_w, ty, "grid"))

        bar_w = (chart_w - gap * (len(VARIANTS) + 1)) / len(VARIANTS)
        sep_x = None

        for i, variant in enumerate(VARIANTS):
            value = values[i]
            bx = chart_x + gap + i * (bar_w + gap)

            # Platform separator after U5 (index 2 → between 2 and 3)
            if i == 3 and sep_x is None:
                sx = chart_x + gap + 3 * (bar_w + gap) - gap / 2
                body.append(line(sx, chart_y, sx, chart_y + chart_h + 4, color="#94a3b8", width=1.2, dash="4 3"))

            if value is None or value <= 0:
                body.append(rect(bx, chart_y + chart_h - 4, bar_w, 4, "#cbd5e1"))
                body.append(text(bx + bar_w / 2, chart_y + chart_h + 17, variant["short"], "tick", "middle"))
                continue

            if use_log:
                log_v = math.log10(value)
                bh = chart_h * (log_v - ymin_log) / (ymax_log - ymin_log)
            else:
                bh = chart_h * value / ymax

            bh = max(bh, 2.0)
            by = chart_y + chart_h - bh
            body.append(rect(bx, by, bar_w, bh, variant["color"], rx=3))
            body.append(text(bx + bar_w / 2, by - 5, fmt_value(value, unit), "tick", "middle"))
            body.append(text(bx + bar_w / 2, chart_y + chart_h + 17, variant["short"], "tick", "middle"))

    # Platform labels in last panel
    lx = margin_x + 8
    ly = height - 18
    body.append(rect(lx, ly - 12, 12, 12, "#1d4ed8", rx=2))
    body.append(text(lx + 18, ly, "STM32U5 (U5-v0/v1/v2)", "small"))
    body.append(rect(lx + 220, ly - 12, 12, 12, "#b45309", rx=2))
    body.append(text(lx + 238, ly, "MAX78000 (MX-v0/v1/cpu)", "small"))
    save_svg(path, width, height, body)


# ── Radar ────────────────────────────────────────────────────────────────────

def radar_plot(rows):
    width, height = 980, 780
    cx, cy, r = 490, 400, 255
    body = title_block(
        "STM32U5 vs MAX78000 — Best-Ratio Radar",
        "Outer ring = best value across all 5 variants; proportional ratio to best.",
    )
    metrics = RADAR_METRICS
    angles = [-math.pi / 2 + 2 * math.pi * i / len(metrics) for i in range(len(metrics))]
    for ring in range(1, 6):
        rr = r * ring / 5
        points = [(cx + rr * math.cos(a), cy + rr * math.sin(a)) for a in angles]
        body.append(polygon(points, "#e2e8f0", opacity=0.0, stroke_width=1))
    for idx, (_, label, _) in enumerate(metrics):
        a = angles[idx]
        x, y = cx + r * math.cos(a), cy + r * math.sin(a)
        lx, ly = cx + (r + 48) * math.cos(a), cy + (r + 48) * math.sin(a)
        body.append(line(cx, cy, x, y, color="#cbd5e1", width=1))
        anchor = "middle"
        if math.cos(a) > 0.25: anchor = "start"
        elif math.cos(a) < -0.25: anchor = "end"
        body.append(text(lx, ly, label, "label", anchor))
    series = []
    for variant in VARIANTS:
        scores = []
        for i, (key, _, direction) in enumerate(metrics):
            all_values = [rows[v["id"]].get(key) for v in VARIANTS]
            scores.append(best_ratio_score(all_values, direction)[VARIANTS.index(variant)])
        points = []
        for s, a in zip(scores, angles):
            value = 0.0 if s is None else max(0.0, min(1.0, s))
            points.append((cx + r * value * math.cos(a), cy + r * value * math.sin(a)))
        series.append((variant, points))
    for variant, points in series:
        body.append(polygon(points, variant["color"], opacity=0.08, stroke_width=0))
    for variant, points in series:
        body.append(polyline(points + [points[0]], variant["color"], width=3.0, opacity=0.95))
    for variant, points in series:
        for x, y in points:
            body.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.8" fill="{variant["color"]}" stroke="#ffffff" stroke-width="1.2"/>')
    lx, ly = 36, height - 130
    for i, variant in enumerate(VARIANTS):
        y = ly + i * 24
        body.append(rect(lx, y - 12, 14, 14, variant["color"], rx=2))
        body.append(text(lx + 22, y, variant["label"], "label"))
    save_svg(PLOTS / "01_normalized_score_radar.svg", width, height, body)


# ── Heatmap ───────────────────────────────────────────────────────────────────

def heatmap(rows, all_keys):
    keys = [k for k in all_keys if sum(rows[v["id"]].get(k) is not None for v in VARIANTS) >= 2]
    keys = sorted(keys)
    cell_w = 108
    row_h = 21
    left = 360
    top = 92
    width = left + cell_w * len(VARIANTS) + 58
    height = top + row_h * len(keys) + 54
    body = title_block(
        "All Numeric Metrics Heatmap — Cross-Platform",
        "Green means better after direction normalization. Separator marks STM32U5 | MAX78000 boundary.",
    )
    for i, variant in enumerate(VARIANTS):
        body.append(text(left + i * cell_w + cell_w / 2, top - 18, variant["short"], "label", "middle"))
    # Platform separator
    sep_x = left + 3 * cell_w
    body.append(line(sep_x, top - 28, sep_x, top + row_h * len(keys) + 4, color="#1e40af", width=1.5, dash="4 3"))

    for r_idx, key in enumerate(keys):
        y = top + r_idx * row_h
        body.append(text(24, y + 14, key, "small"))
        values = [rows[v["id"]].get(key) for v in VARIANTS]
        sc = score(values, metric_direction(key))
        for c_idx, value in enumerate(values):
            x = left + c_idx * cell_w
            if value is None:
                fill, label = "#e5e7eb", ""
            else:
                s = 0.0 if sc[c_idx] is None else sc[c_idx]
                red   = int(220 - 110 * s)
                green = int(82  + 120 * s)
                blue  = int(82  +  60 * s)
                fill  = f"rgb({red},{green},{blue})"
                label = fmt_value(value)
            body.append(rect(x, y, cell_w - 2, row_h - 2, fill))
            body.append(text(x + cell_w / 2, y + 14, label, "small", "middle"))
    save_svg(PLOTS / "07_all_numeric_metrics_heatmap.svg", width, height, body)


# ── Log metric overview ───────────────────────────────────────────────────────

def log_metric_overview(rows):
    metrics = [
        ("memory.text_bytes",                       "Text bytes",    "#2563eb"),
        ("memory.bss_bytes",                        "BSS bytes",     "#0891b2"),
        ("static_sram_usage.static_sram_bytes",     "Static SRAM",   "#16a34a"),
        ("elf_size_bytes",                          "ELF bytes",     "#64748b"),
        ("cnn_latency_us.avg",                      "CNN latency µs","#dc2626"),
        ("cycles.avg",                              "Cycles",        "#9333ea"),
        ("compute_estimate.mac_ops_per_inference",  "MAC ops",       "#ea580c"),
        ("energy_estimate.energy_per_inference_uj", "Energy µJ",     "#be123c"),
        ("energy_estimate.mac_ops_per_joule",       "MAC/J",         "#0f766e"),
    ]
    width, height = 1160, 700
    left, right, top, bottom = 95, 870, 92, 590
    all_values = [rows[v["id"]].get(key) for key, _, _ in metrics for v in VARIANTS
                  if rows[v["id"]].get(key) is not None and rows[v["id"]].get(key) > 0]
    if not all_values: return
    ymin = 10 ** math.floor(math.log10(min(all_values)))
    ymax = 10 ** math.ceil(math.log10(max(all_values)))
    body = title_block(
        "Online Metric Overview — Log Scale (Cross-Platform)",
        "Raw values with mixed units; x-axis: U5-v0, U5-v1, U5-v2, MX-v0, MX-v1, MX-cpu.",
    )
    for power in range(int(math.log10(ymin)), int(math.log10(ymax)) + 1):
        value = 10 ** power
        y = bottom - (math.log10(value) - math.log10(ymin)) / (math.log10(ymax) - math.log10(ymin)) * (bottom - top)
        body.append(line(left, y, right, y, "grid"))
        body.append(text(left - 12, y + 4, f"1e{power}", "tick", "end"))
    body.append(line(left, top, left, bottom))
    body.append(line(left, bottom, right, bottom))
    xs = [left + i * (right - left) / (len(VARIANTS) - 1) for i in range(len(VARIANTS))]
    for x, variant in zip(xs, VARIANTS):
        body.append(text(x, bottom + 24, variant["short"], "label", "middle"))
    # Platform separator after U5 (between index 2 and 3)
    sep_x = xs[2] + (xs[3] - xs[2]) / 2
    body.append(line(sep_x, top, sep_x, bottom + 8, color="#94a3b8", width=1.2, dash="5 4"))
    body.append(text(sep_x - 38, top - 8, "STM32U5", "platform_u5"))
    body.append(text(sep_x + 6,  top - 8, "MAX78000", "platform_mx"))
    end_labels = []
    for key, label, color in metrics:
        points = []
        for x, variant in zip(xs, VARIANTS):
            value = rows[variant["id"]].get(key)
            if value is None or value <= 0: continue
            y = bottom - (math.log10(value) - math.log10(ymin)) / (math.log10(ymax) - math.log10(ymin)) * (bottom - top)
            points.append((x, y))
        if len(points) >= 2:
            body.append(polyline(points, color, width=2.5, opacity=0.9))
        for x, y in points:
            body.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"/>')
        if points: end_labels.append((label, color, points[-1][1]))
    label_ys = spread_y_positions([item[2] + 4 for item in end_labels], 18, top + 14, bottom - 8)
    for (label, color, endpoint_y), label_y in zip(end_labels, label_ys):
        body.append(line(right, endpoint_y, right + 14, label_y - 4, color=color, width=1.2))
        body.append(text(right + 18, label_y, label, "label"))
    save_svg(PLOTS / "09_log_metric_overview.svg", width, height, body)


# ── Bubble frontier ───────────────────────────────────────────────────────────

def axis_ticks(lo, hi, count=5):
    if hi <= lo: return [lo] * count
    return [lo + i * (hi - lo) / (count - 1) for i in range(count)]

def scale(value, lo, hi, a, b):
    if hi == lo: return (a + b) / 2
    return a + (value - lo) / (hi - lo) * (b - a)

def bubble_frontier(rows, spec):
    width, height = 900, 580
    left, right, top, bottom = 92, 780, 82, 460
    x_key, y_key, size_key = spec["x_key"], spec["y_key"], spec["size_key"]
    xs_raw    = [rows[v["id"]].get(x_key) for v in VARIANTS]
    ys_raw    = [rows[v["id"]].get(y_key) for v in VARIANTS]
    sizes_raw = [rows[v["id"]].get(size_key) for v in VARIANTS]
    valid = [(x, y, s) for x, y, s in zip(xs_raw, ys_raw, sizes_raw) if x and y and s]
    if not valid: return
    xs_v, ys_v, sz_v = zip(*valid)
    xmin, xmax = min(xs_v), max(xs_v)
    ymin, ymax = min(ys_v), max(ys_v)
    smin, smax = min(sz_v), max(sz_v)
    xpad = (xmax - xmin) * 0.12 if xmax > xmin else xmax * 0.08
    ypad = (ymax - ymin) * 0.14 if ymax > ymin else ymax * 0.08
    xmin, xmax = xmin - xpad, xmax + xpad
    ymin, ymax = ymin - ypad, ymax + ypad
    body = title_block(spec["title"], spec["subtitle"])
    body.append(line(left, top, left, bottom))
    body.append(line(left, bottom, right, bottom))
    for tick, value in enumerate(axis_ticks(xmin, xmax)):
        x = left + tick * (right - left) / 4
        body.append(line(x, bottom, x, bottom + 5))
        body.append(text(x, bottom + 22, spec["x_fmt"](value), "tick", "middle"))
        body.append(line(x, top, x, bottom, "grid"))
    for tick, value_y in enumerate(axis_ticks(ymin, ymax)):
        y = bottom - tick * (bottom - top) / 4
        body.append(line(left - 5, y, left, y))
        body.append(text(left - 10, y + 4, spec["y_fmt"](value_y), "tick", "end"))
        body.append(line(left, y, right, y, "grid"))
    body.append(text((left + right) / 2, height - 44, spec["x_label"], "label", "middle"))
    body.append(text(24, (top + bottom) / 2, spec["y_label"], "label", "middle", rotate=-90))
    body.append(text(right - 4, top - 20, f"Bubble size: {spec['size_label']}", "small", "end"))
    points = []
    for variant, xraw, yraw, sraw in zip(VARIANTS, xs_raw, ys_raw, sizes_raw):
        if not (xraw and yraw and sraw): continue
        x = left + (xraw - xmin) / (xmax - xmin) * (right - left)
        y = bottom - (yraw - ymin) / (ymax - ymin) * (bottom - top)
        radius = 10 + 21 * ((sraw - smin) / (smax - smin) if smax > smin else 0.5)
        body.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" fill="{variant["color"]}" fill-opacity="0.80" stroke="#111827" stroke-width="1"/>')
        points.append((variant, x, y, radius, sraw))
    label_ys = spread_y_positions([y + 4 for _, _, y, _, _ in points], 38, top + 20, bottom - 26)
    for (variant, x, y, radius, sraw), label_y in zip(points, label_ys):
        label_x = x + radius + 10
        anchor = "start"
        if label_x > width - 148: label_x = x - radius - 10; anchor = "end"
        sign = 1 if anchor == "start" else -1
        body.append(line(x + sign * radius * 0.72, y, label_x - sign * 5, label_y - 5, color=variant["color"], width=1.1))
        body.append(text(label_x, label_y, variant["label"], "label", anchor))
        body.append(text(label_x, label_y + 15, spec["size_fmt"](sraw), "tick", anchor))
    save_svg(PLOTS / spec["file"], width, height, body)

def frontier(rows):
    specs = [
        {
            "file": "10_latency_sram_energy_frontier.svg",
            "title": "Latency vs SRAM Frontier — Cross-Platform",
            "subtitle": "Lower-left is better; bubble size = energy per inference.",
            "x_key": "static_sram_usage.static_sram_bytes",
            "y_key": "cnn_latency_ms_avg",
            "size_key": "energy_estimate.energy_per_inference_uj",
            "x_label": "Static SRAM",   "y_label": "CNN latency",   "size_label": "Energy/inf",
            "x_fmt": lambda v: f"{v/1024:.0f} KiB",
            "y_fmt": lambda v: f"{v:.1f} ms",
            "size_fmt": lambda v: f"{v:.0f} µJ",
        },
        {
            "file": "11_latency_energy_frontier.svg",
            "title": "Latency vs Energy Frontier — Cross-Platform",
            "subtitle": "Lower-left is better; bubble size = static SRAM.",
            "x_key": "energy_estimate.energy_per_inference_uj",
            "y_key": "cnn_latency_ms_avg",
            "size_key": "static_sram_usage.static_sram_bytes",
            "x_label": "Energy per inference", "y_label": "CNN latency", "size_label": "Static SRAM",
            "x_fmt": lambda v: f"{v:.0f} µJ",
            "y_fmt": lambda v: f"{v:.1f} ms",
            "size_fmt": lambda v: f"{v/1024:.0f} KiB",
        },
        {
            "file": "12_text_sram_frontier.svg",
            "title": "Flash Text vs SRAM Frontier — Cross-Platform",
            "subtitle": "Lower-left is better; bubble size = CNN latency.",
            "x_key": "memory.text_bytes",
            "y_key": "static_sram_usage.static_sram_bytes",
            "size_key": "cnn_latency_ms_avg",
            "x_label": "Text memory", "y_label": "Static SRAM", "size_label": "CNN latency",
            "x_fmt": lambda v: f"{v/1024:.0f} KiB",
            "y_fmt": lambda v: f"{v/1024:.0f} KiB",
            "size_fmt": lambda v: f"{v:.1f} ms",
        },
        {
            "file": "13_throughput_energy_frontier.svg",
            "title": "Throughput vs Energy Frontier — Cross-Platform",
            "subtitle": "Upper-left is better; bubble size = static SRAM.",
            "x_key": "energy_estimate.energy_per_inference_uj",
            "y_key": "inferences_per_second_from_cnn_time",
            "size_key": "static_sram_usage.static_sram_bytes",
            "x_label": "Energy per inference", "y_label": "Throughput", "size_label": "Static SRAM",
            "x_fmt": lambda v: f"{v:.0f} µJ",
            "y_fmt": lambda v: f"{v:.1f}/s",
            "size_fmt": lambda v: f"{v/1024:.0f} KiB",
        },
    ]
    for spec in specs:
        bubble_frontier(rows, spec)


# ── Cross-platform improvement plots ─────────────────────────────────────────

def improvement_fill(value, max_abs):
    if value is None or math.isnan(value): return "#e5e7eb"
    if max_abs <= 0: return "#f8fafc"
    strength = min(1.0, abs(value) / max_abs)
    if value >= 0:
        start, end = (236, 253, 245), (15, 118, 110)
    else:
        start, end = (254, 242, 242), (185, 28, 28)
    rgb = [round(start[i] + (end[i] - start[i]) * strength) for i in range(3)]
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"

def compute_cross_delta_rows(rows):
    delta_rows = []
    for key, label, unit, direction in KEY_METRICS:
        row = {"metric": label, "unit": unit, "direction": direction}
        for comp_name, new_id, base_id in CROSS_PLATFORM_COMPARISONS:
            value      = rows[new_id].get(key)
            base_value = rows[base_id].get(key)
            if value is None or base_value in (None, 0):
                row[comp_name + "_raw_pct"] = None
                row[comp_name + "_improvement_pct"] = None
            else:
                raw_pct = (value - base_value) / base_value * 100.0
                improvement = -raw_pct if direction == "lower" else raw_pct
                row[comp_name + "_raw_pct"] = raw_pct
                row[comp_name + "_improvement_pct"] = improvement
        delta_rows.append(row)
    return delta_rows

def improvement_bar_plot(rows_delta, comparison_key, file_name, title, subtitle):
    valid = [row for row in rows_delta if row.get(comparison_key) is not None]
    if not valid: return
    max_abs = max(abs(row[comparison_key]) for row in valid) or 1.0
    use_log = max_abs > 200
    log_max = math.log10(max_abs + 1) if use_log else 1.0
    half_span = (1030 - 380) * 0.48
    width, left, right, top, row_h = 1120, 380, 1030, 92, 30
    zero_x = left + (right - left) * 0.5
    height = top + len(valid) * row_h + 76
    body = title_block(title, subtitle)
    body.append(line(zero_x, top - 12, zero_x, top + len(valid) * row_h, color="#334155", width=1.3))
    if use_log:
        for tv in [-100000, -10000, -1000, -100, -10, 10, 100, 1000, 10000, 100000]:
            if abs(tv) > max_abs * 1.05: continue
            log_pos = math.copysign(math.log10(abs(tv) + 1), tv) / log_max
            x = zero_x + log_pos * half_span
            body.append(line(x, top - 8, x, top + len(valid) * row_h, "grid"))
            label = f"{tv:+,}%" if abs(tv) < 1000 else f"{tv // 1000:+d}k%"
            body.append(text(x, top - 18, label, "tick", "middle"))
    else:
        for tick in [-max_abs, -max_abs / 2, 0, max_abs / 2, max_abs]:
            x = zero_x + tick / max_abs * half_span
            body.append(line(x, top - 8, x, top + len(valid) * row_h, "grid"))
            body.append(text(x, top - 18, f"{tick:+.0f}%", "tick", "middle"))
    for row_idx, row in enumerate(valid):
        y = top + row_idx * row_h
        value = row[comparison_key]
        body.append(text(32, y + 18, row["metric"], "label"))
        body.append(text(270, y + 18, row["direction"], "small", "end"))
        if use_log:
            bar_w = math.log10(abs(value) + 1) / log_max * half_span
            color_strength = math.log10(abs(value) + 1) / log_max
        else:
            bar_w = abs(value) / max_abs * half_span
            color_strength = abs(value) / max_abs
        fill = improvement_fill(math.copysign(color_strength, value), 1.0)
        x = zero_x if value >= 0 else zero_x - bar_w
        body.append(rect(x, y + 5, bar_w, 18, fill, stroke="#ffffff", rx=3))
        label_x = x + bar_w + 8 if value >= 0 else x - 8
        anchor = "start" if value >= 0 else "end"
        label_str = f"+{value:,.0f}%" if value >= 1000 else pct(value)
        body.append(text(label_x, y + 19, label_str, "label", anchor))
    footer = "right of zero = MAX78000 better; left = STM32U5 better"
    if use_log: footer += " (log₁₀ scale)"
    body.append(text(left, height - 30, footer, "subtitle"))
    save_svg(PLOTS / file_name, width, height, body)

def improvement_heatmap(rows_delta):
    comparisons = [(c + "_improvement_pct", n + " vs " + b) for c, n, b in CROSS_PLATFORM_COMPARISONS]
    values = [abs(row[key]) for row in rows_delta for key, _ in comparisons if row.get(key) is not None]
    max_abs = max(values) if values else 1.0
    left, top, row_h, cell_w = 285, 90, 30, 180
    width = max(1060, left + len(comparisons) * cell_w + 40)
    height = top + len(rows_delta) * row_h + 86
    body = title_block(
        "Cross-Platform Improvement Heatmap",
        "Green = MAX78000 better; red = STM32U5 better; values are direction-aware percentages.",
    )
    for idx, (_, label) in enumerate(comparisons):
        body.append(text(left + idx * cell_w + cell_w / 2, top - 16, label, "label", "middle"))
    for row_idx, row in enumerate(rows_delta):
        y = top + row_idx * row_h
        body.append(text(32, y + 19, row["metric"], "label"))
        body.append(text(240, y + 19, row["direction"], "small", "end"))
        for col_idx, (key, _) in enumerate(comparisons):
            x = left + col_idx * cell_w
            value = row.get(key)
            body.append(rect(x, y, cell_w - 5, row_h - 4, improvement_fill(value, max_abs), rx=3))
            body.append(text(x + cell_w / 2, y + 19, pct(value) if value is not None else "n/a", "label", "middle"))
    body.append(text(left, height - 34, "positive = MAX78000 variant is better for that metric", "subtitle"))
    save_svg(PLOTS / "16_cross_platform_improvement_heatmap.svg", width, height, body)

def improvement_plots(rows_delta):
    improvement_heatmap(rows_delta)
    improvement_bar_plot(
        rows_delta, "mx_v0_vs_u5_v0_improvement_pct",
        "17_mx_v0_vs_u5_v0_improvement_bars.svg",
        "MAX78000 v0 vs STM32U5 v0 — Cross-Platform",
        "Both float32 baselines; positive = MAX78000 better for that metric.",
    )
    improvement_bar_plot(
        rows_delta, "mx_v1_vs_u5_v1_improvement_pct",
        "18_mx_v1_vs_u5_v1_improvement_bars.svg",
        "MAX78000 v1 vs STM32U5 v1 — Cross-Platform",
        "INT8 variants on both platforms.",
    )
    improvement_bar_plot(
        rows_delta, "mx_v1_vs_u5_v2_improvement_pct",
        "19_mx_v1_vs_u5_v2_improvement_bars.svg",
        "MAX78000 v1 vs STM32U5 v2 — Best vs Best",
        "Best MAX78000 (v1 INT8 HW) vs best STM32U5 (v2 pruned INT8).",
    )
    improvement_bar_plot(
        rows_delta, "mx_cpu_vs_u5_v0_improvement_pct",
        "20_mx_cpu_vs_u5_v0_improvement_bars.svg",
        "MAX78000 int8 cpu vs STM32U5 v0 — CPU Baseline Comparison",
        "Both platforms without CNN HW acceleration; shows raw CPU performance difference.",
    )
    improvement_bar_plot(
        rows_delta, "mx_cpu_vs_u5_v1_improvement_pct",
        "21_mx_cpu_vs_u5_v1_improvement_bars.svg",
        "MAX78000 int8 cpu vs STM32U5 v1 — INT8 CPU Comparison",
        "Both INT8 software-only inference; no hardware acceleration on either platform.",
    )


# ── Power analysis ────────────────────────────────────────────────────────────

def load_power_reports():
    reports = {}
    for variant in VARIANTS:
        path = variant["pwr_dir"] / "peak_report.json"
        if path.exists():
            reports[variant["id"]] = json.loads(path.read_text())
    return reports

def representative_peak_number(report):
    target = report.get("avg_peak_duration_ms")
    peaks  = report.get("per_peak", [])
    if not peaks: return None
    if target is None: return peaks[len(peaks) // 2]["peak_number"]
    return min(peaks, key=lambda p: abs(p["duration_ms"] - target))["peak_number"]

def decimate_points(points, max_points):
    if len(points) <= max_points: return points
    step = math.ceil(len(points) / max_points)
    out = []
    for idx in range(0, len(points), step):
        chunk = points[idx:idx + step]
        if chunk:
            out.append((sum(p[0] for p in chunk)/len(chunk), sum(p[1] for p in chunk)/len(chunk)))
    return out

def read_power_trace_windows(report, pad_ms=20.0, max_points=900):
    csv_path = Path(report["csv"])
    if not csv_path.exists(): return {}
    peaks = report.get("per_peak", [])
    intervals = [{
        "peak_number": int(p["peak_number"]),
        "read_start": p["start_ms"] - pad_ms,
        "read_end":   p["end_ms"]   + pad_ms,
        "start_ms":   p["start_ms"],
        "end_ms":     p["end_ms"],
        "duration_ms":      p["duration_ms"],
        "baseline_ma":      p["baseline_current_ua"] / 1000.0,
        "energy_total_uj":  p.get("energy_total_uj"),
        "energy_uj":        p.get("energy_uj"),
        "points": [],
    } for p in peaks]
    intervals.sort(key=lambda item: item["read_start"])
    current_idx = 0
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            time_ms = float(row["Timestamp(ms)"])
            while current_idx < len(intervals) and time_ms > intervals[current_idx]["read_end"]:
                current_idx += 1
            if current_idx >= len(intervals): break
            interval = intervals[current_idx]
            if time_ms < interval["read_start"]: continue
            interval["points"].append((time_ms - interval["start_ms"], float(row["Current(uA)"]) / 1000.0))
    return {item["peak_number"]: {**{k: v for k, v in item.items() if k != "points"},
                                   "points": decimate_points(item["points"], max_points)}
            for item in intervals}

def power_tables(reports):
    summary_rows = []
    for variant in VARIANTS:
        report = reports.get(variant["id"])
        if not report: continue
        selected = representative_peak_number(report)
        summary_rows.append({
            "variant": variant["id"], "label": variant["label"], "platform": variant["platform"],
            "detected_peaks": report.get("detected_peaks"),
            "selected_peak": selected,
            "avg_peak_duration_ms": report.get("avg_peak_duration_ms"),
            "avg_peak_mean_current_ma": report.get("avg_peak_mean_current_ua", 0.0) / 1000.0,
            "avg_peak_mean_excess_current_ma": report.get("avg_peak_mean_excess_current_ua", 0.0) / 1000.0,
            "avg_peak_energy_total_uj": report.get("avg_peak_energy_total_uj"),
            "avg_peak_energy_uj": report.get("avg_peak_energy_uj"),
            "duration_vs_summary_ratio": report.get("duration_vs_summary_ratio"),
        })
    if not summary_rows: return
    with (TABLES / "power_peak_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader(); writer.writerows(summary_rows)
    md = ["| Variant | Platform | Peaks | Avg duration | Avg current | Avg energy (total) | Avg energy (excess) |",
          "|---|---|---:|---:|---:|---:|---:|"]
    for row in summary_rows:
        md.append(f"| {row['label']} | {row['platform']} | {row['detected_peaks']} | "
                  f"{row['avg_peak_duration_ms']:.2f} ms | {row['avg_peak_mean_current_ma']:.3f} mA | "
                  f"{row['avg_peak_energy_total_uj']/1000:.3f} mJ | {row['avg_peak_energy_uj']:.2f} µJ |")
    (TABLES / "power_peak_summary.md").write_text("\n".join(md) + "\n")

def power_window_overlay(reports, windows_by_variant, selection_by_variant, file_name, title, subtitle, variants_filter=None):
    width, height = 1120, 680
    left, right, top, bottom = 92, 920, 92, 540
    active_variants = variants_filter if variants_filter is not None else VARIANTS
    selected = [(v, selection_by_variant.get(v["id"]),
                 windows_by_variant.get(v["id"], {}).get(selection_by_variant.get(v["id"])))
                for v in active_variants
                if windows_by_variant.get(v["id"], {}).get(selection_by_variant.get(v["id"])) and
                   windows_by_variant.get(v["id"], {}).get(selection_by_variant.get(v["id"]))["points"]]
    if not selected: return
    max_dur = max(w["duration_ms"] for _, _, w in selected)
    idle_preroll_ms = max(2.0, min(25.0, max_dur * 0.03))
    x_values = [p[0] for _, _, w in selected for p in w["points"] if p[0] >= -idle_preroll_ms]
    y_values = [p[1] for _, _, w in selected for p in w["points"]]
    if not x_values or not y_values: return
    xmin, xmax = -idle_preroll_ms, max(x_values)
    ymin, ymax = min(y_values), max(y_values)
    ypad = (ymax - ymin) * 0.12
    ymin, ymax = ymin - ypad, ymax + ypad
    body = title_block(title, subtitle)
    body.append(line(left, top, left, bottom)); body.append(line(left, bottom, right, bottom))
    for idx, value in enumerate(axis_ticks(xmin, xmax, 6)):
        x = left + idx * (right - left) / 5
        body.append(line(x, top, x, bottom, "grid"))
        body.append(text(x, bottom + 22, f"{value:.1f} ms", "tick", "middle"))
    zero_x = scale(0.0, xmin, xmax, left, right)
    body.append(line(zero_x, top, zero_x, bottom, color="#111827", width=1.2, dash="5 4"))
    body.append(text(zero_x, top - 9, "0 ms", "tick", "middle"))
    for idx, value in enumerate(axis_ticks(ymin, ymax, 6)):
        y = bottom - idx * (bottom - top) / 5
        body.append(line(left, y, right, y, "grid"))
        body.append(text(left - 10, y + 4, f"{value:.1f} mA", "tick", "end"))
    body.append(text((left + right) / 2, height - 50, "Time relative to detected inference start (ms)", "label", "middle"))
    body.append(text(24, (top + bottom) / 2, "Current (mA)", "label", "middle", rotate=-90))
    end_labels = []
    for variant, peak_number, window in selected:
        raw_points = window["points"]
        points = [(scale(t, xmin, xmax, left, right), scale(c, ymin, ymax, bottom, top))
                  for t, c in raw_points if t >= xmin]
        if not points: continue
        body.append(polyline(points, variant["color"], width=2.2, opacity=0.9))
        start_x = scale(0.0, xmin, xmax, left, right)
        end_x   = scale(window["duration_ms"], xmin, xmax, left, right)
        body.append(line(start_x, top, start_x, bottom, color=variant["color"], width=1, dash="4 4"))
        body.append(line(end_x,   top, end_x,   bottom, color=variant["color"], width=1, dash="4 4"))
        anchor_t = max(0.0, min(window["duration_ms"] * 0.72, window["duration_ms"] - 8.0))
        closest = min((p for p in raw_points if p[0] >= 0.0), key=lambda p: abs(p[0] - anchor_t), default=raw_points[0])
        ax = scale(closest[0], xmin, xmax, left, right)
        ay = scale(closest[1], ymin, ymax, bottom, top)
        end_labels.append((variant, peak_number, ax, ay, window))
    label_ys = spread_y_positions([item[3] + 4 for item in end_labels], 36, top + 20, bottom - 32)
    for (variant, peak_number, anchor_x, anchor_y, window), label_y in zip(end_labels, label_ys):
        label_x = right + 22
        body.append(line(anchor_x, anchor_y, label_x - 6, label_y - 4, color=variant["color"], width=1.15))
        body.append(rect(label_x - 5, label_y - 14, 190, 34, "#ffffff", stroke="#e2e8f0", opacity=0.88, rx=4))
        body.append(text(label_x, label_y, f"{variant['label']} p{peak_number}", "label"))
        etxt = f"{window['duration_ms']:.2f} ms, {window['energy_total_uj']/1000:.3f} mJ" if window.get("energy_total_uj") else f"{window['duration_ms']:.2f} ms"
        body.append(text(label_x, label_y + 15, etxt, "small"))
    save_svg(PLOTS / file_name, width, height, body)

def power_average_bars(reports):
    rows = {v["id"]: {
        "avg_peak_duration_ms":           reports[v["id"]]["avg_peak_duration_ms"],
        "avg_peak_mean_current_ma":        reports[v["id"]]["avg_peak_mean_current_ua"] / 1000.0,
        "avg_peak_mean_excess_current_ma": reports[v["id"]]["avg_peak_mean_excess_current_ua"] / 1000.0,
        "avg_peak_energy_total_uj":        reports[v["id"]]["avg_peak_energy_total_uj"],
        "avg_peak_energy_uj":              reports[v["id"]]["avg_peak_energy_uj"],
    } for v in VARIANTS if v["id"] in reports}
    bar_grid([
        ("avg_peak_duration_ms",           "Peak duration",        "ms",  "lower"),
        ("avg_peak_mean_current_ma",        "Mean current",         "mA",  "lower"),
        ("avg_peak_mean_excess_current_ma", "Mean excess current",  "mA",  "lower"),
        ("avg_peak_energy_total_uj",        "Total peak energy",    "uJ",  "lower"),
        ("avg_peak_energy_uj",              "Excess peak energy",   "uJ",  "lower"),
    ], rows, PLOTS / "26_power_peak_average_bars.svg",
    "Power Peak Averages — Cross-Platform",
    "Averaged inference window metrics from power peak analysis.",
    cols=3, log_scale=True)

def power_analysis():
    reports = load_power_reports()
    if not reports: return
    (OUT / "power_peak_sources.json").write_text(
        json.dumps({v["id"]: str(v["pwr_dir"]) for v in VARIANTS}, indent=2) + "\n")
    power_tables(reports)
    windows = {v["id"]: read_power_trace_windows(reports[v["id"]], pad_ms=80.0, max_points=1400)
               for v in VARIANTS if v["id"] in reports}
    representative = {v["id"]: representative_peak_number(reports[v["id"]])
                      for v in VARIANTS if v["id"] in reports}
    power_window_overlay(reports, windows, representative,
        "23_power_representative_inference_window_overlay.svg",
        "Power Trace Overlay: Representative Inference Window — Cross-Platform",
        "One detected window per variant, selected by duration closest to that variant's average.")
    power_window_overlay(reports, windows, {v["id"]: 3 for v in VARIANTS},
        "24_power_peak03_inference_window_overlay.svg",
        "Power Trace Overlay: Peak 3 — Cross-Platform",
        "Same peak index from all variants, aligned to detected inference start.")
    long_group  = [v for v in VARIANTS if v["id"] in {"u5_v0", "u5_v1", "u5_v2", "mx_v0_0"}]
    short_group = [v for v in VARIANTS if v["id"] in {"mx_v0", "mx_v1"}]
    power_window_overlay(reports, windows, representative,
        "30_power_long_inference_overlay.svg",
        "Power Trace: Long Inference Group — STM32U5 & MAX78000 CPU",
        "STM32U5 (Float32, INT8, Pruned+INT8) and MAX78000 CPU-only; representative peaks.",
        variants_filter=long_group)
    power_window_overlay(reports, windows, representative,
        "31_power_short_inference_overlay.svg",
        "Power Trace: Short Inference Group — MAX78000 CNN Accelerator",
        "MAX78000 hardware CNN accelerator variants only (INT8, Pruned+INT8); representative peaks.",
        variants_filter=short_group)
    power_window_overlay(reports, windows, {v["id"]: 3 for v in VARIANTS},
        "32_power_long_peak03_overlay.svg",
        "Power Trace (Peak 3): Long Inference Group — STM32U5 & MAX78000 CPU",
        "STM32U5 and MAX78000 CPU-only run; peak 3 per variant.",
        variants_filter=long_group)
    power_window_overlay(reports, windows, {v["id"]: 3 for v in VARIANTS},
        "33_power_short_peak03_overlay.svg",
        "Power Trace (Peak 3): Short Inference Group — MAX78000 CNN Accelerator",
        "MAX78000 hardware CNN accelerator variants only; peak 3 per variant.",
        variants_filter=short_group)
    power_average_bars(reports)


# ── Tables ────────────────────────────────────────────────────────────────────

def write_tables(rows, all_keys):
    with (TABLES / "online_all_numeric_metrics.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric"] + [v["id"] for v in VARIANTS])
        for key in all_keys:
            writer.writerow([key] + [rows[v["id"]].get(key, "") for v in VARIANTS])

    with (TABLES / "online_key_metrics.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "unit", "direction"] + [v["id"] for v in VARIANTS])
        for key, label, unit, direction in KEY_METRICS:
            writer.writerow([label, unit, direction] + [rows[v["id"]].get(key, "") for v in VARIANTS])

    headers = [v["label"] for v in VARIANTS]
    md = ["| Metric | Unit | Direction | " + " | ".join(headers) + " |",
          "|---|---:|---|" + "---:|" * len(VARIANTS)]
    for key, label, unit, direction in KEY_METRICS:
        vals = [fmt_value(rows[v["id"]].get(key), unit) for v in VARIANTS]
        md.append(f"| {label} | {unit} | {direction} | " + " | ".join(vals) + " |")
    (TABLES / "online_key_metrics.md").write_text("\n".join(md) + "\n")

    rows_delta = compute_cross_delta_rows(rows)
    comparisons = [(c, n, b) for c, n, b in CROSS_PLATFORM_COMPARISONS]
    with (TABLES / "online_metric_deltas.csv").open("w", newline="") as f:
        fieldnames = ["metric", "unit", "direction"]
        for comp, _, _ in comparisons:
            fieldnames += [comp + "_raw_pct", comp + "_improvement_pct"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows_delta)

    comp_headers = [f"{n} vs {b}" for _, n, b in comparisons]
    md = ["| Metric | Direction | " + " | ".join(comp_headers) + " |",
          "|---|---|" + "---:|" * len(comparisons)]
    for row in rows_delta:
        vals = [pct(row.get(c + "_improvement_pct")) for c, _, _ in comparisons]
        md.append(f"| {row['metric']} | {row['direction']} | " + " | ".join(vals) + " |")
    (TABLES / "online_metric_deltas.md").write_text("\n".join(md) + "\n")


# ── Dashboard ─────────────────────────────────────────────────────────────────

def write_dashboard():
    plot_files = sorted(PLOTS.glob("*.svg"))
    key_table   = (TABLES / "online_key_metrics.md").read_text()
    delta_table = (TABLES / "online_metric_deltas.md").read_text()
    pwr_path    = TABLES / "power_peak_summary.md"
    pwr_table   = pwr_path.read_text() if pwr_path.exists() else None

    def md_to_html(md):
        lines = [l for l in md.splitlines() if l.startswith("|")]
        if len(lines) < 2: return ""
        header = [c.strip() for c in lines[0].strip("|").split("|")]
        rows = [[c.strip() for c in l.strip("|").split("|")] for l in lines[2:]]
        out = ["<table><thead><tr>"] + [f"<th>{html.escape(c)}</th>" for c in header] + ["</tr></thead><tbody>"]
        for row in rows:
            out.append("<tr>")
            out += [f"<td>{html.escape(c)}</td>" for c in row]
            out.append("</tr>")
        out += ["</tbody></table>"]
        return "\n".join(out)

    body = [
        "<!doctype html><html lang=\"en\"><head>",
        "<meta charset=\"utf-8\">",
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">",
        "<title>STM32U5 vs MAX78000 — ai85kws20netv3</title>",
        "<style>",
        "body{font-family:Arial,Helvetica,sans-serif;margin:0;background:#f8fafc;color:#111827}",
        "main{max-width:1280px;margin:0 auto;padding:32px}",
        "h1{font-size:30px;margin:0 0 4px}",
        ".sub{color:#475569;margin:0 0 24px;font-size:14px}",
        "section{margin:28px 0}",
        ".plot{background:#fff;border:1px solid #e2e8f0;border-radius:8px;margin:18px 0;padding:14px}",
        ".plot img{width:100%;height:auto;display:block}",
        "table{width:100%;border-collapse:collapse;background:#fff;border:1px solid #e2e8f0;font-size:12px}",
        "th,td{padding:7px 9px;border-bottom:1px solid #e2e8f0;text-align:right}",
        "th:first-child,td:first-child{text-align:left}",
        "th{background:#e2e8f0}",
        ".legend{display:flex;gap:24px;margin:8px 0 18px;flex-wrap:wrap}",
        ".chip{display:inline-flex;align-items:center;gap:6px;font-size:13px}",
        ".dot{width:14px;height:14px;border-radius:3px;display:inline-block}",
        "</style></head><body><main>",
        "<h1>STM32U5 vs MAX78000 — ai85kws20netv3</h1>",
        "<p class=\"sub\">Cross-platform online inference comparison. STM32U5: v0 float32, v1 INT8, v2 pruned INT8. MAX78000: v0 CNN accelerator, v1 pruned CNN accelerator, cpu INT8 software-only.</p>",
        "<div class=\"legend\">",
        *[f'<span class="chip"><span class="dot" style="background:{v["color"]}"></span>{html.escape(v["label"])}</span>'
          for v in VARIANTS],
        "</div>",
        "<section><h2>Key Metrics</h2>", md_to_html(key_table), "</section>",
        "<section><h2>Cross-Platform Relative Improvements</h2>",
        "<p style=\"font-size:13px;color:#475569\">Positive = MAX78000 variant is better for that metric.</p>",
        md_to_html(delta_table), "</section>",
    ]
    if pwr_table:
        body += ["<section><h2>Power Peak Summary</h2>", md_to_html(pwr_table), "</section>"]
    body += ["<section><h2>Plots</h2>"]
    for plot in plot_files:
        rel = plot.relative_to(OUT)
        body.append(f'<div class="plot"><h3>{html.escape(plot.stem)}</h3><img src="{html.escape(str(rel))}" alt="{html.escape(plot.stem)}"></div>')
    body += ["</section></main></body></html>"]
    (OUT / "index.html").write_text("\n".join(body))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ensure_dirs()
    rows = {}
    all_keys_set = set()
    for variant in VARIANTS:
        data = json.loads(variant["path"].read_text())
        flat = flatten(data)
        fill_report_fallback(data, flat, variant.get("xcube_project"))
        flat["source_path"] = str(variant["path"])
        rows[variant["id"]] = flat
        all_keys_set.update(k for k in flat if k != "source_path")
    all_keys = sorted(all_keys_set)

    (OUT / "online_sources.json").write_text(json.dumps({v["id"]: str(v["path"]) for v in VARIANTS}, indent=2) + "\n")
    (OUT / "online_flat_metrics.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")

    write_tables(rows, all_keys)

    radar_plot(rows)
    bar_grid([
        ("cnn_latency_ms_avg",                     "CNN latency avg",  "ms",    "lower"),
        ("inferences_per_second_from_cnn_time",     "Throughput",       "inf/s", "higher"),
        ("estimated_end_to_end_lower_bound_ms",     "End-to-end",       "ms",    "lower"),
        ("compute_estimate.mops_per_second",        "MOPS/s",           "MOPS/s","higher"),
        ("cycles.avg",                              "Cycles avg",       "cycles","lower"),
        ("energy_estimate.energy_per_inference_uj", "Energy/inference", "uJ",    "lower"),
    ], rows, PLOTS / "03_latency_throughput_energy.svg",
    "Latency, Throughput, Energy — Cross-Platform",
    "Log scale used where range exceeds 10× (marked bars). Separator: STM32U5 | MAX78000.",
    cols=3, log_scale=True)

    bar_grid([
        ("memory.text_bytes",                       "Text",       "B", "lower"),
        ("memory.bss_bytes",                        "BSS",        "B", "lower"),
        ("static_sram_usage.static_sram_bytes",     "Static SRAM","B", "lower"),
        ("elf_size_bytes",                          "ELF size",   "B", "lower"),
        ("model_info.weights_bytes",                "AI weights", "B", "lower"),
        ("model_info.activations_bytes",            "AI activations","B","lower"),
    ], rows, PLOTS / "02_memory_breakdown.svg",
    "Memory Footprint — Cross-Platform",
    "Log scale. STM32U5 has much larger firmware due to CMSIS/HAL.",
    cols=3, log_scale=True)

    bar_grid([
        ("energy_estimate.current_ma",              "Current",    "mA",    "lower"),
        ("energy_estimate.power_w",                 "Power",      "W",     "lower"),
        ("energy_estimate.energy_per_inference_uj", "Energy/inf", "uJ",    "lower"),
        ("energy_estimate.mac_ops_per_joule",       "MAC/J",      "MAC/J", "higher"),
        ("compute_estimate.mac_ops_per_inference",  "MAC ops",    "MAC",   "lower"),
        ("compute_estimate.mops_per_second",        "MOPS/s",     "MOPS/s","higher"),
    ], rows, PLOTS / "04_energy_efficiency.svg",
    "Energy and Compute Efficiency — Cross-Platform",
    "Log scale. MAX78000 CNN accelerator achieves orders-of-magnitude better efficiency.",
    cols=3, log_scale=True)

    heatmap(rows, all_keys)
    log_metric_overview(rows)
    frontier(rows)
    improvement_plots(compute_cross_delta_rows(rows))
    power_analysis()
    write_dashboard()
    print(f"wrote results to {OUT}")


if __name__ == "__main__":
    main()
