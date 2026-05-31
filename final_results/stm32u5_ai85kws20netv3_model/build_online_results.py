#!/usr/bin/env python3

import csv
import html
import json
import math
import re
from pathlib import Path


ROOT = Path("/home/pascal/Documents/ml_on_mcu/MLonMCU_Project")
OUT = ROOT / "final_results" / "stm32u5_ai85kws20netv3_model"
TABLES = OUT / "tables"
PLOTS = OUT / "plots"

VARIANTS = [
    {
        "id": "v0",
        "label": "v0 unquantized",
        "short": "v0",
        "color": "#475569",
        "path": ROOT / "Experiments/General Profiling/U5/ai85kws20netv3_model_v0/online/on.json",
    },
    {
        "id": "v1",
        "label": "v1 quantized",
        "short": "v1",
        "color": "#0f766e",
        "path": ROOT / "Experiments/General Profiling/U5/ai85kws20netv3_model_v1/online/on.json",
    },
    {
        "id": "v2",
        "label": "v2 pruned quantized",
        "short": "v2",
        "color": "#c2410c",
        "path": ROOT / "Experiments/General Profiling/U5/ai85kws20netv3_model_v2/online/on.json",
    },
]

KEY_METRICS = [
    ("cnn_latency_ms_avg", "CNN latency avg", "ms", "lower"),
    ("cycles.avg", "Cycles avg", "cycles", "lower"),
    ("inferences_per_second_from_cnn_time", "Throughput", "inf/s", "higher"),
    ("estimated_end_to_end_lower_bound_ms", "End-to-end lower bound", "ms", "lower"),
    ("memory.text_bytes", "Text", "B", "lower"),
    ("memory.data_bytes", "Data", "B", "lower"),
    ("memory.bss_bytes", "BSS", "B", "lower"),
    ("static_sram_usage.static_sram_bytes", "Static SRAM", "B", "lower"),
    ("elf_size_bytes", "ELF size", "B", "lower"),
    ("model_info.weights_bytes", "AI weights", "B", "lower"),
    ("model_info.activations_bytes", "AI activations", "B", "lower"),
    ("compute_estimate.mac_ops_per_inference", "MAC ops", "MAC", "lower"),
    ("compute_estimate.mops_per_second", "MOPS/s", "MOPS/s", "higher"),
    ("cycles_estimate.mac_ops_per_cycle", "MAC/cycle", "MAC/cycle", "higher"),
    ("energy_estimate.current_ma", "Current", "mA", "lower"),
    ("energy_estimate.power_w", "Power", "W", "lower"),
    ("energy_estimate.energy_per_inference_uj", "Energy/inference", "uJ", "lower"),
    ("energy_estimate.mac_ops_per_joule", "MAC/J", "MAC/J", "higher"),
]

RADAR_METRICS = [
    ("cnn_latency_ms_avg", "Latency", "lower"),
    ("cycles.avg", "Cycles", "lower"),
    ("estimated_end_to_end_lower_bound_ms", "End-to-end", "lower"),
    ("static_sram_usage.static_sram_bytes", "SRAM", "lower"),
    ("memory.text_bytes", "Text", "lower"),
    ("memory.bss_bytes", "BSS", "lower"),
    ("elf_size_bytes", "ELF", "lower"),
    ("energy_estimate.energy_per_inference_uj", "Energy", "lower"),
    ("inferences_per_second_from_cnn_time", "Throughput", "higher"),
    ("energy_estimate.mac_ops_per_joule", "MAC/J", "higher"),
]


def power_analysis_dir(variant_id):
    return ROOT / "Experiments" / "PWR Consumption" / "U5" / f"ai85kws20netv3_model_{variant_id}" / "online" / "on_peak_analysis"


def ensure_dirs():
    TABLES.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)


def parse_num(value):
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
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
    if unit.lower().startswith("k"):
        return number * 1024.0
    return number


def parse_xcube_ai_report(report_path):
    if not report_path.exists():
        return {}

    text = report_path.read_text(errors="replace")
    info = {}

    m = re.search(r"input\s+1/1\s*:\s*'[^']+',\s*([^,]+),\s*([0-9.,]+)\s*(KBytes|Bytes)", text)
    if m:
        info["model_info.input_bytes"] = parse_size_to_bytes(m.group(2), m.group(3))

    m = re.search(r"output\s+1/1\s*:\s*'[^']+',\s*([^,]+),\s*([0-9.,]+)\s*(KBytes|Bytes)", text)
    if m:
        info["model_info.output_bytes"] = parse_size_to_bytes(m.group(2), m.group(3))

    m = re.search(r"macc\s*:\s*([0-9,]+)", text)
    if m:
        info["model_info.macc_from_report"] = float(m.group(1).replace(",", ""))

    m = re.search(r"weights \(ro\)\s*:\s*([0-9,]+)\s*B", text)
    if m:
        info["model_info.weights_bytes"] = float(m.group(1).replace(",", ""))

    m = re.search(r"activations \(rw\)\s*:\s*([0-9,]+)\s*B", text)
    if m:
        info["model_info.activations_bytes"] = float(m.group(1).replace(",", ""))

    return info


def fill_report_fallback_metrics(data, flat_rows):
    project = data.get("project")
    if not project:
        return

    report_path = Path(project) / "X-CUBE-AI" / "App" / "network_generate_report.txt"
    report_info = parse_xcube_ai_report(report_path)
    for key, value in report_info.items():
        flat_rows.setdefault(key, value)


def fmt_value(value, unit=""):
    if value is None or math.isnan(value):
        return ""
    abs_v = abs(value)
    if unit == "B":
        return f"{value:,.0f}"
    if unit == "cycles" or unit == "MAC":
        return f"{value:,.0f}"
    if abs_v >= 1_000_000:
        return f"{value:,.0f}"
    if abs_v >= 100:
        return f"{value:,.2f}"
    if abs_v >= 10:
        return f"{value:,.3f}"
    if abs_v >= 1:
        return f"{value:,.4f}"
    return f"{value:.6g}"


def pct(value):
    if value is None or math.isnan(value):
        return ""
    return f"{value:+.2f}%"


def metric_direction(key):
    lower_tokens = [
        "bytes", "latency", "cycles", "time", "current", "power", "energy",
        "sram", "bss", "data", "text", "uart", "end_to_end", "count",
    ]
    higher_tokens = [
        "per_second", "mops", "mac_ops_per_cycle", "mac_ops_per_joule",
        "inferences_per_second", "clock_mhz",
    ]
    low_key = key.lower()
    if any(token in low_key for token in higher_tokens):
        return "higher"
    if any(token in low_key for token in lower_tokens):
        return "lower"
    return "higher"


def score(values, direction):
    present = [v for v in values if v is not None and not math.isnan(v)]
    if not present:
        return [None for _ in values]
    lo, hi = min(present), max(present)
    if hi == lo:
        return [1.0 if v is not None else None for v in values]
    result = []
    for value in values:
        if value is None or math.isnan(value):
            result.append(None)
            continue
        raw = (value - lo) / (hi - lo)
        result.append(1.0 - raw if direction == "lower" else raw)
    return result


def best_ratio_score(values, direction):
    present = [v for v in values if v is not None and not math.isnan(v) and v > 0]
    if not present:
        return [None for _ in values]
    best = min(present) if direction == "lower" else max(present)
    result = []
    for value in values:
        if value is None or math.isnan(value) or value <= 0:
            result.append(None)
        elif direction == "lower":
            result.append(best / value)
        else:
            result.append(value / best)
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
        "</style>",
        '<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>',
    ]


def save_svg(path, width, height, body):
    content = svg_header(width, height) + body + ["</svg>"]
    path.write_text("\n".join(content))


def line(x1, y1, x2, y2, cls="axis", color=None, width=None, dash=None):
    attrs = [f'x1="{x1:.2f}"', f'y1="{y1:.2f}"', f'x2="{x2:.2f}"', f'y2="{y2:.2f}"']
    if color:
        attrs.append(f'stroke="{color}"')
    else:
        attrs.append(f'class="{cls}"')
    if width:
        attrs.append(f'stroke-width="{width}"')
    if dash:
        attrs.append(f'stroke-dasharray="{dash}"')
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
    if subtitle:
        body.append(text(32, 58, subtitle, "subtitle"))
    return body


def spread_y_positions(desired, min_gap, lo, hi):
    if not desired:
        return []
    indexed = sorted(enumerate(desired), key=lambda item: item[1])
    ys = [min(max(y, lo), hi) for _, y in indexed]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + min_gap)
    overflow = ys[-1] - hi
    if overflow > 0:
        ys = [y - overflow for y in ys]
    for i in range(len(ys) - 2, -1, -1):
        ys[i] = min(ys[i], ys[i + 1] - min_gap)
    underflow = lo - ys[0]
    if underflow > 0:
        ys = [y + underflow for y in ys]
    result = [0.0] * len(desired)
    for (idx, _), y in zip(indexed, ys):
        result[idx] = y
    return result


def bar_grid(metrics, rows, path, title, subtitle, cols=3):
    panel_w = 360
    panel_h = 235
    margin_x = 38
    margin_y = 84
    width = margin_x * 2 + cols * panel_w
    nrows = math.ceil(len(metrics) / cols)
    height = margin_y + nrows * panel_h + 34
    body = title_block(title, subtitle)

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
        present = [v for v in values if v is not None]

        body.append(text(x0, y0 + 10, label, "label"))
        body.append(text(x0, y0 + 27, unit, "small"))
        if not present:
            body.append(text(chart_x, chart_y + chart_h / 2, "n/a", "subtitle", "middle"))
            continue

        ymax = max(present) * 1.15
        if ymax <= 0:
            ymax = 1.0
        body.append(line(chart_x, chart_y, chart_x, chart_y + chart_h))
        body.append(line(chart_x, chart_y + chart_h, chart_x + chart_w, chart_y + chart_h))
        for tick in range(1, 4):
            ty = chart_y + chart_h - chart_h * tick / 4
            body.append(line(chart_x, ty, chart_x + chart_w, ty, "grid"))
        gap = 18
        bar_w = (chart_w - gap * (len(VARIANTS) + 1)) / len(VARIANTS)
        for i, variant in enumerate(VARIANTS):
            value = values[i]
            bx = chart_x + gap + i * (bar_w + gap)
            if value is None:
                body.append(rect(bx, chart_y + chart_h - 4, bar_w, 4, "#cbd5e1"))
                body.append(text(bx + bar_w / 2, chart_y + chart_h + 17, variant["short"], "tick", "middle"))
                continue
            bh = chart_h * value / ymax
            by = chart_y + chart_h - bh
            body.append(rect(bx, by, bar_w, bh, variant["color"], rx=3))
            body.append(text(bx + bar_w / 2, by - 5, fmt_value(value, unit), "tick", "middle"))
            body.append(text(bx + bar_w / 2, chart_y + chart_h + 17, variant["short"], "tick", "middle"))
    save_svg(path, width, height, body)


def radar_plot(rows):
    width, height = 940, 760
    cx, cy, r = 470, 390, 245
    body = title_block(
        "STM32U5 Online Best-Ratio Radar",
        "Outer ring = best value; inner values are proportional ratios to the best variant, not min-max scores.",
    )
    metrics = RADAR_METRICS
    angles = [-math.pi / 2 + 2 * math.pi * i / len(metrics) for i in range(len(metrics))]
    for ring in range(1, 6):
        rr = r * ring / 5
        points = [(cx + rr * math.cos(a), cy + rr * math.sin(a)) for a in angles]
        body.append(polygon(points, "#e2e8f0", opacity=0.0, stroke_width=1))
    for idx, (_, label, _) in enumerate(metrics):
        a = angles[idx]
        x = cx + r * math.cos(a)
        y = cy + r * math.sin(a)
        lx = cx + (r + 46) * math.cos(a)
        ly = cy + (r + 46) * math.sin(a)
        body.append(line(cx, cy, x, y, color="#cbd5e1", width=1))
        anchor = "middle"
        if math.cos(a) > 0.25:
            anchor = "start"
        elif math.cos(a) < -0.25:
            anchor = "end"
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
        body.append(polyline(points + [points[0]], variant["color"], width=3.5, opacity=0.95))
    for variant, points in series:
        for x, y in points:
            body.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.8" fill="{variant["color"]}" stroke="#ffffff" stroke-width="1.2"/>')
    body.append(text(cx + 8, cy - 8, "0", "tick"))
    lx, ly = 36, height - 110
    for i, variant in enumerate(VARIANTS):
        y = ly + i * 24
        body.append(rect(lx, y - 12, 14, 14, variant["color"], rx=2))
        body.append(text(lx + 22, y, variant["label"], "label"))
    save_svg(PLOTS / "01_normalized_score_radar.svg", width, height, body)


def heatmap(rows, all_keys):
    keys = [k for k in all_keys if sum(rows[v["id"]].get(k) is not None for v in VARIANTS) >= 2]
    keys = sorted(keys)
    cell_w = 116
    row_h = 21
    left = 360
    top = 92
    width = left + cell_w * len(VARIANTS) + 58
    height = top + row_h * len(keys) + 54
    body = title_block(
        "All Online Numeric Metrics Heatmap",
        "Green means better after direction normalization where direction is inferable from the metric name.",
    )
    for i, variant in enumerate(VARIANTS):
        body.append(text(left + i * cell_w + cell_w / 2, top - 18, variant["short"], "label", "middle"))
    for r_idx, key in enumerate(keys):
        y = top + r_idx * row_h
        body.append(text(24, y + 14, key, "small"))
        values = [rows[v["id"]].get(key) for v in VARIANTS]
        sc = score(values, metric_direction(key))
        for c_idx, value in enumerate(values):
            x = left + c_idx * cell_w
            if value is None:
                fill = "#e5e7eb"
                label = ""
            else:
                s = 0.0 if sc[c_idx] is None else sc[c_idx]
                red = int(220 - 110 * s)
                green = int(82 + 120 * s)
                blue = int(82 + 60 * s)
                fill = f"rgb({red},{green},{blue})"
                label = fmt_value(value)
            body.append(rect(x, y, cell_w - 2, row_h - 2, fill))
            body.append(text(x + cell_w / 2, y + 14, label, "small", "middle"))
    save_svg(PLOTS / "07_all_numeric_metrics_heatmap.svg", width, height, body)


def log_metric_overview(rows):
    metrics = [
        ("memory.text_bytes", "Text bytes", "#2563eb"),
        ("memory.bss_bytes", "BSS bytes", "#0891b2"),
        ("static_sram_usage.static_sram_bytes", "Static SRAM bytes", "#16a34a"),
        ("elf_size_bytes", "ELF bytes", "#64748b"),
        ("cnn_latency_us.avg", "CNN latency us", "#dc2626"),
        ("cycles.avg", "Cycles", "#9333ea"),
        ("compute_estimate.mac_ops_per_inference", "MAC ops", "#ea580c"),
        ("energy_estimate.energy_per_inference_uj", "Energy uJ", "#be123c"),
        ("energy_estimate.mac_ops_per_joule", "MAC/J", "#0f766e"),
    ]
    width, height = 1120, 680
    left, right, top, bottom = 95, 850, 92, 580
    all_values = []
    for key, _, _ in metrics:
        for variant in VARIANTS:
            value = rows[variant["id"]].get(key)
            if value is not None and value > 0:
                all_values.append(value)
    ymin = 10 ** math.floor(math.log10(min(all_values)))
    ymax = 10 ** math.ceil(math.log10(max(all_values)))
    body = title_block(
        "Online Metric Overview on Log Scale",
        "Raw values with mixed units; useful for spotting broad magnitude changes.",
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
    end_labels = []
    for key, label, color in metrics:
        points = []
        for x, variant in zip(xs, VARIANTS):
            value = rows[variant["id"]].get(key)
            if value is None or value <= 0:
                continue
            y = bottom - (math.log10(value) - math.log10(ymin)) / (math.log10(ymax) - math.log10(ymin)) * (bottom - top)
            points.append((x, y))
        if len(points) >= 2:
            body.append(polyline(points, color, width=2.5, opacity=0.9))
        for x, y in points:
            body.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"/>')
        if points:
            end_labels.append((label, color, points[-1][1]))
    label_ys = spread_y_positions([item[2] + 4 for item in end_labels], 18, top + 14, bottom - 8)
    for (label, color, endpoint_y), label_y in zip(end_labels, label_ys):
        body.append(line(right, endpoint_y, right + 14, label_y - 4, color=color, width=1.2))
        body.append(text(right + 18, label_y, label, "label"))
    save_svg(PLOTS / "09_log_metric_overview.svg", width, height, body)


def axis_ticks(lo, hi, count=5):
    if hi <= lo:
        return [lo for _ in range(count)]
    return [lo + i * (hi - lo) / (count - 1) for i in range(count)]


def bubble_frontier(rows, spec):
    width, height = 860, 560
    left, right, top, bottom = 92, 760, 82, 450
    x_key, y_key, size_key = spec["x_key"], spec["y_key"], spec["size_key"]
    xs_raw = [rows[v["id"]].get(x_key) for v in VARIANTS]
    ys_raw = [rows[v["id"]].get(y_key) for v in VARIANTS]
    sizes_raw = [rows[v["id"]].get(size_key) for v in VARIANTS]
    xmin, xmax = min(xs_raw), max(xs_raw)
    ymin, ymax = min(ys_raw), max(ys_raw)
    smin, smax = min(sizes_raw), max(sizes_raw)
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

    body.append(text((left + right) / 2, height - 48, spec["x_label"], "label", "middle"))
    body.append(text(24, (top + bottom) / 2, spec["y_label"], "label", "middle", rotate=-90))
    body.append(text(right - 4, top - 20, f"Bubble size: {spec['size_label']}", "small", "end"))

    points = []
    for variant, xraw, yraw, sraw in zip(VARIANTS, xs_raw, ys_raw, sizes_raw):
        x = left + (xraw - xmin) / (xmax - xmin) * (right - left)
        y = bottom - (yraw - ymin) / (ymax - ymin) * (bottom - top)
        radius = 10 + 21 * ((sraw - smin) / (smax - smin) if smax > smin else 0.5)
        body.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" fill="{variant["color"]}" fill-opacity="0.80" stroke="#111827" stroke-width="1"/>')
        points.append((variant, x, y, radius, sraw))
    label_ys = spread_y_positions([y + 4 for _, _, y, _, _ in points], 38, top + 20, bottom - 26)
    for (variant, x, y, radius, sraw), label_y in zip(points, label_ys):
        label_x = x + radius + 10
        anchor = "start"
        if label_x > width - 148:
            label_x = x - radius - 10
            anchor = "end"
        sign = 1 if anchor == "start" else -1
        body.append(line(x + sign * radius * 0.72, y, label_x - sign * 5, label_y - 5, color=variant["color"], width=1.1))
        body.append(text(label_x, label_y, variant["label"], "label", anchor))
        body.append(text(label_x, label_y + 15, spec["size_fmt"](sraw), "tick", anchor))
    save_svg(PLOTS / spec["file"], width, height, body)


def frontier(rows):
    specs = [
        {
            "file": "10_latency_sram_energy_frontier.svg",
            "title": "Latency vs SRAM Frontier",
            "subtitle": "Lower-left is better; marker radius follows online energy per inference.",
            "x_key": "static_sram_usage.static_sram_bytes",
            "y_key": "cnn_latency_ms_avg",
            "size_key": "energy_estimate.energy_per_inference_uj",
            "x_label": "Static SRAM",
            "y_label": "CNN latency",
            "size_label": "Energy/inference",
            "x_fmt": lambda v: f"{v / 1024:.0f} KiB",
            "y_fmt": lambda v: f"{v:.0f} ms",
            "size_fmt": lambda v: f"{v / 1000:.1f} mJ",
        },
        {
            "file": "11_latency_energy_sram_frontier.svg",
            "title": "Latency vs Energy Frontier",
            "subtitle": "Lower-left is better; marker radius follows static SRAM.",
            "x_key": "energy_estimate.energy_per_inference_uj",
            "y_key": "cnn_latency_ms_avg",
            "size_key": "static_sram_usage.static_sram_bytes",
            "x_label": "Energy per inference",
            "y_label": "CNN latency",
            "size_label": "Static SRAM",
            "x_fmt": lambda v: f"{v / 1000:.1f} mJ",
            "y_fmt": lambda v: f"{v:.0f} ms",
            "size_fmt": lambda v: f"{v / 1024:.0f} KiB",
        },
        {
            "file": "12_text_sram_latency_frontier.svg",
            "title": "Flash Text vs SRAM Frontier",
            "subtitle": "Lower-left is better; marker radius follows CNN latency.",
            "x_key": "memory.text_bytes",
            "y_key": "static_sram_usage.static_sram_bytes",
            "size_key": "cnn_latency_ms_avg",
            "x_label": "Text memory",
            "y_label": "Static SRAM",
            "size_label": "CNN latency",
            "x_fmt": lambda v: f"{v / 1024:.0f} KiB",
            "y_fmt": lambda v: f"{v / 1024:.0f} KiB",
            "size_fmt": lambda v: f"{v:.0f} ms",
        },
        {
            "file": "13_weights_activations_latency_frontier.svg",
            "title": "AI Weights vs Activations Frontier",
            "subtitle": "Lower-left is better; marker radius follows CNN latency.",
            "x_key": "model_info.weights_bytes",
            "y_key": "model_info.activations_bytes",
            "size_key": "cnn_latency_ms_avg",
            "x_label": "AI weights",
            "y_label": "AI activations",
            "size_label": "CNN latency",
            "x_fmt": lambda v: f"{v / 1024:.0f} KiB",
            "y_fmt": lambda v: f"{v / 1024:.0f} KiB",
            "size_fmt": lambda v: f"{v:.0f} ms",
        },
        {
            "file": "14_throughput_energy_sram_frontier.svg",
            "title": "Throughput vs Energy Frontier",
            "subtitle": "Upper-left is better; marker radius follows static SRAM.",
            "x_key": "energy_estimate.energy_per_inference_uj",
            "y_key": "inferences_per_second_from_cnn_time",
            "size_key": "static_sram_usage.static_sram_bytes",
            "x_label": "Energy per inference",
            "y_label": "Throughput",
            "size_label": "Static SRAM",
            "x_fmt": lambda v: f"{v / 1000:.1f} mJ",
            "y_fmt": lambda v: f"{v:.1f}/s",
            "size_fmt": lambda v: f"{v / 1024:.0f} KiB",
        },
        {
            "file": "15_macs_latency_energy_frontier.svg",
            "title": "MAC Ops vs Latency Frontier",
            "subtitle": "Lower-left is better; marker radius follows energy per inference.",
            "x_key": "compute_estimate.mac_ops_per_inference",
            "y_key": "cnn_latency_ms_avg",
            "size_key": "energy_estimate.energy_per_inference_uj",
            "x_label": "MAC ops per inference",
            "y_label": "CNN latency",
            "size_label": "Energy/inference",
            "x_fmt": lambda v: f"{v / 1e6:.1f}M",
            "y_fmt": lambda v: f"{v:.0f} ms",
            "size_fmt": lambda v: f"{v / 1000:.1f} mJ",
        },
    ]

    for spec in specs:
        bubble_frontier(rows, spec)


def compute_delta_rows(rows):
    comparisons = [("v1_vs_v0", "v1", "v0"), ("v2_vs_v1", "v2", "v1"), ("v2_vs_v0", "v2", "v0")]
    rows_delta = []
    for key, label, unit, direction in KEY_METRICS:
        row = {"metric": label, "unit": unit, "direction": direction}
        for name, variant, base in comparisons:
            value = rows[variant].get(key)
            base_value = rows[base].get(key)
            if value is None or base_value in (None, 0):
                row[name + "_raw_pct"] = None
                row[name + "_improvement_pct"] = None
                continue
            raw_pct = (value - base_value) / base_value * 100.0
            improvement = -raw_pct if direction == "lower" else raw_pct
            row[name + "_raw_pct"] = raw_pct
            row[name + "_improvement_pct"] = improvement
        rows_delta.append(row)
    return rows_delta


def improvement_fill(value, max_abs):
    if value is None or math.isnan(value):
        return "#e5e7eb"
    if max_abs <= 0:
        return "#f8fafc"
    strength = min(1.0, abs(value) / max_abs)
    if value >= 0:
        start = (236, 253, 245)
        end = (15, 118, 110)
    else:
        start = (254, 242, 242)
        end = (185, 28, 28)
    rgb = [round(start[i] + (end[i] - start[i]) * strength) for i in range(3)]
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"


def improvement_heatmap(rows_delta):
    comparisons = [
        ("v1_vs_v0_improvement_pct", "v1 vs v0"),
        ("v2_vs_v1_improvement_pct", "v2 vs v1"),
        ("v2_vs_v0_improvement_pct", "v2 vs v0"),
    ]
    values = [
        abs(row[key])
        for row in rows_delta
        for key, _ in comparisons
        if row.get(key) is not None
    ]
    max_abs = max(values) if values else 1.0
    width = 1060
    left = 285
    top = 90
    row_h = 30
    cell_w = 205
    height = top + len(rows_delta) * row_h + 86
    body = title_block(
        "Relative Improvement Heatmap",
        "Green = better, red = regression; values are direction-aware percentages from the delta table.",
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
            label = pct(value) if value is not None else "n/a"
            body.append(text(x + cell_w / 2, y + 19, label, "label", "middle"))
    body.append(text(left, height - 34, "positive = deployment got better for that metric", "subtitle"))
    save_svg(PLOTS / "16_relative_improvement_heatmap.svg", width, height, body)


def improvement_bar_plot(rows_delta, comparison_key, file_name, title, subtitle):
    valid = [row for row in rows_delta if row.get(comparison_key) is not None]
    max_abs = max(abs(row[comparison_key]) for row in valid) if valid else 1.0
    width = 1120
    left = 380
    right = 1030
    top = 92
    row_h = 30
    height = top + len(valid) * row_h + 76
    zero_x = left + (right - left) * 0.5
    body = title_block(title, subtitle)
    body.append(line(zero_x, top - 12, zero_x, top + len(valid) * row_h, color="#334155", width=1.3))
    for tick in [-max_abs, -max_abs / 2, 0, max_abs / 2, max_abs]:
        x = zero_x + tick / max_abs * (right - left) * 0.48
        body.append(line(x, top - 8, x, top + len(valid) * row_h, "grid"))
        body.append(text(x, top - 18, f"{tick:+.0f}%", "tick", "middle"))
    for row_idx, row in enumerate(valid):
        y = top + row_idx * row_h
        value = row[comparison_key]
        body.append(text(32, y + 18, row["metric"], "label"))
        body.append(text(270, y + 18, row["direction"], "small", "end"))
        bar_w = abs(value) / max_abs * (right - left) * 0.48
        if value >= 0:
            x = zero_x
        else:
            x = zero_x - bar_w
        body.append(rect(x, y + 5, bar_w, 18, improvement_fill(value, max_abs), stroke="#ffffff", rx=3))
        label_x = x + bar_w + 8 if value >= 0 else x - 8
        anchor = "start" if value >= 0 else "end"
        body.append(text(label_x, y + 19, pct(value), "label", anchor))
    body.append(text(left, height - 30, "right of zero = improvement; left of zero = regression", "subtitle"))
    save_svg(PLOTS / file_name, width, height, body)


def stepwise_improvement_plot(rows_delta):
    selected = [
        "CNN latency avg",
        "Throughput",
        "Static SRAM",
        "Text",
        "AI weights",
        "AI activations",
        "MAC ops",
        "Energy/inference",
        "MAC/J",
    ]
    rows = [row for row in rows_delta if row["metric"] in selected]
    comparisons = [
        ("v1_vs_v0_improvement_pct", "v1-v0", "#0f766e"),
        ("v2_vs_v1_improvement_pct", "v2-v1", "#c2410c"),
        ("v2_vs_v0_improvement_pct", "v2-v0", "#475569"),
    ]
    max_abs = max(
        abs(row[key])
        for row in rows
        for key, _, _ in comparisons
        if row.get(key) is not None
    )
    width, height = 1240, 720
    left, right, top, bottom = 92, 1130, 108, 520
    body = title_block(
        "Stepwise Relative Improvements",
        "Selected deployment metrics; local steps use their own baseline, so v2-v0 is compounded and not additive.",
    )
    body.append(line(left, bottom, right, bottom))
    body.append(line(left, top, left, bottom))
    for tick in range(1, 5):
        y = bottom - tick * (bottom - top) / 4
        body.append(line(left, y, right, y, "grid"))
        body.append(text(left - 10, y + 4, f"{max_abs * tick / 4:.0f}%", "tick", "end"))
    group_w = (right - left) / len(rows)
    bar_w = 18
    for idx, row in enumerate(rows):
        cx = left + group_w * idx + group_w / 2
        body.append(text(cx, bottom + 58, row["metric"], "tick", "middle", rotate=-32))
        for comp_idx, (key, _, color) in enumerate(comparisons):
            value = row.get(key)
            if value is None:
                continue
            bh = abs(value) / max_abs * (bottom - top)
            x = cx - bar_w * 1.7 + comp_idx * (bar_w + 4)
            y = bottom - bh if value >= 0 else bottom
            fill = color if value >= 0 else "#b91c1c"
            body.append(rect(x, y, bar_w, bh, fill, rx=2))
            if value >= 0:
                label_y = max(top + 12, y - 6 - (comp_idx % 2) * 5)
            else:
                label_y = bottom + 18 + comp_idx * 12
            body.append(text(x + bar_w / 2, label_y, pct(value), "tick", "middle"))
    lx, ly = 910, 82
    for i, (_, label, color) in enumerate(comparisons):
        x = lx + i * 78
        body.append(rect(x, ly - 12, 14, 14, color, rx=2))
        body.append(text(x + 20, ly, label, "label"))
    save_svg(PLOTS / "20_stepwise_relative_improvements.svg", width, height, body)


def improvement_profile_plot(rows_delta, metric_names, file_name, title, subtitle):
    rows = [row for row in rows_delta if row["metric"] in metric_names]
    comparisons = [
        ("v1_vs_v0_improvement_pct", "v1 vs v0", "#0f766e"),
        ("v2_vs_v1_improvement_pct", "v2 vs v1", "#c2410c"),
        ("v2_vs_v0_improvement_pct", "v2 vs v0", "#475569"),
    ]
    width, height = 980, 560
    left, right, top, bottom = 95, 850, 88, 455
    max_abs = max(
        abs(row[key])
        for row in rows
        for key, _, _ in comparisons
        if row.get(key) is not None
    )
    body = title_block(title, subtitle)
    zero_y = bottom - (bottom - top) * 0.25
    body.append(line(left, zero_y, right, zero_y, color="#334155", width=1.2))
    body.append(line(left, top, left, bottom))
    for tick in [-max_abs / 2, 0, max_abs / 2, max_abs]:
        y = zero_y - tick / max_abs * (bottom - top) * 0.62
        body.append(line(left, y, right, y, "grid"))
        body.append(text(left - 12, y + 4, f"{tick:+.0f}%", "tick", "end"))
    xs = [left + i * (right - left) / (len(rows) - 1) for i in range(len(rows))]
    for x, row in zip(xs, rows):
        body.append(text(x, bottom + 24, row["metric"], "tick", "middle", rotate=-25))
    end_labels = []
    for key, label, color in comparisons:
        points = []
        for x, row in zip(xs, rows):
            value = row.get(key)
            if value is None:
                continue
            y = zero_y - value / max_abs * (bottom - top) * 0.62
            points.append((x, y))
        if len(points) > 1:
            body.append(polyline(points, color, width=3.0, opacity=0.95))
        for x, y in points:
            body.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5.2" fill="{color}" stroke="#ffffff" stroke-width="1.2"/>')
        if points:
            end_labels.append((label, color, points[-1][1]))
    label_ys = spread_y_positions([item[2] + 4 for item in end_labels], 18, top + 12, bottom - 6)
    for (label, color, endpoint_y), label_y in zip(end_labels, label_ys):
        body.append(line(right, endpoint_y, right + 14, label_y - 4, color=color, width=1.2))
        body.append(text(right + 18, label_y, label, "label"))
    save_svg(PLOTS / file_name, width, height, body)


def improvement_plots(rows_delta):
    improvement_heatmap(rows_delta)
    improvement_bar_plot(
        rows_delta,
        "v2_vs_v0_improvement_pct",
        "17_v2_vs_v0_relative_improvement_bars.svg",
        "Relative Improvement: v2 vs v0",
        "Final pruned-quantized deployment against the unquantized baseline; direction-aware percentages.",
    )
    improvement_bar_plot(
        rows_delta,
        "v2_vs_v1_improvement_pct",
        "18_v2_vs_v1_relative_improvement_bars.svg",
        "Relative Improvement: v2 vs v1",
        "Effect of pruning on top of quantization; negative bars show regressions.",
    )
    improvement_bar_plot(
        rows_delta,
        "v1_vs_v0_improvement_pct",
        "19_v1_vs_v0_relative_improvement_bars.svg",
        "Relative Improvement: v1 vs v0",
        "Effect of quantization against the unquantized baseline.",
    )
    stepwise_improvement_plot(rows_delta)
    improvement_profile_plot(
        rows_delta,
        ["CNN latency avg", "Cycles avg", "Throughput", "MOPS/s", "MAC/cycle", "Energy/inference", "MAC/J"],
        "21_compute_efficiency_improvement_profile.svg",
        "Compute and Energy Improvement Profile",
        "Line plot of relative improvements for latency, throughput, efficiency, and energy metrics.",
    )
    improvement_profile_plot(
        rows_delta,
        ["Text", "Data", "BSS", "Static SRAM", "ELF size", "AI weights", "AI activations", "MAC ops"],
        "22_memory_model_improvement_profile.svg",
        "Memory and Model Improvement Profile",
        "Line plot of relative improvements for firmware memory and model-size metrics.",
    )


def load_power_reports():
    reports = {}
    for variant in VARIANTS:
        path = power_analysis_dir(variant["id"]) / "peak_report.json"
        if path.exists():
            reports[variant["id"]] = json.loads(path.read_text())
    return reports


def representative_peak_number(report):
    target = report.get("avg_peak_duration_ms")
    peaks = report.get("per_peak", [])
    if not peaks:
        return None
    if target is None:
        return peaks[len(peaks) // 2]["peak_number"]
    best = min(peaks, key=lambda peak: abs(peak["duration_ms"] - target))
    return best["peak_number"]


def decimate_points(points, max_points):
    if len(points) <= max_points:
        return points
    step = math.ceil(len(points) / max_points)
    out = []
    for idx in range(0, len(points), step):
        chunk = points[idx:idx + step]
        if not chunk:
            continue
        out.append((
            sum(point[0] for point in chunk) / len(chunk),
            sum(point[1] for point in chunk) / len(chunk),
        ))
    return out


def read_power_trace_windows(report, pad_ms=20.0, max_points=900):
    csv_path = Path(report["csv"])
    peaks = report.get("per_peak", [])
    intervals = []
    for peak in peaks:
        intervals.append({
            "peak_number": int(peak["peak_number"]),
            "read_start": peak["start_ms"] - pad_ms,
            "read_end": peak["end_ms"] + pad_ms,
            "start_ms": peak["start_ms"],
            "end_ms": peak["end_ms"],
            "duration_ms": peak["duration_ms"],
            "baseline_ma": peak["baseline_current_ua"] / 1000.0,
            "energy_total_uj": peak["energy_total_uj"],
            "energy_uj": peak["energy_uj"],
            "points": [],
        })
    intervals.sort(key=lambda item: item["read_start"])
    current_idx = 0
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            time_ms = float(row["Timestamp(ms)"])
            while current_idx < len(intervals) and time_ms > intervals[current_idx]["read_end"]:
                current_idx += 1
            if current_idx >= len(intervals):
                break
            interval = intervals[current_idx]
            if time_ms < interval["read_start"]:
                continue
            current_ma = float(row["Current(uA)"]) / 1000.0
            interval["points"].append((time_ms - interval["start_ms"], current_ma))
    return {
        item["peak_number"]: {
            **{key: value for key, value in item.items() if key != "points"},
            "points": decimate_points(item["points"], max_points),
        }
        for item in intervals
    }


def power_tables(reports):
    summary_rows = []
    for variant in VARIANTS:
        report = reports.get(variant["id"])
        if not report:
            continue
        selected = representative_peak_number(report)
        summary_rows.append({
            "variant": variant["id"],
            "label": variant["label"],
            "detected_peaks": report.get("detected_peaks"),
            "selected_peak": selected,
            "avg_peak_duration_ms": report.get("avg_peak_duration_ms"),
            "avg_peak_mean_current_ma": report.get("avg_peak_mean_current_ua", 0.0) / 1000.0,
            "avg_peak_mean_excess_current_ma": report.get("avg_peak_mean_excess_current_ua", 0.0) / 1000.0,
            "avg_peak_energy_total_uj": report.get("avg_peak_energy_total_uj"),
            "avg_peak_energy_uj": report.get("avg_peak_energy_uj"),
            "duration_vs_summary_ratio": report.get("duration_vs_summary_ratio"),
        })

    with (TABLES / "power_peak_summary.csv").open("w", newline="") as f:
        fieldnames = list(summary_rows[0].keys()) if summary_rows else ["variant"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    md = [
        "| Variant | Peaks | Selected window | Avg duration | Avg mean current | Avg total energy | Avg excess energy | Duration/summary |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        md.append(
            f"| {row['label']} | {row['detected_peaks']} | {row['selected_peak']} | "
            f"{row['avg_peak_duration_ms']:.2f} ms | {row['avg_peak_mean_current_ma']:.3f} mA | "
            f"{row['avg_peak_energy_total_uj'] / 1000:.2f} mJ | {row['avg_peak_energy_uj']:.2f} uJ | "
            f"{row['duration_vs_summary_ratio']:.3f} |"
        )
    (TABLES / "power_peak_summary.md").write_text("\n".join(md) + "\n")

    with (TABLES / "power_peak_metrics_all.csv").open("w", newline="") as f:
        fieldnames = [
            "variant", "peak_number", "duration_ms", "mean_current_ma", "mean_excess_current_ma",
            "charge_total_uc", "charge_uc", "energy_total_uj", "energy_uj",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for variant in VARIANTS:
            report = reports.get(variant["id"])
            if not report:
                continue
            for peak in report.get("per_peak", []):
                writer.writerow({
                    "variant": variant["id"],
                    "peak_number": peak["peak_number"],
                    "duration_ms": peak["duration_ms"],
                    "mean_current_ma": peak["mean_current_ua"] / 1000.0,
                    "mean_excess_current_ma": peak["mean_excess_current_ua"] / 1000.0,
                    "charge_total_uc": peak["charge_total_uc"],
                    "charge_uc": peak["charge_uc"],
                    "energy_total_uj": peak["energy_total_uj"],
                    "energy_uj": peak["energy_uj"],
                })


def scale(value, lo, hi, a, b):
    if hi == lo:
        return (a + b) / 2
    return a + (value - lo) / (hi - lo) * (b - a)


def power_window_overlay(reports, windows_by_variant, selection_by_variant, file_name, title, subtitle):
    width, height = 1120, 660
    left, right, top, bottom = 92, 920, 92, 520
    selected = []
    for variant in VARIANTS:
        peak_number = selection_by_variant.get(variant["id"])
        window = windows_by_variant.get(variant["id"], {}).get(peak_number)
        if window and window["points"]:
            selected.append((variant, peak_number, window))
    x_values = [point[0] for _, _, window in selected for point in window["points"] if point[0] >= 0.0]
    y_values = [point[1] for _, _, window in selected for point in window["points"]]
    if not x_values or not y_values:
        return
    xmin, xmax = 0.0, max(x_values)
    ymin, ymax = min(y_values), max(y_values)
    ypad = (ymax - ymin) * 0.12
    ymin, ymax = ymin - ypad, ymax + ypad
    body = title_block(title, subtitle)
    body.append(line(left, top, left, bottom))
    body.append(line(left, bottom, right, bottom))
    for idx, value in enumerate(axis_ticks(xmin, xmax, 6)):
        x = left + idx * (right - left) / 5
        body.append(line(x, top, x, bottom, "grid"))
        body.append(text(x, bottom + 22, f"{value:.0f} ms", "tick", "middle"))
    for idx, value in enumerate(axis_ticks(ymin, ymax, 6)):
        y = bottom - idx * (bottom - top) / 5
        body.append(line(left, y, right, y, "grid"))
        body.append(text(left - 10, y + 4, f"{value:.1f} mA", "tick", "end"))
    body.append(text((left + right) / 2, height - 54, "Time relative to detected inference start", "label", "middle"))
    body.append(text(24, (top + bottom) / 2, "Current", "label", "middle", rotate=-90))
    end_labels = []
    for variant, peak_number, window in selected:
        raw_points = window["points"]
        points = [
            (
                scale(t, xmin, xmax, left, right),
                scale(current, ymin, ymax, bottom, top),
            )
            for t, current in raw_points
            if t >= 0.0
        ]
        if not points:
            continue
        body.append(polyline(points, variant["color"], width=2.2, opacity=0.9))
        start_x = scale(0.0, xmin, xmax, left, right)
        end_x = scale(window["duration_ms"], xmin, xmax, left, right)
        body.append(line(start_x, top, start_x, bottom, color=variant["color"], width=1, dash="4 4"))
        body.append(line(end_x, top, end_x, bottom, color=variant["color"], width=1, dash="4 4"))
        anchor_time = max(0.0, min(window["duration_ms"] - 8.0, window["duration_ms"] * 0.72))
        anchor_t, anchor_current = min((point for point in raw_points if point[0] >= 0.0), key=lambda point: abs(point[0] - anchor_time))
        anchor_x = scale(anchor_t, xmin, xmax, left, right)
        anchor_y = scale(anchor_current, ymin, ymax, bottom, top)
        end_labels.append((variant, peak_number, anchor_x, anchor_y, window))
    label_ys = spread_y_positions([item[3] + 4 for item in end_labels], 32, top + 20, bottom - 32)
    for (variant, peak_number, anchor_x, anchor_y, window), label_y in zip(end_labels, label_ys):
        label_x = right + 22
        label_w = 182
        body.append(line(anchor_x, anchor_y, label_x - 6, label_y - 4, color=variant["color"], width=1.15))
        body.append(rect(label_x - 5, label_y - 14, label_w, 34, "#ffffff", stroke="#e2e8f0", opacity=0.88, rx=4))
        body.append(text(label_x, label_y, f"{variant['label']} p{peak_number}", "label"))
        body.append(text(label_x, label_y + 15, f"{window['duration_ms']:.1f} ms, {window['energy_total_uj']/1000:.2f} mJ", "small"))
    save_svg(PLOTS / file_name, width, height, body)


def power_six_peak_grid(reports, windows_by_variant):
    width, height = 1280, 840
    cols = 3
    panel_w, panel_h = 390, 245
    left_margin, top_margin = 55, 92
    body = title_block(
        "Power Windows: Six Inference Peaks",
        "Each panel overlays v0/v1/v2 for the same peak index; traces are aligned to detected inference start.",
    )
    all_points = [
        point
        for variant_windows in windows_by_variant.values()
        for window in variant_windows.values()
        for point in window["points"]
    ]
    ymin = min(current for _, current in all_points)
    ymax = max(current for _, current in all_points)
    ypad = (ymax - ymin) * 0.12
    ymin, ymax = ymin - ypad, ymax + ypad
    for peak_number in range(1, 7):
        col = (peak_number - 1) % cols
        row = (peak_number - 1) // cols
        x0 = left_margin + col * panel_w
        y0 = top_margin + row * panel_h
        left, right = x0 + 40, x0 + panel_w - 28
        top, bottom = y0 + 32, y0 + panel_h - 46
        x_max = max(
            windows_by_variant[variant["id"]][peak_number]["duration_ms"]
            for variant in VARIANTS
            if peak_number in windows_by_variant.get(variant["id"], {})
        ) + 30
        x_min = -20
        body.append(text(x0, y0 + 12, f"Peak {peak_number}", "label"))
        body.append(line(left, top, left, bottom))
        body.append(line(left, bottom, right, bottom))
        for tick in range(1, 4):
            y = bottom - tick * (bottom - top) / 4
            body.append(line(left, y, right, y, "grid"))
        for variant in VARIANTS:
            window = windows_by_variant.get(variant["id"], {}).get(peak_number)
            if not window:
                continue
            points = [
                (
                    scale(t, x_min, x_max, left, right),
                    scale(current, ymin, ymax, bottom, top),
                )
                for t, current in decimate_points(window["points"], 260)
            ]
            body.append(polyline(points, variant["color"], width=1.8, opacity=0.85))
            end_x = scale(window["duration_ms"], x_min, x_max, left, right)
            body.append(line(end_x, top, end_x, bottom, color=variant["color"], width=0.8, dash="3 4"))
        body.append(text(left, bottom + 17, "0", "tick", "middle"))
        body.append(text(right, bottom + 17, f"{x_max:.0f} ms", "tick", "middle"))
        body.append(text(left - 8, top + 4, f"{ymax:.1f}", "tick", "end"))
        body.append(text(left - 8, bottom + 4, f"{ymin:.1f}", "tick", "end"))
    lx, ly = 940, 58
    for i, variant in enumerate(VARIANTS):
        x = lx + i * 100
        body.append(rect(x, ly - 12, 14, 14, variant["color"], rx=2))
        body.append(text(x + 20, ly, variant["short"], "label"))
    save_svg(PLOTS / "25_power_six_peak_window_grid.svg", width, height, body)


def power_average_bars(reports):
    rows = {}
    for variant in VARIANTS:
        report = reports.get(variant["id"])
        if not report:
            continue
        rows[variant["id"]] = {
            "avg_peak_duration_ms": report["avg_peak_duration_ms"],
            "avg_peak_mean_current_ma": report["avg_peak_mean_current_ua"] / 1000.0,
            "avg_peak_mean_excess_current_ma": report["avg_peak_mean_excess_current_ua"] / 1000.0,
            "avg_peak_energy_total_uj": report["avg_peak_energy_total_uj"],
            "avg_peak_energy_uj": report["avg_peak_energy_uj"],
            "duration_vs_summary_ratio": report["duration_vs_summary_ratio"],
        }
    bar_grid(
        [
            ("avg_peak_duration_ms", "Peak duration", "ms", "lower"),
            ("avg_peak_mean_current_ma", "Mean current", "mA", "lower"),
            ("avg_peak_mean_excess_current_ma", "Mean excess current", "mA", "lower"),
            ("avg_peak_energy_total_uj", "Total peak energy", "uJ", "lower"),
            ("avg_peak_energy_uj", "Excess peak energy", "uJ", "lower"),
            ("duration_vs_summary_ratio", "Duration / firmware latency", "ratio", "lower"),
        ],
        rows,
        PLOTS / "26_power_peak_average_bars.svg",
        "Power Peak Average Metrics",
        "Averages from the six detected online inference windows in each power trace.",
        cols=3,
    )


def power_energy_duration_scatter(reports):
    width, height = 900, 600
    left, right, top, bottom = 92, 760, 92, 500
    points = []
    for variant in VARIANTS:
        report = reports.get(variant["id"])
        if not report:
            continue
        for peak in report.get("per_peak", []):
            points.append((variant, peak))
    xvals = [peak["duration_ms"] for _, peak in points]
    yvals = [peak["energy_total_uj"] / 1000.0 for _, peak in points]
    xmin, xmax = min(xvals), max(xvals)
    ymin, ymax = min(yvals), max(yvals)
    xpad = (xmax - xmin) * 0.1
    ypad = (ymax - ymin) * 0.12
    xmin, xmax = xmin - xpad, xmax + xpad
    ymin, ymax = ymin - ypad, ymax + ypad
    body = title_block(
        "Power Peak Energy vs Duration",
        "Each point is one detected inference window; lower-left is better.",
    )
    body.append(line(left, top, left, bottom))
    body.append(line(left, bottom, right, bottom))
    for idx, value in enumerate(axis_ticks(xmin, xmax, 5)):
        x = left + idx * (right - left) / 4
        body.append(line(x, top, x, bottom, "grid"))
        body.append(text(x, bottom + 22, f"{value:.0f} ms", "tick", "middle"))
    for idx, value in enumerate(axis_ticks(ymin, ymax, 5)):
        y = bottom - idx * (bottom - top) / 4
        body.append(line(left, y, right, y, "grid"))
        body.append(text(left - 10, y + 4, f"{value:.1f} mJ", "tick", "end"))
    body.append(text((left + right) / 2, height - 42, "Detected inference-window duration", "label", "middle"))
    body.append(text(24, (top + bottom) / 2, "Total peak energy", "label", "middle", rotate=-90))
    for variant, peak in points:
        x = scale(peak["duration_ms"], xmin, xmax, left, right)
        y = scale(peak["energy_total_uj"] / 1000.0, ymin, ymax, bottom, top)
        body.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="6" fill="{variant["color"]}" fill-opacity="0.78" stroke="#111827" stroke-width="0.7"/>')
        body.append(text(x + 8, y + 4, str(int(peak["peak_number"])), "tick"))
    lx, ly = 610, 88
    for i, variant in enumerate(VARIANTS):
        y = ly + i * 22
        body.append(rect(lx, y - 12, 14, 14, variant["color"], rx=2))
        body.append(text(lx + 22, y, variant["label"], "label"))
    save_svg(PLOTS / "27_power_energy_duration_scatter.svg", width, height, body)


def edge_panel(body, selected, x_transform, x_range, y_range, x0, y0, panel_w, panel_h, title, zero_label):
    left, right = x0 + 58, x0 + panel_w - 105
    top, bottom = y0 + 38, y0 + panel_h - 48
    xmin, xmax = x_range
    ymin, ymax = y_range
    body.append(text(x0 + 12, y0 + 16, title, "label"))
    body.append(line(left, top, left, bottom))
    body.append(line(left, bottom, right, bottom))
    zero_x = scale(0.0, xmin, xmax, left, right)
    body.append(line(zero_x, top, zero_x, bottom, color="#111827", width=1.2, dash="4 4"))
    body.append(text(zero_x, top - 8, zero_label, "tick", "middle"))
    for idx, value in enumerate(axis_ticks(xmin, xmax, 5)):
        x = left + idx * (right - left) / 4
        body.append(line(x, top, x, bottom, "grid"))
        body.append(text(x, bottom + 19, f"{value:.0f} ms", "tick", "middle"))
    for idx, value in enumerate(axis_ticks(ymin, ymax, 5)):
        y = bottom - idx * (bottom - top) / 4
        body.append(line(left, y, right, y, "grid"))
        body.append(text(left - 10, y + 4, f"{value:.1f}", "tick", "end"))
    labels = []
    for variant, peak_number, window in selected:
        edge_points = []
        for t, current in window["points"]:
            xt = x_transform(t, window)
            if xmin <= xt <= xmax:
                edge_points.append((xt, current))
        if len(edge_points) < 2:
            continue
        points = [
            (
                scale(t, xmin, xmax, left, right),
                scale(current, ymin, ymax, bottom, top),
            )
            for t, current in edge_points
        ]
        body.append(polyline(points, variant["color"], width=2.0, opacity=0.9))
        label_anchor = points[min(len(points) - 1, int(len(points) * 0.75))]
        labels.append((variant, peak_number, label_anchor[0], label_anchor[1]))
    label_ys = spread_y_positions([item[3] + 4 for item in labels], 22, top + 16, bottom - 16)
    for (variant, peak_number, anchor_x, anchor_y), label_y in zip(labels, label_ys):
        label_x = right + 16
        body.append(line(anchor_x, anchor_y, label_x - 6, label_y - 4, color=variant["color"], width=1.0))
        body.append(text(label_x, label_y, f"{variant['short']} p{peak_number}", "label"))


def power_edge_zoom_overlay(windows_by_variant, selection_by_variant, file_name, title, subtitle):
    selected = []
    for variant in VARIANTS:
        peak_number = selection_by_variant.get(variant["id"])
        window = windows_by_variant.get(variant["id"], {}).get(peak_number)
        if window and window["points"]:
            selected.append((variant, peak_number, window))
    if not selected:
        return
    rising_range = (-55.0, 115.0)
    falling_range = (-115.0, 55.0)
    y_values = []
    for _, _, window in selected:
        for t, current in window["points"]:
            if rising_range[0] <= t <= rising_range[1]:
                y_values.append(current)
            t_end = t - window["duration_ms"]
            if falling_range[0] <= t_end <= falling_range[1]:
                y_values.append(current)
    ymin, ymax = min(y_values), max(y_values)
    ypad = (ymax - ymin) * 0.14
    y_range = (ymin - ypad, ymax + ypad)
    width, height = 1220, 600
    body = title_block(title, subtitle)
    edge_panel(
        body,
        selected,
        lambda t, window: t,
        rising_range,
        y_range,
        34,
        92,
        570,
        430,
        "Rising edge aligned to detected start",
        "start",
    )
    edge_panel(
        body,
        selected,
        lambda t, window: t - window["duration_ms"],
        falling_range,
        y_range,
        610,
        92,
        570,
        430,
        "Falling edge aligned to detected end",
        "end",
    )
    body.append(text(610, height - 38, "Current in mA; x-axis is local time around each detected edge.", "subtitle", "middle"))
    lx, ly = 42, height - 36
    for i, variant in enumerate(VARIANTS):
        x = lx + i * 128
        body.append(rect(x, ly - 12, 14, 14, variant["color"], rx=2))
        body.append(text(x + 20, ly, variant["label"], "label"))
    save_svg(PLOTS / file_name, width, height, body)


def power_analysis_plots_and_tables():
    reports = load_power_reports()
    if len(reports) != len(VARIANTS):
        return
    (OUT / "power_peak_sources.json").write_text(json.dumps({
        variant["id"]: str(power_analysis_dir(variant["id"]))
        for variant in VARIANTS
    }, indent=2) + "\n")
    power_tables(reports)
    windows_by_variant = {
        variant["id"]: read_power_trace_windows(reports[variant["id"]], pad_ms=80.0, max_points=1400)
        for variant in VARIANTS
    }
    representative_selection = {
        variant["id"]: representative_peak_number(reports[variant["id"]])
        for variant in VARIANTS
    }
    power_window_overlay(
        reports,
        windows_by_variant,
        representative_selection,
        "23_power_representative_inference_window_overlay.svg",
        "Power Trace Overlay: Representative Inference Window",
        "One detected window per variant, selected by duration closest to that variant's six-peak average.",
    )
    power_window_overlay(
        reports,
        windows_by_variant,
        {variant["id"]: 3 for variant in VARIANTS},
        "24_power_peak03_inference_window_overlay.svg",
        "Power Trace Overlay: Peak 3",
        "Same peak index from all three traces, aligned to detected inference start.",
    )
    power_six_peak_grid(reports, windows_by_variant)
    power_average_bars(reports)
    power_energy_duration_scatter(reports)
    power_edge_zoom_overlay(
        windows_by_variant,
        representative_selection,
        "28_power_representative_edge_zoom_overlay.svg",
        "Power Edge Zoom: Representative Windows",
        "Rising and falling edges shown separately so all variants stay readable despite different durations.",
    )
    power_edge_zoom_overlay(
        windows_by_variant,
        {variant["id"]: 3 for variant in VARIANTS},
        "29_power_peak03_edge_zoom_overlay.svg",
        "Power Edge Zoom: Peak 3",
        "Same peak index for all variants; rising edge and falling edge are aligned independently.",
    )


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

    md = ["| Metric | Unit | Direction | v0 unquantized | v1 quantized | v2 pruned quantized |",
          "|---|---:|---|---:|---:|---:|"]
    for key, label, unit, direction in KEY_METRICS:
        vals = [fmt_value(rows[v["id"]].get(key), unit) for v in VARIANTS]
        md.append(f"| {label} | {unit} | {direction} | {vals[0]} | {vals[1]} | {vals[2]} |")
    (TABLES / "online_key_metrics.md").write_text("\n".join(md) + "\n")

    tex = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Metric & Unit & v0 & v1 & v2 \\\\",
        "\\midrule",
    ]
    for key, label, unit, _ in KEY_METRICS:
        vals = [fmt_value(rows[v["id"]].get(key), unit) for v in VARIANTS]
        tex.append(f"{label} & {unit} & {vals[0]} & {vals[1]} & {vals[2]} \\\\")
    tex += ["\\bottomrule", "\\end{tabular}", ""]
    (TABLES / "online_key_metrics.tex").write_text("\n".join(tex))

    comparisons = [("v1_vs_v0", "v1", "v0"), ("v2_vs_v1", "v2", "v1"), ("v2_vs_v0", "v2", "v0")]
    rows_delta = compute_delta_rows(rows)

    with (TABLES / "online_metric_deltas.csv").open("w", newline="") as f:
        fieldnames = ["metric", "unit", "direction"]
        for comp, _, _ in comparisons:
            fieldnames += [comp + "_raw_pct", comp + "_improvement_pct"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_delta)

    md = ["| Metric | Direction | v1 vs v0 improvement | v2 vs v1 improvement | v2 vs v0 improvement |",
          "|---|---|---:|---:|---:|"]
    for row in rows_delta:
        md.append(
            f"| {row['metric']} | {row['direction']} | "
            f"{pct(row.get('v1_vs_v0_improvement_pct'))} | "
            f"{pct(row.get('v2_vs_v1_improvement_pct'))} | "
            f"{pct(row.get('v2_vs_v0_improvement_pct'))} |"
        )
    (TABLES / "online_metric_deltas.md").write_text("\n".join(md) + "\n")


def write_readme(rows):
    def v(variant, key):
        return rows[variant].get(key)

    lines = [
        "# STM32U5 ai85kws20netv3 Online Results",
        "",
        "Source data: online `on.json` files from `Experiments/General Profiling/U5`.",
        "Missing v0 AI-core fields are backfilled from the matching X-CUBE-AI `network_generate_report.txt` because that older firmware log did not emit `model_info` over UART.",
        "",
        "Variants:",
        "- `v0`: unquantized deployment",
        "- `v1`: quantized deployment",
        "- `v2`: pruned quantized deployment",
        "",
        "Key online findings:",
        f"- CNN latency: v0 `{v('v0','cnn_latency_ms_avg'):.2f} ms`, v1 `{v('v1','cnn_latency_ms_avg'):.2f} ms`, v2 `{v('v2','cnn_latency_ms_avg'):.2f} ms`.",
        f"- Static SRAM: v0 `{v('v0','static_sram_usage.static_sram_bytes')/1024:.1f} KiB`, v1 `{v('v1','static_sram_usage.static_sram_bytes')/1024:.1f} KiB`, v2 `{v('v2','static_sram_usage.static_sram_bytes')/1024:.1f} KiB`.",
        f"- Energy per inference estimate: v0 `{v('v0','energy_estimate.energy_per_inference_uj'):.1f} uJ`, v1 `{v('v1','energy_estimate.energy_per_inference_uj'):.1f} uJ`, v2 `{v('v2','energy_estimate.energy_per_inference_uj'):.1f} uJ`.",
        f"- v2 vs v1 latency improvement: `{(v('v1','cnn_latency_ms_avg') - v('v2','cnn_latency_ms_avg')) / v('v1','cnn_latency_ms_avg') * 100:.2f}%`.",
        f"- v2 vs v1 static SRAM improvement: `{(v('v1','static_sram_usage.static_sram_bytes') - v('v2','static_sram_usage.static_sram_bytes')) / v('v1','static_sram_usage.static_sram_bytes') * 100:.2f}%`.",
        "",
        "Generated tables are in `tables/`; generated SVG plots are in `plots/`.",
        "",
    ]
    (OUT / "README.md").write_text("\n".join(lines))


def write_dashboard():
    plot_files = sorted(PLOTS.glob("*.svg"))
    key_table = (TABLES / "online_key_metrics.md").read_text()
    delta_table = (TABLES / "online_metric_deltas.md").read_text()
    power_table_path = TABLES / "power_peak_summary.md"
    power_table = power_table_path.read_text() if power_table_path.exists() else None

    def md_table_to_html(md):
        lines = [line for line in md.splitlines() if line.startswith("|")]
        if len(lines) < 2:
            return ""
        header = [cell.strip() for cell in lines[0].strip("|").split("|")]
        rows = []
        for line in lines[2:]:
            rows.append([cell.strip() for cell in line.strip("|").split("|")])
        html_lines = ["<table>", "<thead><tr>"]
        html_lines += [f"<th>{html.escape(cell)}</th>" for cell in header]
        html_lines += ["</tr></thead>", "<tbody>"]
        for row in rows:
            html_lines.append("<tr>")
            html_lines += [f"<td>{html.escape(cell)}</td>" for cell in row]
            html_lines.append("</tr>")
        html_lines += ["</tbody></table>"]
        return "\n".join(html_lines)

    body = [
        "<!doctype html>",
        "<html lang=\"en\">",
        "<head>",
        "<meta charset=\"utf-8\">",
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">",
        "<title>STM32U5 ai85kws20netv3 Online Results</title>",
        "<style>",
        "body{font-family:Arial,Helvetica,sans-serif;margin:0;background:#f8fafc;color:#111827}",
        "main{max-width:1240px;margin:0 auto;padding:32px}",
        "h1{font-size:30px;margin:0 0 6px}",
        "p{color:#475569;margin:0 0 24px}",
        "section{margin:28px 0}",
        ".plot{background:#fff;border:1px solid #e2e8f0;border-radius:8px;margin:18px 0;padding:14px}",
        ".plot img{width:100%;height:auto;display:block}",
        "table{width:100%;border-collapse:collapse;background:#fff;border:1px solid #e2e8f0;font-size:13px}",
        "th,td{padding:8px 10px;border-bottom:1px solid #e2e8f0;text-align:right}",
        "th:first-child,td:first-child{text-align:left}",
        "th{background:#e2e8f0}",
        "code{background:#e2e8f0;padding:1px 4px;border-radius:4px}",
        "</style>",
        "</head>",
        "<body><main>",
        "<h1>STM32U5 ai85kws20netv3 Online Results</h1>",
        "<p>Online-only comparison of v0, v1, and v2 result summaries.</p>",
        "<section><h2>Key Metrics</h2>",
        md_table_to_html(key_table),
        "</section>",
        "<section><h2>Relative Improvements</h2>",
        md_table_to_html(delta_table),
        "</section>",
    ]
    if power_table:
        body += [
            "<section><h2>Power Peak Summary</h2>",
            md_table_to_html(power_table),
            "</section>",
        ]
    body += [
        "<section><h2>Plots</h2>",
    ]
    for plot in plot_files:
        rel = plot.relative_to(OUT)
        body.append(f'<div class="plot"><h3>{html.escape(plot.stem)}</h3><img src="{html.escape(str(rel))}" alt="{html.escape(plot.stem)}"></div>')
    body += ["</section>", "</main></body></html>"]
    (OUT / "index.html").write_text("\n".join(body))


def main():
    ensure_dirs()
    raw = {}
    rows = {}
    for variant in VARIANTS:
        data = json.loads(variant["path"].read_text())
        raw[variant["id"]] = data
        rows[variant["id"]] = flatten(data)
        fill_report_fallback_metrics(data, rows[variant["id"]])
        rows[variant["id"]]["source_path"] = str(variant["path"])

    all_keys = sorted({key for variant_rows in rows.values() for key in variant_rows if key != "source_path"})

    (OUT / "online_sources.json").write_text(json.dumps({
        variant["id"]: str(variant["path"]) for variant in VARIANTS
    }, indent=2) + "\n")
    (OUT / "online_flat_metrics.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")

    write_tables(rows, all_keys)
    write_readme(rows)

    radar_plot(rows)
    bar_grid(
        [
            ("cnn_latency_ms_avg", "CNN latency avg", "ms", "lower"),
            ("cycles.avg", "Cycles avg", "cycles", "lower"),
            ("inferences_per_second_from_cnn_time", "Throughput", "inf/s", "higher"),
            ("estimated_end_to_end_lower_bound_ms", "End-to-end lower bound", "ms", "lower"),
            ("compute_estimate.mops_per_second", "MOPS/s", "MOPS/s", "higher"),
            ("cycles_estimate.mac_ops_per_cycle", "MAC/cycle", "MAC/cycle", "higher"),
        ],
        rows,
        PLOTS / "03_latency_cycles_throughput.svg",
        "Online Latency, Cycles, and Throughput",
        "Raw online metrics from firmware BENCH events.",
        cols=3,
    )
    bar_grid(
        [
            ("memory.text_bytes", "Text", "B", "lower"),
            ("memory.data_bytes", "Data", "B", "lower"),
            ("memory.bss_bytes", "BSS", "B", "lower"),
            ("static_sram_usage.static_sram_bytes", "Static SRAM", "B", "lower"),
            ("elf_size_bytes", "ELF size", "B", "lower"),
            ("memory.dec_bytes", "Firmware dec", "B", "lower"),
        ],
        rows,
        PLOTS / "02_firmware_memory_breakdown.svg",
        "Firmware Memory Comparison",
        "Text/data/BSS/static SRAM/ELF from online result summaries.",
        cols=3,
    )
    bar_grid(
        [
            ("model_info.weights_bytes", "AI weights", "B", "lower"),
            ("model_info.activations_bytes", "AI activations", "B", "lower"),
            ("model_info.input_bytes", "Input buffer", "B", "lower"),
            ("model_info.output_bytes", "Output buffer", "B", "lower"),
            ("compute_estimate.mac_ops_per_inference", "MAC ops", "MAC", "lower"),
            ("compute_estimate.mops_per_second", "MOPS/s", "MOPS/s", "higher"),
        ],
        rows,
        PLOTS / "05_model_core_metrics.svg",
        "AI Model Core Metrics",
        "Firmware model_info values, with v0 backfilled from the matching X-CUBE-AI report.",
        cols=3,
    )
    bar_grid(
        [
            ("energy_estimate.current_ma", "Current", "mA", "lower"),
            ("energy_estimate.power_w", "Power", "W", "lower"),
            ("energy_estimate.energy_per_inference_uj", "Energy/inference", "uJ", "lower"),
            ("energy_estimate.mac_ops_per_joule", "MAC/J", "MAC/J", "higher"),
            ("energy_estimate.mac_ops_per_second_per_watt", "MAC/s/W", "MAC/s/W", "higher"),
            ("compute_estimate.mac_ops_per_second", "MAC/s", "MAC/s", "higher"),
        ],
        rows,
        PLOTS / "04_energy_power_efficiency.svg",
        "Online Energy and Efficiency Estimates",
        "Energy uses current values stored in the online summary files.",
        cols=3,
    )
    bar_grid(
        [
            ("cnn_latency_ms_avg", "CNN latency avg", "ms", "lower"),
            ("static_sram_usage.static_sram_bytes", "Static SRAM", "B", "lower"),
            ("memory.text_bytes", "Text", "B", "lower"),
            ("memory.bss_bytes", "BSS", "B", "lower"),
            ("energy_estimate.energy_per_inference_uj", "Energy/inference", "uJ", "lower"),
            ("inferences_per_second_from_cnn_time", "Throughput", "inf/s", "higher"),
        ],
        rows,
        PLOTS / "06_compact_tradeoff_bars.svg",
        "Compact Online Tradeoff Bars",
        "The most important deployment-facing metrics in one view.",
        cols=3,
    )
    heatmap(rows, all_keys)
    log_metric_overview(rows)
    frontier(rows)
    improvement_plots(compute_delta_rows(rows))
    power_analysis_plots_and_tables()
    write_dashboard()

    print(f"wrote results to {OUT}")


if __name__ == "__main__":
    main()
