#!/usr/bin/env python3
import argparse
import csv
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
from xml.sax.saxutils import escape

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


@dataclass
class PeakMetrics:
    peak_number: int
    window_start_ms: float
    window_end_ms: float
    peak_time_ms: float
    peak_current_ua: float
    baseline_current_ua: float
    threshold_current_ua: float
    start_ms: float
    end_ms: float
    duration_ms: float
    mean_current_ua: float
    mean_excess_current_ua: float
    charge_total_uc: float
    charge_uc: float
    energy_total_uj: Optional[float]
    energy_uj: Optional[float]


def percentile(values, q):
    if not values:
        raise ValueError("empty values")
    s = sorted(values)
    idx = max(0, min(len(s) - 1, int(q * (len(s) - 1))))
    return s[idx]


def read_power_csv(path: Path):
    times_ms = []
    currents_ua = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            times_ms.append(float(row["Timestamp(ms)"]))
            currents_ua.append(float(row["Current(uA)"]))
    if len(times_ms) < 2:
        raise RuntimeError(f"Not enough rows in {path}")
    return times_ms, currents_ua


def nearest_index(times_ms, target_ms):
    lo = 0
    hi = len(times_ms) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if times_ms[mid] < target_ms:
            lo = mid + 1
        else:
            hi = mid
    if lo > 0 and abs(times_ms[lo - 1] - target_ms) < abs(times_ms[lo] - target_ms):
        return lo - 1
    return lo


def find_period_peaks(times_ms, currents_ua, expected_peaks, period_ms):
    peaks = []
    total_start = times_ms[0]
    for i in range(expected_peaks):
        w0 = total_start + i * period_ms
        w1 = w0 + period_ms
        i0 = nearest_index(times_ms, w0)
        i1 = min(len(times_ms) - 1, nearest_index(times_ms, w1))
        if i1 <= i0:
            continue
        local_max_idx = max(range(i0, i1), key=lambda idx: currents_ua[idx])
        peaks.append((i + 1, w0, w1, local_max_idx))
    return peaks


def find_target_peaks(times_ms, currents_ua, peak_times_ms, search_half_window_ms):
    peaks = []
    if not peak_times_ms:
        return peaks

    ordered_targets = sorted(float(v) for v in peak_times_ms)
    total_start = times_ms[0]
    total_end = times_ms[-1]

    for i, target_ms in enumerate(ordered_targets):
        if i == 0:
            w0 = total_start
        else:
            w0 = 0.5 * (ordered_targets[i - 1] + target_ms)

        if i == len(ordered_targets) - 1:
            w1 = total_end
        else:
            w1 = 0.5 * (target_ms + ordered_targets[i + 1])

        search_lo = max(w0, target_ms - search_half_window_ms)
        search_hi = min(w1, target_ms + search_half_window_ms)
        i0 = nearest_index(times_ms, search_lo)
        i1 = min(len(times_ms) - 1, nearest_index(times_ms, search_hi))
        if i1 <= i0:
            continue
        local_max_idx = max(range(i0, i1 + 1), key=lambda idx: currents_ua[idx])
        peaks.append((i + 1, w0, w1, local_max_idx))

    return peaks


def find_window_start_peaks(times_ms, currents_ua, window_start_times_ms, period_ms, expected_duration_ms):
    peaks = []
    if not window_start_times_ms:
        return peaks

    ordered_starts = sorted(float(v) for v in window_start_times_ms)
    total_end = times_ms[-1]
    search_span_ms = expected_duration_ms if expected_duration_ms is not None else period_ms

    for i, start_ms in enumerate(ordered_starts):
        w0 = max(times_ms[0], start_ms)
        if i + 1 < len(ordered_starts):
            w1 = min(total_end, ordered_starts[i + 1])
        else:
            w1 = min(total_end, start_ms + period_ms)

        search_hi = min(w1, start_ms + search_span_ms)
        i0 = nearest_index(times_ms, w0)
        i1 = min(len(times_ms) - 1, nearest_index(times_ms, search_hi))
        if i1 <= i0:
            continue
        local_max_idx = max(range(i0, i1 + 1), key=lambda idx: currents_ua[idx])
        peaks.append((i + 1, w0, w1, local_max_idx))

    return peaks


def moving_average(values, radius):
    if radius <= 0:
        return list(values)
    prefix = [0.0]
    for v in values:
        prefix.append(prefix[-1] + v)
    out = []
    n = len(values)
    for i in range(n):
        lo = max(0, i - radius)
        hi = min(n, i + radius + 1)
        out.append((prefix[hi] - prefix[lo]) / (hi - lo))
    return out


def compute_positive_peak_metrics(
    times_ms,
    currents_ua,
    peak_specs,
    threshold_fraction,
    voltage_v,
    local_baseline_window_ms,
    local_baseline_guard_ms,
):
    dt_ms = times_ms[1] - times_ms[0]
    metrics = []

    for peak_number, w0, w1, peak_idx in peak_specs:
        i0 = nearest_index(times_ms, w0)
        i1 = min(len(times_ms) - 1, nearest_index(times_ms, w1))
        peak_time_ms = times_ms[peak_idx]
        peak_current_ua = currents_ua[peak_idx]

        exclude_ms = 2.0
        ex0 = max(i0, nearest_index(times_ms, peak_time_ms - exclude_ms))
        ex1 = min(i1, nearest_index(times_ms, peak_time_ms + exclude_ms))
        baseline_samples = currents_ua[i0:ex0] + currents_ua[ex1:i1]
        if not baseline_samples:
            baseline_samples = currents_ua[i0:i1]
        global_baseline_ua = percentile(baseline_samples, 0.2)

        pre0 = max(i0, nearest_index(times_ms, peak_time_ms - local_baseline_window_ms))
        pre1 = max(pre0 + 1, min(peak_idx, nearest_index(times_ms, peak_time_ms - local_baseline_guard_ms)))
        local_pre_samples = currents_ua[pre0:pre1]
        local_pre_baseline_ua = percentile(local_pre_samples, 0.5) if local_pre_samples else global_baseline_ua

        baseline_ua = max(global_baseline_ua, local_pre_baseline_ua)
        threshold_ua = baseline_ua + threshold_fraction * (peak_current_ua - baseline_ua)

        start_idx = peak_idx
        while start_idx > i0 and currents_ua[start_idx] > threshold_ua:
            start_idx -= 1
        if start_idx < peak_idx:
            start_idx += 1

        end_idx = peak_idx
        while end_idx < i1 - 1 and currents_ua[end_idx] > threshold_ua:
            end_idx += 1
        if end_idx > peak_idx:
            end_idx -= 1

        peak_currents = currents_ua[start_idx:end_idx + 1]
        mean_current_ua = sum(peak_currents) / len(peak_currents)
        mean_excess_ua = sum((v - baseline_ua) for v in peak_currents) / len(peak_currents)
        duration_ms = times_ms[end_idx] - times_ms[start_idx] + dt_ms
        charge_total_uc = sum(v * dt_ms / 1000.0 for v in peak_currents)
        charge_uc = sum((v - baseline_ua) * dt_ms / 1000.0 for v in peak_currents)
        energy_total_uj = None if voltage_v is None else charge_total_uc * voltage_v
        energy_uj = None if voltage_v is None else charge_uc * voltage_v

        metrics.append(
            PeakMetrics(
                peak_number=peak_number,
                window_start_ms=w0,
                window_end_ms=w1,
                peak_time_ms=peak_time_ms,
                peak_current_ua=peak_current_ua,
                baseline_current_ua=baseline_ua,
                threshold_current_ua=threshold_ua,
                start_ms=times_ms[start_idx],
                end_ms=times_ms[end_idx],
                duration_ms=duration_ms,
                mean_current_ua=mean_current_ua,
                mean_excess_current_ua=mean_excess_ua,
                charge_total_uc=charge_total_uc,
                charge_uc=charge_uc,
                energy_total_uj=energy_total_uj,
                energy_uj=energy_uj,
            )
        )

    return metrics


def compute_transition_window_metrics(
    times_ms,
    currents_ua,
    peak_specs,
    voltage_v,
    start_fraction,
    end_fraction,
    local_baseline_window_ms,
    local_baseline_guard_ms,
    settle_ms,
    smooth_ms,
    expected_duration_ms,
):
    dt_ms = times_ms[1] - times_ms[0]
    smooth_radius = max(1, int(round((smooth_ms / dt_ms) / 2.0)))
    settle_samples = max(1, int(round(settle_ms / dt_ms)))
    metrics = []

    for peak_number, w0, w1, peak_idx in peak_specs:
        i0 = nearest_index(times_ms, w0)
        i1 = min(len(times_ms) - 1, nearest_index(times_ms, w1))
        segment_times = times_ms[i0:i1]
        segment_currents = currents_ua[i0:i1]
        local_peak_idx = max(0, peak_idx - i0)
        peak_time_ms = times_ms[peak_idx]
        peak_current_ua = currents_ua[peak_idx]

        pre0 = max(i0, nearest_index(times_ms, peak_time_ms - local_baseline_window_ms))
        pre1 = max(pre0 + 1, min(peak_idx, nearest_index(times_ms, peak_time_ms - local_baseline_guard_ms)))
        post0 = min(i1 - 1, max(peak_idx + 1, nearest_index(times_ms, peak_time_ms + local_baseline_guard_ms)))
        post1 = min(i1, nearest_index(times_ms, peak_time_ms + local_baseline_window_ms))
        if post1 <= post0:
            post1 = i1

        pre_samples = currents_ua[pre0:pre1]
        post_samples = currents_ua[post0:post1]
        pre_level = percentile(pre_samples, 0.5) if pre_samples else percentile(segment_currents, 0.8)
        post_level = percentile(post_samples, 0.5) if post_samples else percentile(segment_currents, 0.2)

        if pre_samples and pre_level <= post_level:
            boosted_pre_level = percentile(pre_samples, 0.8)
            if boosted_pre_level > post_level:
                pre_level = boosted_pre_level

        if pre_level <= post_level:
            fallback = compute_positive_peak_metrics(
                times_ms,
                currents_ua,
                [(peak_number, w0, w1, peak_idx)],
                0.25,
                voltage_v,
                local_baseline_window_ms,
                local_baseline_guard_ms,
            )[0]
            metrics.append(fallback)
            continue

        delta = pre_level - post_level
        start_threshold = pre_level - start_fraction * delta
        end_threshold = post_level + end_fraction * delta
        smoothed = moving_average(segment_currents, smooth_radius)

        search_back_samples = max(1, int(round(local_baseline_window_ms / dt_ms)))
        start_search_lo = max(0, local_peak_idx - search_back_samples)
        candidate_starts = []
        prev_below = smoothed[start_search_lo] <= start_threshold
        for j in range(start_search_lo + 1, local_peak_idx + 1):
            cur_above = smoothed[j] >= start_threshold
            if cur_above and prev_below:
                candidate_starts.append(j)
            prev_below = not cur_above
        if not candidate_starts:
            if expected_duration_ms is not None:
                candidate_starts = [max(start_search_lo, local_peak_idx - int(round(expected_duration_ms / dt_ms)))]
            else:
                candidate_starts = [max(start_search_lo, local_peak_idx - search_back_samples // 2)]

        end_local = local_peak_idx
        found_end = False
        while end_local < len(smoothed) - settle_samples:
            if all(v <= end_threshold for v in smoothed[end_local:end_local + settle_samples]):
                found_end = True
                break
            end_local += 1
        if not found_end:
            end_local = len(smoothed) - 1

        if expected_duration_ms is not None:
            start_local = min(
                candidate_starts,
                key=lambda j: abs(((segment_times[end_local] - segment_times[j]) + dt_ms) - expected_duration_ms),
            )
        else:
            start_local = candidate_starts[0]

        if expected_duration_ms is not None:
            detected_duration_ms = (segment_times[end_local] - segment_times[start_local]) + dt_ms
            if detected_duration_ms < (0.85 * expected_duration_ms):
                fallback_span = max(1, int(round(expected_duration_ms / dt_ms)))
                start_local = max(0, end_local - fallback_span + 1)

        start_idx = i0 + start_local
        end_idx = i0 + end_local
        if end_idx <= start_idx:
            end_idx = min(i1 - 1, start_idx + 1)

        event_currents = currents_ua[start_idx:end_idx + 1]
        mean_current_ua = sum(event_currents) / len(event_currents)
        mean_excess_ua = sum((v - post_level) for v in event_currents) / len(event_currents)
        duration_ms = times_ms[end_idx] - times_ms[start_idx] + dt_ms
        charge_total_uc = sum(v * dt_ms / 1000.0 for v in event_currents)
        charge_uc = sum((v - post_level) * dt_ms / 1000.0 for v in event_currents)
        energy_total_uj = None if voltage_v is None else charge_total_uc * voltage_v
        energy_uj = None if voltage_v is None else charge_uc * voltage_v

        metrics.append(
            PeakMetrics(
                peak_number=peak_number,
                window_start_ms=w0,
                window_end_ms=w1,
                peak_time_ms=peak_time_ms,
                peak_current_ua=peak_current_ua,
                baseline_current_ua=post_level,
                threshold_current_ua=end_threshold,
                start_ms=times_ms[start_idx],
                end_ms=times_ms[end_idx],
                duration_ms=duration_ms,
                mean_current_ua=mean_current_ua,
                mean_excess_current_ua=mean_excess_ua,
                charge_total_uc=charge_total_uc,
                charge_uc=charge_uc,
                energy_total_uj=energy_total_uj,
                energy_uj=energy_uj,
            )
        )

    return metrics


def compute_plateau_window_metrics(
    times_ms,
    currents_ua,
    peak_specs,
    threshold_fraction,
    voltage_v,
    local_baseline_window_ms,
    local_baseline_guard_ms,
    transition_settle_ms,
    transition_smooth_ms,
    plateau_coarse_fraction,
):
    dt_ms = times_ms[1] - times_ms[0]
    smooth_radius = max(1, int(round((transition_smooth_ms / dt_ms) / 2.0)))
    settle_samples = max(1, int(round(transition_settle_ms / dt_ms)))
    metrics = []

    smoothed = moving_average(currents_ua, smooth_radius)

    for peak_number, w0, w1, peak_idx in peak_specs:
        i0 = nearest_index(times_ms, w0)
        i1 = min(len(times_ms) - 1, nearest_index(times_ms, w1))
        peak_time_ms = times_ms[peak_idx]
        peak_current_ua = currents_ua[peak_idx]

        pre0 = max(i0, nearest_index(times_ms, peak_time_ms - local_baseline_window_ms))
        pre1 = max(pre0 + 1, min(peak_idx, nearest_index(times_ms, peak_time_ms - local_baseline_guard_ms)))
        pre_samples = currents_ua[pre0:pre1]
        baseline_ua = percentile(pre_samples, 0.5) if pre_samples else percentile(currents_ua[i0:i1], 0.2)

        coarse_threshold_ua = baseline_ua + plateau_coarse_fraction * (peak_current_ua - baseline_ua)

        coarse_start_idx = peak_idx
        while coarse_start_idx > i0 and currents_ua[coarse_start_idx] >= coarse_threshold_ua:
            coarse_start_idx -= 1
        if coarse_start_idx < peak_idx:
            coarse_start_idx += 1

        coarse_end_idx = peak_idx
        while coarse_end_idx < i1 - 1 and currents_ua[coarse_end_idx] >= coarse_threshold_ua:
            coarse_end_idx += 1
        if coarse_end_idx > peak_idx:
            coarse_end_idx -= 1

        plateau_samples = currents_ua[coarse_start_idx:coarse_end_idx + 1]
        plateau_level_ua = percentile(plateau_samples, 0.5) if plateau_samples else peak_current_ua
        threshold_ua = baseline_ua + threshold_fraction * (plateau_level_ua - baseline_ua)

        search_back_samples = max(1, int(round(local_baseline_window_ms / dt_ms)))
        start_search_lo = max(i0, coarse_start_idx - search_back_samples)
        start_idx = coarse_start_idx
        for idx in range(start_search_lo, coarse_start_idx + 1):
            if smoothed[idx] >= threshold_ua:
                start_idx = idx
                break

        end_search_hi = min(i1 - 1, coarse_end_idx + search_back_samples)
        end_idx = coarse_end_idx
        for idx in range(coarse_end_idx, max(coarse_end_idx + 1, end_search_hi - settle_samples + 1)):
            if all(v < threshold_ua for v in smoothed[idx:idx + settle_samples]):
                end_idx = idx
                break

        event_currents = currents_ua[start_idx:end_idx + 1]
        mean_current_ua = sum(event_currents) / len(event_currents)
        mean_excess_ua = sum((v - baseline_ua) for v in event_currents) / len(event_currents)
        duration_ms = times_ms[end_idx] - times_ms[start_idx] + dt_ms
        charge_total_uc = sum(v * dt_ms / 1000.0 for v in event_currents)
        charge_uc = sum((v - baseline_ua) * dt_ms / 1000.0 for v in event_currents)
        energy_total_uj = None if voltage_v is None else charge_total_uc * voltage_v
        energy_uj = None if voltage_v is None else charge_uc * voltage_v

        metrics.append(
            PeakMetrics(
                peak_number=peak_number,
                window_start_ms=w0,
                window_end_ms=w1,
                peak_time_ms=peak_time_ms,
                peak_current_ua=peak_current_ua,
                baseline_current_ua=baseline_ua,
                threshold_current_ua=threshold_ua,
                start_ms=times_ms[start_idx],
                end_ms=times_ms[end_idx],
                duration_ms=duration_ms,
                mean_current_ua=mean_current_ua,
                mean_excess_current_ua=mean_excess_ua,
                charge_total_uc=charge_total_uc,
                charge_uc=charge_uc,
                energy_total_uj=energy_total_uj,
                energy_uj=energy_uj,
            )
        )

    return metrics


def compute_fixed_duration_to_end_metrics(
    times_ms,
    currents_ua,
    peak_specs,
    voltage_v,
    local_baseline_window_ms,
    local_baseline_guard_ms,
    transition_start_fraction,
    transition_end_fraction,
    transition_settle_ms,
    transition_smooth_ms,
    expected_duration_ms,
):
    if expected_duration_ms is None:
        return compute_transition_window_metrics(
            times_ms,
            currents_ua,
            peak_specs,
            voltage_v,
            transition_start_fraction,
            transition_end_fraction,
            local_baseline_window_ms,
            local_baseline_guard_ms,
            transition_settle_ms,
            transition_smooth_ms,
            expected_duration_ms,
        )

    dt_ms = times_ms[1] - times_ms[0]
    duration_samples = max(1, int(round(expected_duration_ms / dt_ms)))
    transition_metrics = compute_transition_window_metrics(
        times_ms,
        currents_ua,
        peak_specs,
        voltage_v,
        transition_start_fraction,
        transition_end_fraction,
        local_baseline_window_ms,
        local_baseline_guard_ms,
        transition_settle_ms,
        transition_smooth_ms,
        expected_duration_ms,
    )
    metrics = []
    for spec, transition_metric in zip(peak_specs, transition_metrics):
        peak_number, w0, w1, _ = spec
        i0 = nearest_index(times_ms, w0)
        i1 = min(len(times_ms) - 1, nearest_index(times_ms, w1))
        end_idx = nearest_index(times_ms, transition_metric.end_ms)
        start_idx = max(i0, end_idx - duration_samples + 1)
        if end_idx <= start_idx:
            end_idx = min(i1 - 1, start_idx + 1)

        event_currents = currents_ua[start_idx:end_idx + 1]
        baseline_ua = transition_metric.baseline_current_ua
        mean_current_ua = sum(event_currents) / len(event_currents)
        mean_excess_ua = sum((v - baseline_ua) for v in event_currents) / len(event_currents)
        duration_ms = times_ms[end_idx] - times_ms[start_idx] + dt_ms
        charge_total_uc = sum(v * dt_ms / 1000.0 for v in event_currents)
        charge_uc = sum((v - baseline_ua) * dt_ms / 1000.0 for v in event_currents)
        energy_total_uj = None if voltage_v is None else charge_total_uc * voltage_v
        energy_uj = None if voltage_v is None else charge_uc * voltage_v

        metrics.append(
            PeakMetrics(
                peak_number=peak_number,
                window_start_ms=w0,
                window_end_ms=w1,
                peak_time_ms=transition_metric.peak_time_ms,
                peak_current_ua=transition_metric.peak_current_ua,
                baseline_current_ua=baseline_ua,
                threshold_current_ua=transition_metric.threshold_current_ua,
                start_ms=times_ms[start_idx],
                end_ms=times_ms[end_idx],
                duration_ms=duration_ms,
                mean_current_ua=mean_current_ua,
                mean_excess_current_ua=mean_excess_ua,
                charge_total_uc=charge_total_uc,
                charge_uc=charge_uc,
                energy_total_uj=energy_total_uj,
                energy_uj=energy_uj,
            )
        )
    return metrics


def compute_fixed_duration_from_start_metrics(
    times_ms,
    currents_ua,
    peak_specs,
    voltage_v,
    local_baseline_window_ms,
    local_baseline_guard_ms,
    transition_start_fraction,
    transition_end_fraction,
    transition_settle_ms,
    transition_smooth_ms,
    expected_duration_ms,
    threshold_fraction,
    plateau_coarse_fraction,
):
    """Detect the rising edge (start), then measure exactly expected_duration_ms forward.
    This correctly handles cases where the end of the power-active period includes
    overhead (e.g. UART printing) that should be excluded from the measured window."""
    if expected_duration_ms is None:
        return compute_plateau_window_metrics(
            times_ms, currents_ua, peak_specs, threshold_fraction, voltage_v,
            local_baseline_window_ms, local_baseline_guard_ms,
            transition_settle_ms, transition_smooth_ms, plateau_coarse_fraction,
        )

    dt_ms = times_ms[1] - times_ms[0]
    duration_samples = max(1, int(round(expected_duration_ms / dt_ms)))
    plateau_metrics = compute_plateau_window_metrics(
        times_ms, currents_ua, peak_specs, threshold_fraction, voltage_v,
        local_baseline_window_ms, local_baseline_guard_ms,
        transition_settle_ms, transition_smooth_ms, plateau_coarse_fraction,
    )
    metrics = []
    for spec, pm in zip(peak_specs, plateau_metrics):
        peak_number, w0, w1, _ = spec
        i0 = nearest_index(times_ms, w0)
        i1 = min(len(times_ms) - 1, nearest_index(times_ms, w1))
        start_idx = nearest_index(times_ms, pm.start_ms)
        end_idx = min(i1 - 1, start_idx + duration_samples - 1)
        if end_idx <= start_idx:
            end_idx = min(i1 - 1, start_idx + 1)

        event_currents = currents_ua[start_idx:end_idx + 1]
        baseline_ua = pm.baseline_current_ua
        mean_current_ua = sum(event_currents) / len(event_currents)
        mean_excess_ua = sum((v - baseline_ua) for v in event_currents) / len(event_currents)
        duration_ms = times_ms[end_idx] - times_ms[start_idx] + dt_ms
        charge_total_uc = sum(v * dt_ms / 1000.0 for v in event_currents)
        charge_uc = sum((v - baseline_ua) * dt_ms / 1000.0 for v in event_currents)
        energy_total_uj = None if voltage_v is None else charge_total_uc * voltage_v
        energy_uj = None if voltage_v is None else charge_uc * voltage_v

        metrics.append(
            PeakMetrics(
                peak_number=peak_number,
                window_start_ms=w0,
                window_end_ms=w1,
                peak_time_ms=pm.peak_time_ms,
                peak_current_ua=pm.peak_current_ua,
                baseline_current_ua=baseline_ua,
                threshold_current_ua=pm.threshold_current_ua,
                start_ms=times_ms[start_idx],
                end_ms=times_ms[end_idx],
                duration_ms=duration_ms,
                mean_current_ua=mean_current_ua,
                mean_excess_current_ua=mean_excess_ua,
                charge_total_uc=charge_total_uc,
                charge_uc=charge_uc,
                energy_total_uj=energy_total_uj,
                energy_uj=energy_uj,
            )
        )
    return metrics


def compute_fixed_duration_from_window_start_metrics(
    times_ms,
    currents_ua,
    peak_specs,
    voltage_v,
    expected_duration_ms,
):
    if expected_duration_ms is None:
        raise RuntimeError("window_mode=fixed_duration_from_window_start requires expected_duration_ms")

    dt_ms = times_ms[1] - times_ms[0]
    duration_samples = max(1, int(round(expected_duration_ms / dt_ms)))
    metrics = []

    for peak_number, w0, w1, _ in peak_specs:
        i0 = nearest_index(times_ms, w0)
        i1 = min(len(times_ms) - 1, nearest_index(times_ms, w1))
        end_idx = min(i1 - 1, i0 + duration_samples - 1)
        if end_idx <= i0:
            end_idx = min(i1 - 1, i0 + 1)

        event_currents = currents_ua[i0:end_idx + 1]
        event_peak_rel_idx = max(range(len(event_currents)), key=lambda idx: event_currents[idx])
        peak_idx = i0 + event_peak_rel_idx

        post0 = min(i1 - 1, end_idx + 1)
        post_samples = currents_ua[post0:i1] if post0 < i1 else []
        pre_samples = currents_ua[max(0, i0 - duration_samples):i0]
        baseline_samples = post_samples or pre_samples or currents_ua[i0:i1]
        baseline_ua = percentile(baseline_samples, 0.2)
        peak_current_ua = currents_ua[peak_idx]
        threshold_ua = baseline_ua

        mean_current_ua = sum(event_currents) / len(event_currents)
        mean_excess_ua = sum((v - baseline_ua) for v in event_currents) / len(event_currents)
        duration_ms = times_ms[end_idx] - times_ms[i0] + dt_ms
        charge_total_uc = sum(v * dt_ms / 1000.0 for v in event_currents)
        charge_uc = sum((v - baseline_ua) * dt_ms / 1000.0 for v in event_currents)
        energy_total_uj = None if voltage_v is None else charge_total_uc * voltage_v
        energy_uj = None if voltage_v is None else charge_uc * voltage_v

        metrics.append(
            PeakMetrics(
                peak_number=peak_number,
                window_start_ms=w0,
                window_end_ms=w1,
                peak_time_ms=times_ms[peak_idx],
                peak_current_ua=peak_current_ua,
                baseline_current_ua=baseline_ua,
                threshold_current_ua=threshold_ua,
                start_ms=times_ms[i0],
                end_ms=times_ms[end_idx],
                duration_ms=duration_ms,
                mean_current_ua=mean_current_ua,
                mean_excess_current_ua=mean_excess_ua,
                charge_total_uc=charge_total_uc,
                charge_uc=charge_uc,
                energy_total_uj=energy_total_uj,
                energy_uj=energy_uj,
            )
        )

    return metrics


def compute_peak_metrics(
    times_ms,
    currents_ua,
    peak_specs,
    window_mode,
    threshold_fraction,
    voltage_v,
    local_baseline_window_ms,
    local_baseline_guard_ms,
    transition_start_fraction,
    transition_end_fraction,
    transition_settle_ms,
    transition_smooth_ms,
    expected_duration_ms,
    plateau_coarse_fraction,
):
    if window_mode == "plateau_window":
        return compute_plateau_window_metrics(
            times_ms,
            currents_ua,
            peak_specs,
            threshold_fraction,
            voltage_v,
            local_baseline_window_ms,
            local_baseline_guard_ms,
            transition_settle_ms,
            transition_smooth_ms,
            plateau_coarse_fraction,
        )
    if window_mode == "transition_window":
        return compute_transition_window_metrics(
            times_ms,
            currents_ua,
            peak_specs,
            voltage_v,
            transition_start_fraction,
            transition_end_fraction,
            local_baseline_window_ms,
            local_baseline_guard_ms,
            transition_settle_ms,
            transition_smooth_ms,
            expected_duration_ms,
        )
    if window_mode == "fixed_duration_to_end":
        return compute_fixed_duration_to_end_metrics(
            times_ms,
            currents_ua,
            peak_specs,
            voltage_v,
            local_baseline_window_ms,
            local_baseline_guard_ms,
            transition_start_fraction,
            transition_end_fraction,
            transition_settle_ms,
            transition_smooth_ms,
            expected_duration_ms,
        )
    if window_mode == "fixed_duration_from_start":
        return compute_fixed_duration_from_start_metrics(
            times_ms,
            currents_ua,
            peak_specs,
            voltage_v,
            local_baseline_window_ms,
            local_baseline_guard_ms,
            transition_start_fraction,
            transition_end_fraction,
            transition_settle_ms,
            transition_smooth_ms,
            expected_duration_ms,
            threshold_fraction,
            plateau_coarse_fraction,
        )
    if window_mode == "fixed_duration_from_window_start":
        return compute_fixed_duration_from_window_start_metrics(
            times_ms,
            currents_ua,
            peak_specs,
            voltage_v,
            expected_duration_ms,
        )
    return compute_positive_peak_metrics(
        times_ms,
        currents_ua,
        peak_specs,
        threshold_fraction,
        voltage_v,
        local_baseline_window_ms,
        local_baseline_guard_ms,
    )


def write_peak_csv(path: Path, metrics):
    rows = [asdict(m) for m in metrics]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_zoom_csvs(out_dir: Path, times_ms, currents_ua, metrics, zoom_half_window_ms):
    for m in metrics:
        duration = max(0.0, m.end_ms - m.start_ms)
        margin = max(zoom_half_window_ms * 0.25, duration * 0.15, 1.0)
        z0 = min(m.peak_time_ms - zoom_half_window_ms, m.start_ms - margin)
        z1 = max(m.peak_time_ms + zoom_half_window_ms, m.end_ms + margin)
        i0 = nearest_index(times_ms, z0)
        i1 = min(len(times_ms) - 1, nearest_index(times_ms, z1))
        out_path = out_dir / f"peak_{m.peak_number:02d}_zoom.csv"
        with out_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time_ms", "time_relative_ms", "current_ua"])
            for idx in range(i0, i1 + 1):
                writer.writerow([times_ms[idx], times_ms[idx] - m.peak_time_ms, currents_ua[idx]])


def cleanup_old_peak_outputs(out_dir: Path):
    for pattern in (
        "peak_*_zoom.csv",
        "peak_*_zoom.svg",
        "peak_*_zoom.png",
    ):
        for path in out_dir.glob(pattern):
            path.unlink(missing_ok=True)
    if plt is None:
        for name in (
            "full_trace.png",
            "peak_overlay.png",
            "peak_zoom_grid.png",
            "peak_zoom.png",
        ):
            (out_dir / name).unlink(missing_ok=True)


def downsample_pairs(xs, ys, max_points):
    if len(xs) <= max_points:
        return xs, ys
    step = max(1, len(xs) // max_points)
    xs2 = xs[::step]
    ys2 = ys[::step]
    if xs2[-1] != xs[-1]:
        xs2.append(xs[-1])
        ys2.append(ys[-1])
    return xs2, ys2


def svg_polyline_points(xs, ys, x_min, x_max, y_min, y_max, left, top, plot_w, plot_h):
    pts = []
    x_span = max(1e-12, x_max - x_min)
    y_span = max(1e-12, y_max - y_min)
    for x, y in zip(xs, ys):
        px = left + (x - x_min) / x_span * plot_w
        py = top + plot_h - (y - y_min) / y_span * plot_h
        pts.append(f"{px:.2f},{py:.2f}")
    return " ".join(pts)


def save_svg_multiline(path: Path, series, title, xlabel, ylabel, width=1400, height=500,
                       vspans=None, vlines=None, hlines=None, legend=True):
    left = 85
    right = 30
    top = 55
    bottom = 60
    plot_w = width - left - right
    plot_h = height - top - bottom

    all_x = [x for s in series for x in s["x"]]
    all_y = [y for s in series for y in s["y"]]
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    y_pad = max(1.0, 0.05 * (y_max - y_min if y_max > y_min else y_max if y_max else 1.0))
    y_min -= y_pad
    y_max += y_pad

    def sx(x):
        return left + (x - x_min) / max(1e-12, x_max - x_min) * plot_w

    def sy(y):
        return top + plot_h - (y - y_min) / max(1e-12, y_max - y_min) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2:.1f}" y="28" text-anchor="middle" font-size="24" font-family="sans-serif">{escape(title)}</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#333" stroke-width="1"/>',
    ]

    for i in range(6):
        gx = left + i * plot_w / 5
        parts.append(f'<line x1="{gx:.2f}" y1="{top}" x2="{gx:.2f}" y2="{top + plot_h}" stroke="#ddd" stroke-width="1"/>')
    for i in range(6):
        gy = top + i * plot_h / 5
        parts.append(f'<line x1="{left}" y1="{gy:.2f}" x2="{left + plot_w}" y2="{gy:.2f}" stroke="#ddd" stroke-width="1"/>')

    for i in range(6):
        xv = x_min + i * (x_max - x_min) / 5
        yv = y_min + (5 - i) * (y_max - y_min) / 5
        gx = left + i * plot_w / 5
        gy = top + i * plot_h / 5
        parts.append(f'<text x="{gx:.2f}" y="{top + plot_h + 22}" text-anchor="middle" font-size="14" font-family="sans-serif">{xv:.3f}</text>')
        parts.append(f'<text x="{left - 10}" y="{gy + 5:.2f}" text-anchor="end" font-size="14" font-family="sans-serif">{yv:.1f}</text>')

    if vspans:
        for x0, x1, color, alpha in vspans:
            parts.append(
                f'<rect x="{sx(x0):.2f}" y="{top}" width="{max(1.0, sx(x1)-sx(x0)):.2f}" height="{plot_h}" '
                f'fill="{color}" fill-opacity="{alpha}"/>'
            )
    if hlines:
        for y, color, dash in hlines:
            parts.append(
                f'<line x1="{left}" y1="{sy(y):.2f}" x2="{left + plot_w}" y2="{sy(y):.2f}" '
                f'stroke="{color}" stroke-width="1.2" stroke-dasharray="{dash}"/>'
            )
    if vlines:
        for x, color, dash in vlines:
            parts.append(
                f'<line x1="{sx(x):.2f}" y1="{top}" x2="{sx(x):.2f}" y2="{top + plot_h}" '
                f'stroke="{color}" stroke-width="1.2" stroke-dasharray="{dash}"/>'
            )

    for s in series:
        xs, ys = downsample_pairs(list(s["x"]), list(s["y"]), s.get("max_points", 4000))
        pts = svg_polyline_points(xs, ys, x_min, x_max, y_min, y_max, left, top, plot_w, plot_h)
        parts.append(
            f'<polyline fill="none" stroke="{s.get("color", "#1f77b4")}" stroke-width="{s.get("stroke_width", 1.2)}" '
            f'points="{pts}"/>'
        )

    parts.append(f'<text x="{width/2:.1f}" y="{height - 15}" text-anchor="middle" font-size="18" font-family="sans-serif">{escape(xlabel)}</text>')
    parts.append(
        f'<text x="22" y="{height/2:.1f}" text-anchor="middle" font-size="18" font-family="sans-serif" '
        f'transform="rotate(-90 22,{height/2:.1f})">{escape(ylabel)}</text>'
    )

    if legend:
        lx = width - 220
        ly = 65
        parts.append(f'<rect x="{lx}" y="{ly}" width="180" height="{28 * len(series) + 12}" fill="white" stroke="#666"/>')
        for i, s in enumerate(series):
            yy = ly + 22 + i * 28
            parts.append(f'<line x1="{lx + 12}" y1="{yy}" x2="{lx + 42}" y2="{yy}" stroke="{s.get("color", "#1f77b4")}" stroke-width="2"/>')
            parts.append(f'<text x="{lx + 50}" y="{yy + 5}" font-size="15" font-family="sans-serif">{escape(s.get("label", f"series {i+1}"))}</text>')

    parts.append("</svg>")
    path.write_text("\n".join(parts))


def save_full_plot(path: Path, times_ms, currents_ua, metrics):
    if plt is None:
        save_svg_multiline(
            path.with_suffix(".svg"),
            [{"x": times_ms, "y": currents_ua, "label": "current", "color": "#1f77b4", "max_points": 8000, "stroke_width": 0.8}],
            "Power trace with detected inference peaks",
            "Time (ms)",
            "Current (uA)",
            vspans=[(m.start_ms, m.end_ms, "#d62728", 0.14) for m in metrics],
            vlines=[(m.peak_time_ms, "#d62728", "6,4") for m in metrics],
            legend=False,
        )
        return
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(times_ms, currents_ua, linewidth=0.6, color="tab:blue")
    for m in metrics:
        ax.axvspan(m.start_ms, m.end_ms, color="tab:red", alpha=0.15)
        ax.axvline(m.peak_time_ms, color="tab:red", linestyle="--", linewidth=0.8)
    ax.set_title("Power trace with detected inference peaks")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Current (uA)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_zoom_grid(path: Path, times_ms, currents_ua, metrics, zoom_half_window_ms):
    if plt is None:
        overlay_series = []
        for m in metrics:
            duration = max(0.0, m.end_ms - m.start_ms)
            margin = max(zoom_half_window_ms * 0.25, duration * 0.15, 1.0)
            z0 = min(m.peak_time_ms - zoom_half_window_ms, m.start_ms - margin)
            z1 = max(m.peak_time_ms + zoom_half_window_ms, m.end_ms + margin)
            i0 = nearest_index(times_ms, z0)
            i1 = min(len(times_ms) - 1, nearest_index(times_ms, z1))
            overlay_series.append(
                {
                    "x": [t - m.peak_time_ms for t in times_ms[i0:i1 + 1]],
                    "y": currents_ua[i0:i1 + 1],
                    "label": f"peak {m.peak_number}",
                    "color": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"][(m.peak_number - 1) % 6],
                    "max_points": 2500,
                    "stroke_width": 1.0,
                }
            )
            save_svg_multiline(
                path.parent / f"peak_{m.peak_number:02d}_zoom.svg",
                [{"x": times_ms[i0:i1 + 1], "y": currents_ua[i0:i1 + 1], "label": f"peak {m.peak_number}", "color": "#1f77b4", "max_points": 6000, "stroke_width": 1.0}],
                f"Peak {m.peak_number}: duration={m.duration_ms:.3f} ms, mean={m.mean_current_ua:.1f} uA, max={m.peak_current_ua:.1f} uA",
                "Time (ms)",
                "Current (uA)",
                vspans=[(m.start_ms, m.end_ms, "#d62728", 0.18)],
                vlines=[(m.peak_time_ms, "#d62728", "6,4")],
                hlines=[
                    (m.baseline_current_ua, "#2ca02c", "6,4"),
                    (m.threshold_current_ua, "#ff7f0e", "3,3"),
                ],
                legend=False,
            )
        if overlay_series:
            save_svg_multiline(
                path.with_suffix(".svg"),
                overlay_series,
                "Peak zoom windows",
                "Time relative to peak max (ms)",
                "Current (uA)",
                legend=True,
            )
        return
    n = len(metrics)
    cols = 2
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.8 * rows), squeeze=False)

    for ax in axes.flat[n:]:
        ax.axis("off")

    for ax, m in zip(axes.flat, metrics):
        duration = max(0.0, m.end_ms - m.start_ms)
        margin = max(zoom_half_window_ms * 0.25, duration * 0.15, 1.0)
        z0 = min(m.peak_time_ms - zoom_half_window_ms, m.start_ms - margin)
        z1 = max(m.peak_time_ms + zoom_half_window_ms, m.end_ms + margin)
        i0 = nearest_index(times_ms, z0)
        i1 = min(len(times_ms) - 1, nearest_index(times_ms, z1))
        xt = times_ms[i0:i1 + 1]
        yt = currents_ua[i0:i1 + 1]

        ax.plot(xt, yt, linewidth=0.8, color="tab:blue")
        ax.axhline(m.baseline_current_ua, color="tab:green", linestyle="--", linewidth=0.8, label="baseline")
        ax.axhline(m.threshold_current_ua, color="tab:orange", linestyle=":", linewidth=0.8, label="threshold")
        ax.axvspan(m.start_ms, m.end_ms, color="tab:red", alpha=0.18, label="detected peak")
        ax.axvline(m.peak_time_ms, color="tab:red", linestyle="--", linewidth=0.8)
        ax.set_title(
            f"Peak {m.peak_number}: duration={m.duration_ms:.3f} ms, "
            f"mean={m.mean_current_ua:.1f} uA, max={m.peak_current_ua:.1f} uA"
        )
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Current (uA)")
        ax.grid(True, alpha=0.3)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    alt_path = path.with_name("peak_zoom.png")
    if alt_path != path:
        fig.savefig(alt_path, dpi=300)
    plt.close(fig)


def save_overlay_plot(path: Path, times_ms, currents_ua, metrics, overlay_half_window_ms):
    if plt is None:
        series = []
        for m in metrics:
            z0 = m.peak_time_ms - overlay_half_window_ms
            z1 = m.peak_time_ms + overlay_half_window_ms
            i0 = nearest_index(times_ms, z0)
            i1 = min(len(times_ms) - 1, nearest_index(times_ms, z1))
            series.append(
                {
                    "x": [t - m.peak_time_ms for t in times_ms[i0:i1 + 1]],
                    "y": currents_ua[i0:i1 + 1],
                    "label": f"peak {m.peak_number}",
                    "color": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"][(m.peak_number - 1) % 6],
                    "max_points": 2500,
                    "stroke_width": 1.0,
                }
            )
        save_svg_multiline(
            path.with_suffix(".svg"),
            series,
            "Peak overlays aligned by maximum",
            "Time relative to peak max (ms)",
            "Current (uA)",
            legend=True,
        )
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    for m in metrics:
        z0 = m.peak_time_ms - overlay_half_window_ms
        z1 = m.peak_time_ms + overlay_half_window_ms
        i0 = nearest_index(times_ms, z0)
        i1 = min(len(times_ms) - 1, nearest_index(times_ms, z1))
        xt = [t - m.peak_time_ms for t in times_ms[i0:i1 + 1]]
        yt = currents_ua[i0:i1 + 1]
        ax.plot(xt, yt, linewidth=0.8, alpha=0.9, label=f"peak {m.peak_number}")
    ax.set_title("Peak overlays aligned by maximum")
    ax.set_xlabel("Time relative to peak max (ms)")
    ax.set_ylabel("Current (uA)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def load_summary_json(path: Optional[Path]):
    if path is None:
        return None
    return json.loads(path.read_text())


def summary_inference_count(summary):
    if not summary:
        return None
    try:
        value = summary.get("counts", {}).get("num_inference_events")
        if value is None:
            return None
        value = int(value)
        return value if value > 0 else None
    except Exception:
        return None


def load_config(path: Optional[Path]):
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text())


def pick_value(cli_value, config, key, default=None):
    if cli_value is not None:
        return cli_value
    if key in config and config[key] is not None:
        return config[key]
    return default


def main():
    ap = argparse.ArgumentParser()
    default_config_path = Path(__file__).with_name("analyze_power_peaks_config.json")
    ap.add_argument("--config", default=str(default_config_path))
    ap.add_argument("--csv")
    ap.add_argument("--summary-json")
    ap.add_argument("--out-dir")
    ap.add_argument("--expected-peaks", type=int)
    ap.add_argument("--period-ms", type=float)
    ap.add_argument("--threshold-fraction", type=float)
    ap.add_argument("--zoom-half-window-ms", type=float)
    ap.add_argument("--overlay-half-window-ms", type=float)
    ap.add_argument("--voltage", type=float)
    ap.add_argument("--peak-times-ms", nargs="*", type=float)
    ap.add_argument("--window-start-times-ms", nargs="*", type=float)
    ap.add_argument("--peak-search-half-window-ms", type=float)
    ap.add_argument("--local-baseline-window-ms", type=float)
    ap.add_argument("--local-baseline-guard-ms", type=float)
    ap.add_argument("--window-mode")
    ap.add_argument("--transition-start-fraction", type=float)
    ap.add_argument("--transition-end-fraction", type=float)
    ap.add_argument("--transition-settle-ms", type=float)
    ap.add_argument("--transition-smooth-ms", type=float)
    ap.add_argument("--plateau-coarse-fraction", type=float)
    ap.add_argument("--expected-duration-ms", type=float)
    ap.add_argument("--skip-zoom-outputs", action="store_true")
    args = ap.parse_args()
    config_path = Path(args.config).expanduser().resolve() if args.config else None
    config = load_config(config_path)

    csv_value = pick_value(args.csv, config, "csv")
    if not csv_value:
        raise RuntimeError("Missing CSV path. Set it in --csv or in the config file.")

    args.summary_json = pick_value(args.summary_json, config, "summary_json")
    args.out_dir = pick_value(args.out_dir, config, "out_dir")
    args.expected_peaks = int(pick_value(args.expected_peaks, config, "expected_peaks", 6))
    args.period_ms = float(pick_value(args.period_ms, config, "period_ms", 5000.0))
    args.threshold_fraction = float(pick_value(args.threshold_fraction, config, "threshold_fraction", 0.25))
    args.zoom_half_window_ms = float(pick_value(args.zoom_half_window_ms, config, "zoom_half_window_ms", 15.0))
    args.overlay_half_window_ms = float(pick_value(args.overlay_half_window_ms, config, "overlay_half_window_ms", 5.0))
    args.voltage = pick_value(args.voltage, config, "voltage")
    args.peak_times_ms = pick_value(args.peak_times_ms, config, "peak_times_ms")
    args.window_start_times_ms = pick_value(args.window_start_times_ms, config, "window_start_times_ms")
    args.peak_search_half_window_ms = float(
        pick_value(args.peak_search_half_window_ms, config, "peak_search_half_window_ms", 120.0)
    )
    args.local_baseline_window_ms = float(pick_value(args.local_baseline_window_ms, config, "local_baseline_window_ms", 5.0))
    args.local_baseline_guard_ms = float(pick_value(args.local_baseline_guard_ms, config, "local_baseline_guard_ms", 0.2))
    args.window_mode = str(pick_value(args.window_mode, config, "window_mode", "positive_peak"))
    args.transition_start_fraction = float(pick_value(args.transition_start_fraction, config, "transition_start_fraction", 0.3))
    args.transition_end_fraction = float(pick_value(args.transition_end_fraction, config, "transition_end_fraction", 0.2))
    args.transition_settle_ms = float(pick_value(args.transition_settle_ms, config, "transition_settle_ms", 20.0))
    args.transition_smooth_ms = float(pick_value(args.transition_smooth_ms, config, "transition_smooth_ms", 5.0))
    args.plateau_coarse_fraction = float(pick_value(args.plateau_coarse_fraction, config, "plateau_coarse_fraction", 0.3))
    args.expected_duration_ms = pick_value(args.expected_duration_ms, config, "expected_duration_ms")
    args.expected_duration_ms = None if args.expected_duration_ms is None else float(args.expected_duration_ms)
    args.write_zoom_outputs = bool(config.get("write_zoom_outputs", True)) and not args.skip_zoom_outputs

    csv_path = Path(csv_value).expanduser().resolve()
    summary_path = Path(args.summary_json).expanduser().resolve() if args.summary_json else None
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else csv_path.parent / (csv_path.stem + "_peak_analysis")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    times_ms, currents_ua = read_power_csv(csv_path)
    dt_ms = times_ms[1] - times_ms[0]
    sample_rate_hz = 1000.0 / dt_ms
    total_duration_ms = times_ms[-1] - times_ms[0]

    summary = load_summary_json(summary_path)
    inferred_peaks = summary_inference_count(summary)
    if args.voltage is None and summary:
        try:
            args.voltage = float(summary["energy_estimate"]["voltage_v"])
        except Exception:
            pass
    expected_duration_ms = args.expected_duration_ms
    if summary and summary.get("cnn_latency_us"):
        try:
            expected_duration_ms = float(summary["cnn_latency_us"]["avg"]) / 1000.0
        except Exception:
            pass
    elif summary and summary.get("latency", {}).get("avg_ms") is not None:
        try:
            expected_duration_ms = float(summary["latency"]["avg_ms"])
        except Exception:
            pass

    if args.window_start_times_ms:
        peak_specs = find_window_start_peaks(
            times_ms,
            currents_ua,
            args.window_start_times_ms,
            args.period_ms,
            expected_duration_ms,
        )
    elif args.peak_times_ms:
        peak_specs = find_target_peaks(
            times_ms,
            currents_ua,
            args.peak_times_ms,
            args.peak_search_half_window_ms,
        )
    else:
        peak_specs = find_period_peaks(times_ms, currents_ua, args.expected_peaks, args.period_ms)
    metrics = compute_peak_metrics(
        times_ms,
        currents_ua,
        peak_specs,
        args.window_mode,
        args.threshold_fraction,
        args.voltage,
        args.local_baseline_window_ms,
        args.local_baseline_guard_ms,
        args.transition_start_fraction,
        args.transition_end_fraction,
        args.transition_settle_ms,
        args.transition_smooth_ms,
        expected_duration_ms,
        args.plateau_coarse_fraction,
    )

    cleanup_old_peak_outputs(out_dir)
    write_peak_csv(out_dir / "peak_metrics.csv", metrics)
    save_full_plot(out_dir / "full_trace.png", times_ms, currents_ua, metrics)
    if args.write_zoom_outputs:
        write_zoom_csvs(out_dir, times_ms, currents_ua, metrics, args.zoom_half_window_ms)
        save_zoom_grid(out_dir / "peak_zoom_grid.png", times_ms, currents_ua, metrics, args.zoom_half_window_ms)
        save_overlay_plot(out_dir / "peak_overlay.png", times_ms, currents_ua, metrics, args.overlay_half_window_ms)

    avg_duration_ms = sum(m.duration_ms for m in metrics) / len(metrics)
    avg_mean_current_ua = sum(m.mean_current_ua for m in metrics) / len(metrics)
    avg_mean_excess_ua = sum(m.mean_excess_current_ua for m in metrics) / len(metrics)
    avg_charge_total_uc = sum(m.charge_total_uc for m in metrics) / len(metrics)
    avg_charge_uc = sum(m.charge_uc for m in metrics) / len(metrics)
    avg_energy_total_uj = None
    avg_energy_uj = None
    if metrics[0].energy_total_uj is not None:
        avg_energy_total_uj = sum(m.energy_total_uj for m in metrics) / len(metrics)
    if metrics[0].energy_uj is not None:
        avg_energy_uj = sum(m.energy_uj for m in metrics) / len(metrics)

    report = {
        "csv": str(csv_path),
        "config": str(config_path) if config_path else None,
        "summary_json": str(summary_path) if summary_path else None,
        "sample_interval_ms": dt_ms,
        "sample_rate_hz": sample_rate_hz,
        "total_duration_ms": total_duration_ms,
        "expected_peaks": args.expected_peaks,
        "detected_peaks": len(metrics),
        "period_ms": args.period_ms,
        "peak_times_ms": args.peak_times_ms,
        "peak_search_half_window_ms": args.peak_search_half_window_ms,
        "summary_inference_events": inferred_peaks,
        "window_mode": args.window_mode,
        "threshold_fraction": args.threshold_fraction,
        "local_baseline_window_ms": args.local_baseline_window_ms,
        "local_baseline_guard_ms": args.local_baseline_guard_ms,
        "transition_start_fraction": args.transition_start_fraction,
        "transition_end_fraction": args.transition_end_fraction,
        "transition_settle_ms": args.transition_settle_ms,
        "transition_smooth_ms": args.transition_smooth_ms,
        "plateau_coarse_fraction": args.plateau_coarse_fraction,
        "voltage_v": args.voltage,
        "avg_peak_duration_ms": avg_duration_ms,
        "avg_peak_mean_current_ua": avg_mean_current_ua,
        "avg_peak_mean_excess_current_ua": avg_mean_excess_ua,
        "avg_peak_charge_total_uc": avg_charge_total_uc,
        "avg_peak_charge_uc": avg_charge_uc,
        "avg_peak_energy_total_uj": avg_energy_total_uj,
        "avg_peak_energy_uj": avg_energy_uj,
        "per_peak": [asdict(m) for m in metrics],
        "summary_voltage_v": None,
        "summary_current_ma": None,
        "summary_power_w": None,
        "summary_energy_per_inference_uj": None,
        "avg_peak_mean_current_vs_summary_current_ratio": None,
        "avg_peak_mean_excess_current_vs_summary_current_ratio": None,
        "avg_peak_energy_total_vs_summary_ratio": None,
        "avg_peak_excess_energy_vs_summary_ratio": None,
    }

    summary_latency_ms_avg = None
    if summary and summary.get("cnn_latency_us"):
        summary_latency_ms_avg = summary["cnn_latency_us"]["avg"] / 1000.0
    elif summary and summary.get("latency", {}).get("avg_ms") is not None:
        summary_latency_ms_avg = float(summary["latency"]["avg_ms"])

    if summary_latency_ms_avg is not None:
        report["summary_cnn_latency_us_avg"] = summary_latency_ms_avg * 1000.0
        report["summary_cnn_latency_ms_avg"] = summary_latency_ms_avg
        report["duration_vs_summary_ratio"] = avg_duration_ms / summary_latency_ms_avg

    if summary and summary.get("energy_estimate"):
        energy = summary["energy_estimate"]
        try:
            report["summary_voltage_v"] = float(energy["voltage_v"])
        except Exception:
            pass
        try:
            report["summary_current_ma"] = float(energy["current_ma"])
        except Exception:
            pass
        try:
            report["summary_power_w"] = float(energy["power_w"])
        except Exception:
            pass
        try:
            report["summary_energy_per_inference_uj"] = float(energy["energy_per_inference_uj"])
        except Exception:
            pass

        if report.get("summary_current_ma") is not None:
            summary_current_ua = report["summary_current_ma"] * 1000.0
            report["avg_peak_mean_current_vs_summary_current_ratio"] = (
                avg_mean_current_ua / summary_current_ua
            )
            report["avg_peak_mean_excess_current_vs_summary_current_ratio"] = (
                avg_mean_excess_ua / summary_current_ua
            )

        if report.get("summary_energy_per_inference_uj") is not None:
            if avg_energy_total_uj is not None:
                report["avg_peak_energy_total_vs_summary_ratio"] = (
                    avg_energy_total_uj / report["summary_energy_per_inference_uj"]
                )
            if avg_energy_uj is not None:
                report["avg_peak_excess_energy_vs_summary_ratio"] = (
                    avg_energy_uj / report["summary_energy_per_inference_uj"]
                )

    (out_dir / "peak_report.json").write_text(json.dumps(report, indent=2))

    print(json.dumps({
        "sample_rate_hz": sample_rate_hz,
        "sample_interval_ms": dt_ms,
        "detected_peaks": len(metrics),
        "avg_peak_duration_ms": avg_duration_ms,
        "avg_peak_mean_current_ua": avg_mean_current_ua,
        "avg_peak_mean_excess_current_ua": avg_mean_excess_ua,
        "avg_peak_charge_total_uc": avg_charge_total_uc,
        "avg_peak_energy_uj": avg_energy_uj,
        "avg_peak_energy_total_uj": avg_energy_total_uj,
        "summary_cnn_latency_ms_avg": report.get("summary_cnn_latency_ms_avg"),
        "duration_vs_summary_ratio": report.get("duration_vs_summary_ratio"),
        "summary_energy_per_inference_uj": report.get("summary_energy_per_inference_uj"),
        "avg_peak_energy_total_vs_summary_ratio": report.get("avg_peak_energy_total_vs_summary_ratio"),
        "avg_peak_excess_energy_vs_summary_ratio": report.get("avg_peak_excess_energy_vs_summary_ratio"),
        "avg_peak_mean_current_vs_summary_current_ratio": report.get("avg_peak_mean_current_vs_summary_current_ratio"),
        "avg_peak_mean_excess_current_vs_summary_current_ratio": report.get("avg_peak_mean_excess_current_vs_summary_current_ratio"),
        "out_dir": str(out_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
