#!/usr/bin/env python3
"""Unified DS-CNN cross-MCU test bench (device-in-the-loop).

One command per (board, model) produces a NORMALIZED record — on-device
accuracy + 12-class confusion matrix + inference latency + flash/SRAM — by
streaming the SAME Google Speech Commands *test* audio to a board running its
EVAL-mode firmware. Each board runs its OWN frontend + model and returns the
prediction (+ cnn_us), so the result reflects the real deployed pipeline and is
directly comparable across boards. Add boards to REGISTRY one at a time.

Why audio (not MFCC): each model is scale-locked to its own frontend's MFCC, so
a shared MFCC set isn't portable — raw audio is, and it also exercises the
frontend (where bugs live).

Flow:
  1. flash the board's EVAL firmware (KWS20_CFG_ENABLE_EVAL=1) — see --flash-hint
  2. python3 testbench.py --board max --model v1 --per-class 12
  -> results/<board>_<model>/{summary.json, confusion_matrix.png}
     + an upserted row in results/RESULTS_LEDGER.md (NEW ledger, one unique row/model;
       the old Experiments/RESULTS_LEDGER.md is never touched)

Serial protocol (host -> board, 115200), one clip at a time:
  host : "EVAL <idx> <nsamples>\n"  then  nsamples * int16 little-endian
  board: "BENCH,event=eval,idx=<idx>,pred_idx=<p>,cnn_us=<us>\r\n"
"""
from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.io import wavfile

# DS-CNN 12-class order — must match each firmware's labels[] / protocol.LABELS.
LABELS = ["down", "go", "left", "no", "off", "on",
          "right", "stop", "up", "yes", "silence", "unknown"]
# Google Speech Commands raw_test/ folder -> our class index.
GSC_DIR_TO_IDX = {
    "down": 0, "go": 1, "left": 2, "no": 3, "off": 4, "on": 5,
    "right": 6, "stop": 7, "up": 8, "yes": 9, "_silence_": 10, "_unknown_": 11,
}
WINDOW = 16000  # 1 s @ 16 kHz
MODEL_ACCURACY = {
    "v0": 92.4,
    "v1": 92.0,
}

PLATFORMS = Path(__file__).resolve().parents[1]          # .../cnn-trad-fpool3_model/platforms
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"                                # all new, unique results live here
LEDGER = RESULTS / "RESULTS_LEDGER.md"                     # NEW ledger, separate from Experiments/
DEFAULT_DATASET = Path("/Users/iker/MAX78000_Toolchain/ai8x-training/data/KWS/raw_test")
SIZE_TOOL = "arm-none-eabi-size"

# (board, model) registry. `elf` is relative to `dir`. `flash_hint` is printed
# (flashing is done by the user — hardware op). Add U5/Coral entries here.
REGISTRY = {
    ("max", "v1"): {
        "name": "MAX78000 int8-accel (kws-ds-cnn-l-kws12)",
        "config_id": "int8-accel",
        "dir": PLATFORMS / "max78000/keyword_spotting_max78000_ds_cnn_8-bit_with-acc",
        "elf": "build/max78000.elf",
        "labels": LABELS,
        "flash_hint": "set KWS20_CFG_ENABLE_EVAL=1, then "
                      "./tools/build_flash_offline.sh  (build+flash the EVAL fw)",
    },
    ("max", "v0"): {
        "name": "MAX78000 fp32-cpu (DS-CNN-L float)",
        "config_id": "fp32-cpu",
        "dir": PLATFORMS / "max78000/keyword_spotting_max78000_ds_cnn_32-bit",
        "elf": "build/max78000.elf",
        "labels": LABELS,
        "flash_hint": "set KWS20_CFG_ENABLE_EVAL=1, then "
                      "./tools/build_flash_max78000_ds_cnn.sh  (build+flash the EVAL fw)",
    },
    ("u5", "v1"): {
        "name": "STM32U5 int8-cpu (DS-CNN-L, X-CUBE-AI int8/CMSIS-NN)",
        "config_id": "int8-cpu",
        "dir": PLATFORMS / "stm32u5/keyword_spotting_u5_ds_cnn_8-bit",
        "elf": "build/Debug/keyword_spotting_u5_ds_cnn_v1.elf",
        "labels": LABELS,
        "flash_hint": "set KWS20_CFG_ENABLE_EVAL=1, then "
                      "./tools/build_flash.sh --mode offline  (CMake build+flash)",
    },
    ("u5", "v0"): {
        "name": "STM32U5 fp32-cpu (DS-CNN-L, X-CUBE-AI float32)",
        "config_id": "fp32-cpu",
        "dir": PLATFORMS / "stm32u5/keyword_spotting_u5_ds_cnn_32-bit",
        "elf": "Debug/keyword_spotting_u5_ds_cnn.elf",
        "labels": LABELS,
        "flash_hint": "set KWS20_CFG_ENABLE_EVAL=1, then "
                      "./tools/build_flash.sh --mode offline  (headless CubeIDE build+flash)",
    },
    ("coral", "v1"): {
        "name": "Coral Edge-TPU int8-accel (DS-CNN-L v2)",
        "config_id": "int8-accel",
        "dir": PLATFORMS / "coral",
        "elf": "build/kws_apps/kws_eval_stream/kws_eval_stream",
        "labels": LABELS,
        "assert_dtr": True,    # Coral CDC drops bytes until DTR is asserted
        "embedded": True,      # runs a baked LittleFS audio set (console drops USB RX)
        "flash_hint": "python3 ../testbench/make_eval_audio_set.py --per-class N , then "
                      "./scripts/build_and_flash_eval_stream.sh",
    },
    ("coral", "v0"): {
        "name": "Coral fp32-cpu (DS-CNN-L float, M7)",
        "config_id": "fp32-cpu",
        "dir": PLATFORMS / "coral",
        "elf": "build/kws_apps/kws_eval_stream_cpu/kws_eval_stream_cpu",
        "labels": LABELS,
        "assert_dtr": True,
        "embedded": True,
        "flash_hint": "python3 ../testbench/make_eval_audio_set.py --per-class N , then "
                      "./scripts/build_and_flash_eval_stream_cpu.sh",
    },
}


def load_clip(path: Path) -> np.ndarray:
    """16 kHz mono wav -> int16[WINDOW] (zero-padded / truncated)."""
    _, data = wavfile.read(path)
    if data.ndim > 1:
        data = data[:, 0]
    data = data.astype(np.int16)
    if len(data) < WINDOW:
        data = np.concatenate([data, np.zeros(WINDOW - len(data), np.int16)])
    return data[:WINDOW]


def build_subset(dataset: Path, per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    items = []
    for d, idx in sorted(GSC_DIR_TO_IDX.items(), key=lambda kv: kv[1]):
        wavs = sorted((dataset / d).glob("*.wav"))
        if not wavs:
            print(f"  ! no wavs for '{d}' (idx {idx}) — skipping")
            continue
        pick = rng.choice(len(wavs), min(per_class, len(wavs)), replace=False)
        items += [(wavs[i], idx) for i in sorted(pick)]
    return items


def stream_eval(items, port, baud, timeout, assert_dtr=False):
    """Stream clips; collect [(true, pred, cnn_us)]."""
    import serial
    out = []
    with serial.Serial(port, baud, timeout=timeout) as ser:
        if assert_dtr:
            ser.dtr = True   # Coral CDC drops RX until DTR is asserted
        time.sleep(0.3)
        ser.reset_input_buffer()
        for idx, (path, true_idx) in enumerate(items):
            ser.write(f"EVAL {idx} {WINDOW}\n".encode())
            ser.write(load_clip(path).astype("<i2").tobytes())
            ser.flush()
            pred, cnn_us = None, None
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                line = ser.readline().decode("utf-8", "replace").strip()
                if line.startswith("BENCH,event=eval,") and f"idx={idx}" in line:
                    fields = dict(kv.split("=", 1) for kv in line.split(",")[1:] if "=" in kv)
                    pred = int(fields.get("pred_idx", -1))
                    cnn_us = float(fields["cnn_us"]) if "cnn_us" in fields else None
                    break
            mark = "ok" if pred == true_idx else "X "
            print(f"  [{idx+1}/{len(items)}] {mark} true={LABELS[true_idx]:8} "
                  f"pred={LABELS[pred] if pred is not None and 0 <= pred < 12 else pred}"
                  f"{f'  {cnn_us/1000:.2f} ms' if cnn_us else ''}")
            out.append((true_idx, pred if pred is not None else -1, cnn_us))
    return out


def read_embedded_eval(port, baud, assert_dtr=False, total_timeout=600):
    """For boards that run an EMBEDDED audio set autonomously (Coral): don't
    stream — just read one pass of self-describing eval lines:
        BENCH,event=eval,idx=,pred_idx=,true_idx=,cnn_us=
    framed by eval_ready/eval_done. Returns [(true, pred, cnn_us)]."""
    import serial
    out, started, waited_note = [], False, False
    deadline = time.monotonic() + total_timeout
    with serial.Serial(port, baud, timeout=2) as ser:
        if assert_dtr:
            ser.dtr = True
        time.sleep(0.3)
        ser.reset_input_buffer()
        print("  waiting for the board's eval pass to start (a pass re-runs every ~8 s)...")
        while time.monotonic() < deadline:
            line = ser.readline().decode("utf-8", "replace").strip()
            if not line:
                continue
            is_eval = line.startswith("BENCH,event=eval,")
            f = (dict(kv.split("=", 1) for kv in line.split(",")[1:] if "=" in kv)
                 if is_eval else {})
            idx = int(f["idx"]) if f.get("idx", "").lstrip("-").isdigit() else None
            # Start a fresh pass on eval_ready OR on idx==0 (robust if we missed ready)
            if "event=eval_ready" in line or (is_eval and idx == 0):
                out, started = [], True
                if "event=eval_ready" in line:
                    print(f"  {line}")
            elif "event=eval_done" in line and started and out:
                break
            if started and is_eval:
                try:
                    t, p = int(f["true_idx"]), int(f["pred_idx"])
                except (KeyError, ValueError):
                    continue
                u = float(f["cnn_us"]) if "cnn_us" in f else None
                out.append((t, p, u))
                mark = "ok" if p == t else "X "
                print(f"  [{len(out)}] {mark} true={LABELS[t]:8} "
                      f"pred={LABELS[p] if 0 <= p < 12 else p}"
                      f"{f'  {u/1000:.2f} ms' if u else ''}")
            elif not started and not waited_note:
                waited_note = True
                print(f"  (board alive, mid-pass — will start at the next pass) e.g. {line[:54]}")
    return out


def confusion_and_metrics(triples):
    n = len(LABELS)
    cm = [[0] * n for _ in range(n)]
    for t, p, _ in triples:
        if 0 <= t < n and 0 <= p < n:
            cm[t][p] += 1
    correct = sum(cm[i][i] for i in range(n))
    total = sum(sum(r) for r in cm)
    acc = correct / total if total else 0.0
    per_class = {}
    for i, lab in enumerate(LABELS):
        support = sum(cm[i])
        pred_i = sum(cm[r][i] for r in range(n))
        per_class[lab] = {
            "support": support,
            "recall": cm[i][i] / support if support else None,
            "precision": cm[i][i] / pred_i if pred_i else None,
        }
    return cm, acc, correct, total, per_class


def latency_stats(triples):
    us = sorted(v for _, _, v in triples if v)
    if not us:
        return None
    a = np.array(us)
    return {"count": len(a), "avg_ms": float(a.mean() / 1000),
            "median_ms": float(np.median(a) / 1000),
            "p95_ms": float(np.percentile(a, 95) / 1000),
            "min_ms": float(a.min() / 1000), "max_ms": float(a.max() / 1000)}


def read_elf_size(elf: Path):
    if not elf.exists():
        return None
    try:
        out = subprocess.check_output([SIZE_TOOL, str(elf)], text=True).splitlines()
        t, d, b = (int(x) for x in out[1].split()[:3])
    except Exception as e:
        print(f"  ! size read failed ({e})")
        return None
    return {"elf": str(elf), "text_bytes": t, "data_bytes": d, "bss_bytes": b,
            "flash_text_kib": t / 1024, "static_sram_kib": (d + b) / 1024}


def plot_confusion(cm, acc, title, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    a = np.array(cm, float)
    norm = a / np.clip(a.sum(1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(LABELS))); ax.set_xticklabels(LABELS, rotation=45, ha="right")
    ax.set_yticks(range(len(LABELS))); ax.set_yticklabels(LABELS)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"{title}\naccuracy {acc*100:.1f}%")
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            if cm[i][j]:
                ax.text(j, i, cm[i][j], ha="center", va="center",
                        color="white" if norm[i][j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def print_confusion(cm):
    print("\nConfusion (rows=true, cols=pred):")
    print("       " + " ".join(f"{l[:4]:>4}" for l in LABELS))
    for i, lab in enumerate(LABELS):
        print(f"{lab[:6]:>6} " + " ".join(f"{cm[i][j]:>4}" for j in range(len(LABELS))))


def upsert_ledger(board, model, row):
    """Append `row`, or REPLACE the existing row for this (board, model) so the
    ledger always holds one unique, latest result per MCU/model."""
    header = (
        "# DS-CNN normalized test-bench ledger (v2)\n\n"
        "Device-in-the-loop, one common GSC **test** audio set per board "
        "(`testbench.py`). One unique row per (board, model) — re-running replaces it. "
        "Model accuracy is the canonical model score; MCU accuracy is the on-device "
        "test-bench result. Flash/SRAM come from the ELF. Energy still needs the "
        "power meter (separate).\n\n"
        "| Timestamp | Board | Config | Model | N | Model accuracy % | MCU accuracy % | "
        "Lat avg (ms) | Lat p95 (ms) | Flash .text (KiB) | SRAM (KiB) | Run |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    if not LEDGER.exists():
        LEDGER.write_text(header)
    lines = LEDGER.read_text().splitlines()

    def is_match(line):
        cells = [c.strip() for c in line.split("|")]
        return len(cells) > 5 and cells[2] == board and cells[4] == model

    for i, line in enumerate(lines):
        if line.startswith("|") and "---" not in line and is_match(line):
            lines[i] = row
            break
    else:
        lines.append(row)
    LEDGER.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--per-class", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--port", default=None)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--timeout", type=float, default=15.0)
    ap.add_argument("--deployed-elf", type=Path, default=None,
                    help="ELF to size for flash/SRAM (default: the EVAL build; "
                         "point at the deployed offline/live build for true footprint)")
    ap.add_argument("--no-serial", action="store_true",
                    help="build+check the subset only (no board)")
    args = ap.parse_args()

    spec = REGISTRY.get((args.board, args.model))
    if not spec:
        sys.exit(f"unknown (board,model)=({args.board},{args.model}); "
                 f"have: {sorted('/'.join(k) for k in REGISTRY)}")
    embedded = spec.get("embedded", False)   # board runs a baked set autonomously (Coral)

    print(f"== {spec['name']} ==")
    items = None
    if not embedded:
        if not args.dataset.exists():
            sys.exit(f"dataset not found: {args.dataset}")
        print(f"Subset: {args.per_class}/class from {args.dataset}")
        items = build_subset(args.dataset, args.per_class, args.seed)
        print(f"  {len(items)} clips, {len({i for _, i in items})} classes")
    else:
        print("  embedded eval — the board runs the baked /eval/audio_set.bin")

    if args.no_serial:
        if items:
            a = load_clip(items[0][0])
            print(f"  sanity: {items[0][0].name} -> int16[{len(a)}] [{a.min()},{a.max()}]")
        print(f"  flash hint: {spec['flash_hint']}")
        if embedded:
            print("  bake first: python3 make_eval_audio_set.py --per-class N")
        return

    port = args.port or next(iter(sorted(
        glob.glob("/dev/tty.usbmodem*") + glob.glob("/dev/cu.usbmodem*"))), None)
    if not port:
        sys.exit("no serial port — pass --port")

    if embedded:
        print(f"Reading embedded eval on {port}...")
        print(f"  (if nothing comes: {spec['flash_hint']})")
        triples = read_embedded_eval(port, args.baud,
                                     assert_dtr=spec.get("assert_dtr", False))
    else:
        print(f"Streaming on {port} (board must be in EVAL mode)...")
        print(f"  (if it times out: {spec['flash_hint']})")
        triples = stream_eval(items, port, args.baud, args.timeout,
                              assert_dtr=spec.get("assert_dtr", False))
    cm, acc, correct, total, per_class = confusion_and_metrics(triples)
    lat = latency_stats(triples)
    size = read_elf_size(args.deployed_elf or (spec["dir"] / spec["elf"]))
    eval_build_mem = size is not None and args.deployed_elf is None
    print_confusion(cm)
    print(f"\nAccuracy: {acc*100:.2f}%  ({correct}/{total})")
    if lat:
        print(f"Latency : avg {lat['avg_ms']:.3f} ms  p95 {lat['p95_ms']:.3f} ms")
    if size:
        note = "  ⚠ EVAL build — pass --deployed-elf for deployed footprint" if eval_build_mem else ""
        print(f"Memory  : flash .text {size['flash_text_kib']:.1f} KiB  "
              f"SRAM {size['static_sram_kib']:.1f} KiB  (from {Path(size['elf']).name}){note}")
    if size:
        size["from_eval_build"] = eval_build_mem

    # One central folder, one unique subfolder per MCU/model (overwrites on re-run).
    run_id = f"{args.board}_{args.model}"
    out_dir = RESULTS / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion(cm, acc, spec["name"], out_dir / "confusion_matrix.png")
    (out_dir / "summary.json").write_text(json.dumps({
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(), "board": args.board, "model": args.model,
        "name": spec["name"], "config_id": spec["config_id"], "dataset": str(args.dataset),
        "per_class": args.per_class, "seed": args.seed, "n_clips": total,
        "model_accuracy": MODEL_ACCURACY.get(args.model), "accuracy": acc, "correct": correct, "labels": LABELS,
        "confusion_matrix": cm, "per_class": per_class, "latency": lat, "memory": size,
    }, indent=2))

    lat_avg = f"{lat['avg_ms']:.3f}" if lat else "—"
    lat_p95 = f"{lat['p95_ms']:.3f}" if lat else "—"
    flash = f"{size['flash_text_kib']:.1f}" if size else "—"
    sram = f"{size['static_sram_kib']:.1f}" if size else "—"
    model_accuracy = MODEL_ACCURACY.get(args.model)
    model_accuracy_str = f"{model_accuracy:.1f}" if model_accuracy is not None else "—"
    upsert_ledger(
        args.board, args.model,
        f"| {datetime.now().strftime('%Y-%m-%d %H:%M')} | {args.board} | "
        f"{spec['config_id']} | {args.model} | {total} | {model_accuracy_str} | {acc*100:.2f} | "
        f"{lat_avg} | {lat_p95} | {flash} | {sram} | results/{run_id} |")
    print(f"\nWrote {out_dir/'summary.json'} + confusion_matrix.png")
    print(f"Upserted {args.board}/{args.model} row in {LEDGER.name}")


if __name__ == "__main__":
    main()
