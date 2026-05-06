#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import wave
from pathlib import Path

import numpy as np
import onnx


def load_config(path: Path) -> dict:
    return json.loads(path.read_text())


def read_wav_pcm16(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sampwidth = wav.getsampwidth()
        rate = wav.getframerate()
        frames = wav.getnframes()
        raw = wav.readframes(frames)
    if sampwidth != 2:
        raise ValueError(f"{path}: expected 16-bit PCM, got sample width {sampwidth}")
    if channels != 1:
        data = np.frombuffer(raw, dtype="<i2").reshape(-1, channels)[:, 0]
    else:
        data = np.frombuffer(raw, dtype="<i2")
    if rate != 16000:
        raise ValueError(f"{path}: expected 16 kHz, got {rate}")
    return data.astype(np.int16, copy=False)


def to_kws_tensor(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fixed = np.zeros(16384, dtype=np.int16)
    n = min(samples.shape[0], 16384)
    fixed[:n] = samples[:n]
    q = np.clip(np.round(fixed.astype(np.float32) / 256.0), -128, 127).astype(np.int8)
    f = (q.astype(np.float32) / 128.0).reshape(128, 128)
    return q, f


def detect_model_layout(model_path: Path) -> str:
    model = onnx.load(str(model_path))
    dims = []
    for dim in model.graph.input[0].type.tensor_type.shape.dim:
        dims.append(dim.dim_value if dim.HasField("dim_value") else None)
    if dims == [1, 1, 128, 128]:
        return "nchw"
    if dims == [1, 128, 128, 1]:
        return "nhwc"
    return "unknown"


def collect_wavs(dataset_root: Path) -> list[Path]:
    wavs = []
    for path in sorted(dataset_root.glob("*/*.wav")):
        wavs.append(path)
    if not wavs:
        raise SystemExit(f"No wav files found below {dataset_root}")
    return wavs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    dataset_root = Path(cfg["dataset_root"]).expanduser().resolve()
    model_path = Path(cfg["model_path"]).expanduser().resolve()
    out_dir = Path(cfg["output_dir"]).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    num_samples = int(cfg.get("num_samples", 260))
    seed = int(cfg.get("seed", 42))
    layout = detect_model_layout(model_path)

    wavs = collect_wavs(dataset_root)
    rng = random.Random(seed)
    rng.shuffle(wavs)
    chosen = wavs[:num_samples]

    labels: list[str] = []
    label_to_idx: dict[str, int] = {}
    x_rows: list[np.ndarray] = []

    for wav_path in chosen:
        label = wav_path.parent.name
        if label not in label_to_idx:
            label_to_idx[label] = len(label_to_idx)
            labels.append(label)
        pcm = read_wav_pcm16(wav_path)
        _, tensor_f = to_kws_tensor(pcm)
        x_rows.append(tensor_f)

    x = np.stack(x_rows, axis=0).astype(np.float32)
    y = np.array([label_to_idx[p.parent.name] for p in chosen], dtype=np.int64)

    x_nhwc = x[..., None]
    x_nchw = x[:, None, :, :]

    nhwc_path = out_dir / "calibration_u5_current_model_nhwc.npz"
    nchw_path = out_dir / "calibration_u5_current_model_nchw.npz"
    meta_path = out_dir / "calibration_u5_current_model_metadata.json"

    np.savez(nhwc_path, x_test=x_nhwc, y_test=y)
    np.savez(nchw_path, x_test=x_nchw, y_test=y)

    meta = {
        "dataset_root": str(dataset_root),
        "model_path": str(model_path),
        "detected_model_layout": layout,
        "num_samples": int(x.shape[0]),
        "x_nhwc_shape": list(x_nhwc.shape),
        "x_nchw_shape": list(x_nchw.shape),
        "labels": labels,
        "value_range": {
            "min": float(x.min()),
            "max": float(x.max())
        }
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"[OK] wrote {nhwc_path}")
    print(f"[OK] wrote {nchw_path}")
    print(f"[OK] wrote {meta_path}")
    print(f"[INFO] detected model layout: {layout}")
    print(f"[INFO] x_nhwc shape: {x_nhwc.shape}")
    print(f"[INFO] x_nchw shape: {x_nchw.shape}")


if __name__ == "__main__":
    main()
