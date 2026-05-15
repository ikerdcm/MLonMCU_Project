#!/usr/bin/env python3

import math
from pathlib import Path

import tensorflow as tf

FRAME_SIZE = 480
FRAME_STEP = 320
FFT_SIZE = 512
FFT_BINS = FFT_SIZE // 2 + 1
MEL_BINS = 40
MFCC_BINS = 10
SPECTROGRAM_FRAMES = 49
SAMPLE_RATE = 16000
LOWER_EDGE_HZ = 20.0
UPPER_EDGE_HZ = 4000.0


def emit_1d(name, values):
    lines = [f"static const float {name}[{len(values)}] = {{"]
    row = []
    for i, value in enumerate(values):
        row.append(f"{float(value):.9e}f")
        if len(row) == 6 or i == len(values) - 1:
            lines.append("    " + ", ".join(row) + ",")
            row = []
    lines.append("};\n")
    return "\n".join(lines)


def emit_2d_flat(name, values, rows, cols):
    flat = values.reshape(-1)
    lines = [f"static const float {name}[{rows * cols}] = {{"]
    row = []
    for i, value in enumerate(flat):
        row.append(f"{float(value):.9e}f")
        if len(row) == 6 or i == len(flat) - 1:
            lines.append("    " + ", ".join(row) + ",")
            row = []
    lines.append("};\n")
    return "\n".join(lines)


def main():
    repo_root = Path(__file__).resolve().parents[2]
    out_path = repo_root / "keyword_spotting_u5_ds_cnn" / "Core" / "Inc" / "ds_cnn_frontend_tables.h"

    window = tf.signal.hann_window(FRAME_SIZE).numpy()
    mel = tf.signal.linear_to_mel_weight_matrix(
        MEL_BINS, FFT_BINS, SAMPLE_RATE, LOWER_EDGE_HZ, UPPER_EDGE_HZ
    ).numpy()
    dct = tf.signal.mfccs_from_log_mel_spectrograms(
        tf.eye(MEL_BINS, dtype=tf.float32)
    ).numpy()[:, :MFCC_BINS]

    coeff = []
    cos_vals = []
    sin_vals = []
    for k in range(FFT_BINS):
        omega = 2.0 * math.pi * k / FFT_SIZE
        coeff.append(2.0 * math.cos(omega))
        cos_vals.append(math.cos(omega))
        sin_vals.append(math.sin(omega))

    text = [
        "#ifndef DS_CNN_FRONTEND_TABLES_H",
        "#define DS_CNN_FRONTEND_TABLES_H",
        "",
        f"#define DS_CNN_FRONTEND_FRAME_SIZE {FRAME_SIZE}u",
        f"#define DS_CNN_FRONTEND_FRAME_STEP {FRAME_STEP}u",
        f"#define DS_CNN_FRONTEND_FFT_SIZE {FFT_SIZE}u",
        f"#define DS_CNN_FRONTEND_FFT_BINS {FFT_BINS}u",
        f"#define DS_CNN_FRONTEND_MEL_BINS {MEL_BINS}u",
        f"#define DS_CNN_FRONTEND_MFCC_BINS {MFCC_BINS}u",
        f"#define DS_CNN_FRONTEND_SPECTROGRAM_FRAMES {SPECTROGRAM_FRAMES}u",
        f"#define DS_CNN_FRONTEND_SAMPLE_RATE {SAMPLE_RATE}u",
        "",
        emit_1d("ds_cnn_frontend_window", window),
        emit_1d("ds_cnn_frontend_goertzel_coeff", coeff),
        emit_1d("ds_cnn_frontend_goertzel_cos", cos_vals),
        emit_1d("ds_cnn_frontend_goertzel_sin", sin_vals),
        emit_2d_flat("ds_cnn_frontend_mel_matrix", mel, FFT_BINS, MEL_BINS),
        emit_2d_flat("ds_cnn_frontend_dct_matrix", dct, MEL_BINS, MFCC_BINS),
        "#endif",
    ]

    out_path.write_text("\n".join(text))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
