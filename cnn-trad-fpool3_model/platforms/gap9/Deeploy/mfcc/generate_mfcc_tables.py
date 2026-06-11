#!/usr/bin/env python3
"""
Generate precomputed MFCC tables for the C implementation.

Matches the preprocessing pipeline used to train ds_cnn_l.onnx (Float32):
  1. Peak-normalise audio by max positive sample value  (tf.reduce_max)
  2. Hann-windowed magnitude mel spectrogram
       frame=480 (30ms), hop=320 (20ms), FFT=512, 40 mel bins (20-4000 Hz)
       HTK mel scale, NO bandwidth normalisation (matches tf.signal.linear_to_mel_weight_matrix)
       magnitude (power=1.0), center=False
  3. Natural log + 1e-6
  4. DCT-II ortho, first 10 coefficients
  (No /20 scaling — that is only for the INT8 model)

Outputs:
  src/mfcc_tables.c   — C arrays for hann window, mel filterbank, DCT matrix
  tests/test_vectors.npz — reference input/output pairs for validation
"""

import math
import os
import struct
import numpy as np

# ── MFCC parameters ───────────────────────────────────────────────────────────
SR         = 16000
FRAME_SIZE = 480
FRAME_STEP = 320
FFT_SIZE   = 512
N_FFT_BINS = FFT_SIZE // 2 + 1   # 257
N_MEL      = 40
F_MIN      = 20.0
F_MAX      = 4000.0
N_MFCC     = 10
N_FRAMES   = 49
LOG_EPS    = 1e-6
AUDIO_LEN  = 16000

OUT_DIR    = os.path.dirname(__file__)
SRC_DIR    = os.path.join(OUT_DIR, "src")
TEST_DIR   = os.path.join(OUT_DIR, "tests")

# ── Hann window (periodic=True, length=480) ───────────────────────────────────
# torch.hann_window(N) with default periodic=True:
#   w[n] = 0.5 * (1 - cos(2*pi*n / N))  for n=0..N-1
def make_hann_window(n):
    ns = np.arange(n, dtype=np.float64)
    return 0.5 * (1.0 - np.cos(2.0 * math.pi * ns / n)).astype(np.float32)

# ── HTK mel scale ─────────────────────────────────────────────────────────────
def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)

def mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)

# ── Mel filterbank (no normalisation, HTK scale) ──────────────────────────────
# Matches tf.signal.linear_to_mel_weight_matrix(40, 257, 16000, 20.0, 4000.0)
# used during training.  Triangular filters peak at 1.0 — NO bandwidth scaling.
# (Previous version applied Slaney 2/bw normalisation, which shifted every
# MFCC coefficient by ~28 units relative to what the model was trained on.)
def make_mel_filterbank(n_mels, n_fft, sr, f_min, f_max):
    n_fft_bins = n_fft // 2 + 1
    # n_mels+2 equally-spaced mel points → convert to Hz
    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)
    mel_pts = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_pts  = mel_to_hz(mel_pts)   # (n_mels+2,) in Hz

    # FFT bin centre frequencies
    fft_freqs = np.linspace(0.0, sr / 2.0, n_fft_bins)  # (n_fft_bins,)

    fb = np.zeros((n_fft_bins, n_mels), dtype=np.float64)
    for i in range(n_mels):
        lower  = hz_pts[i]
        center = hz_pts[i + 1]
        upper  = hz_pts[i + 2]

        # Ascending slope: lower → center
        mask_up = (fft_freqs >= lower) & (fft_freqs <= center)
        fb[mask_up, i] = (fft_freqs[mask_up] - lower) / (center - lower + 1e-12)

        # Descending slope: center → upper
        mask_dn = (fft_freqs > center) & (fft_freqs <= upper)
        fb[mask_dn, i] = (upper - fft_freqs[mask_dn]) / (upper - center + 1e-12)

        # No normalisation — filters peak at 1.0, matching TF's implementation.

    # Transpose to (n_mels, n_fft_bins) — the C layout we use
    return fb.T.astype(np.float32)   # (40, 257)

# ── DCT-II ortho matrix matching tf.signal.mfccs_from_log_mel_spectrograms ───
# dct[c, m] = basis-fn c evaluated at mel bin m
# mfcc = dct @ log_mel   (10,40) @ (40,) → (10,)
def make_dct_matrix(n_mfcc, n_mel):
    m = np.arange(n_mel, dtype=np.float64)
    k = np.arange(n_mfcc, dtype=np.float64)[:, np.newaxis]
    dct = np.cos(math.pi / n_mel * k * (m + 0.5))  # (n_mfcc, n_mel)
    dct[0]  *= 1.0 / math.sqrt(n_mel)
    dct[1:] *= math.sqrt(2.0 / n_mel)
    return dct.astype(np.float32)  # (10, 40)

# ── Python reference MFCC (used for validation) ───────────────────────────────
def compute_mfcc_reference(audio_f32, hann, mel_fb, dct_mat):
    """
    audio_f32: (AUDIO_LEN,) float32
    Returns: (N_FRAMES, N_MFCC) float32
    """
    # 1. Peak normalise by max positive value
    peak = float(audio_f32.max())
    if peak > 0.0:
        audio_f32 = audio_f32 / peak

    output = np.zeros((N_FRAMES, N_MFCC), dtype=np.float32)
    for t in range(N_FRAMES):
        start = t * FRAME_STEP
        frame = audio_f32[start : start + FRAME_SIZE].copy()
        # 2. Apply Hann window
        frame *= hann
        # 3. Zero-pad to FFT_SIZE and compute FFT
        padded = np.zeros(FFT_SIZE, dtype=np.float64)
        padded[:FRAME_SIZE] = frame
        spectrum = np.fft.rfft(padded)  # (257,) complex
        mag = np.abs(spectrum).astype(np.float32)  # (257,) magnitude
        # 4. Mel filterbank: (40,257) @ (257,) → (40,)
        mel = mel_fb @ mag  # (40,)
        # 5. Log + eps
        log_mel = np.log(mel + LOG_EPS)
        # 6. DCT: (10,40) @ (40,) → (10,)
        mfcc = dct_mat @ log_mel
        output[t] = mfcc
    return output  # (49, 10)

# ── Generate tables ───────────────────────────────────────────────────────────
print("Computing MFCC tables...")
hann    = make_hann_window(FRAME_SIZE)
mel_fb  = make_mel_filterbank(N_MEL, FFT_SIZE, SR, F_MIN, F_MAX)
dct_mat = make_dct_matrix(N_MFCC, N_MEL)

print(f"  Hann window:    {hann.shape}  range=[{hann.min():.4f}, {hann.max():.4f}]")
print(f"  Mel filterbank: {mel_fb.shape}  nnz={np.count_nonzero(mel_fb)}")
print(f"  DCT matrix:     {dct_mat.shape}")

# Optional: validate against torchaudio if available
try:
    import torch
    import torchaudio.transforms as T

    mel_spec_ref = T.MelSpectrogram(
        sample_rate=SR, n_fft=FFT_SIZE, win_length=FRAME_SIZE, hop_length=FRAME_STEP,
        f_min=F_MIN, f_max=F_MAX, n_mels=N_MEL,
        window_fn=torch.hann_window, normalized=False, power=1.0,
        center=False, norm=None, mel_scale='htk',
    )
    ref_fb = mel_spec_ref.mel_scale.fb.numpy().T  # (40, 257)
    ref_hann = torch.hann_window(FRAME_SIZE).numpy()

    fb_err   = np.abs(mel_fb - ref_fb).max()
    hann_err = np.abs(hann - ref_hann).max()
    print(f"\n[torchaudio validation]")
    print(f"  Mel FB max abs err vs torchaudio:   {fb_err:.2e}  {'✓' if fb_err < 1e-5 else '✗ FAIL'}")
    print(f"  Hann window max abs err vs torch:   {hann_err:.2e}  {'✓' if hann_err < 1e-7 else '✗ FAIL'}")
except ImportError:
    print("  (torchaudio not available — skipping cross-check)")

# ── Write C tables ─────────────────────────────────────────────────────────────
def fmt_float(v):
    return f"{v:.9e}f"

lines = [
    "/* AUTO-GENERATED by generate_mfcc_tables.py — do not edit manually */",
    "",
    '#include "mfcc.h"',
    "",
]

# Hann window
lines.append(f"const float mfcc_hann_window[{FRAME_SIZE}] = {{")
for i in range(0, FRAME_SIZE, 8):
    chunk = hann[i:i+8]
    lines.append("    " + ", ".join(fmt_float(v) for v in chunk) + ",")
lines.append("};")
lines.append("")

# Mel filterbank [N_MEL][N_FFT_BINS]
lines.append(f"const float mfcc_mel_filterbank[{N_MEL}][{N_FFT_BINS}] = {{")
for i in range(N_MEL):
    row = mel_fb[i]
    lines.append(f"    /* mel bin {i:2d} */ {{")
    for j in range(0, N_FFT_BINS, 8):
        chunk = row[j:j+8]
        lines.append("        " + ", ".join(fmt_float(v) for v in chunk) + ",")
    lines.append("    },")
lines.append("};")
lines.append("")

# DCT matrix [N_MFCC][N_MEL]
lines.append(f"const float mfcc_dct_matrix[{N_MFCC}][{N_MEL}] = {{")
for c in range(N_MFCC):
    row = dct_mat[c]
    lines.append(f"    /* coeff {c} */ {{")
    lines.append("        " + ", ".join(fmt_float(v) for v in row) + ",")
    lines.append("    },")
lines.append("};")
lines.append("")

tables_path = os.path.join(SRC_DIR, "mfcc_tables.c")
with open(tables_path, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"\nWrote: {tables_path}")

# ── Generate test vectors ─────────────────────────────────────────────────────
test_cases = {}

# 1. Silence (all zeros)
silence = np.zeros(AUDIO_LEN, dtype=np.float32)
# peak=0, no normalisation → all zeros → log(0+eps) = log(eps) → constant MFCC
silence_mfcc = compute_mfcc_reference(silence, hann, mel_fb, dct_mat)
test_cases["silence"] = (silence, silence_mfcc)

# 2. Pure 1 kHz sine wave at full scale
t = np.arange(AUDIO_LEN) / SR
sine_1k = np.sin(2 * math.pi * 1000.0 * t).astype(np.float32)
sine_mfcc = compute_mfcc_reference(sine_1k, hann, mel_fb, dct_mat)
test_cases["sine_1k"] = (sine_1k, sine_mfcc)

# 3. "Left" test input from ds_cnn_test_input_left.h (raw MFCC, no audio)
# We load expected MFCC output directly and create a dummy audio that we
# can verify end-to-end via a chirp signal with known output.
# Instead, use white noise to stress-test numerical stability.
rng = np.random.default_rng(42)
noise = rng.standard_normal(AUDIO_LEN).astype(np.float32) * 0.5
noise_mfcc = compute_mfcc_reference(noise, hann, mel_fb, dct_mat)
test_cases["white_noise"] = (noise, noise_mfcc)

# 4. Load real "left" keyword audio if available and compute MFCC
try:
    import soundfile as sf
    wav_candidates = [
        os.path.join(OUT_DIR, "../../data/left_b528edb3_nohash_0.wav"),
        os.path.expanduser("~/Downloads/left_b528edb3_nohash_0.wav"),
    ]
    for cand in wav_candidates:
        if os.path.exists(cand):
            audio, sr_file = sf.read(cand, dtype='float32', always_2d=False)
            if sr_file != SR:
                print(f"  Skipping {cand}: SR={sr_file} (need {SR})")
                break
            if len(audio) > AUDIO_LEN:
                audio = audio[:AUDIO_LEN]
            elif len(audio) < AUDIO_LEN:
                audio = np.pad(audio, (0, AUDIO_LEN - len(audio)))
            real_mfcc = compute_mfcc_reference(audio, hann, mel_fb, dct_mat)
            test_cases["left_keyword"] = (audio, real_mfcc)
            print(f"  Loaded real audio: {cand}")
            break
except ImportError:
    pass

# Save test vectors
npz_path = os.path.join(TEST_DIR, "test_vectors.npz")
save_dict = {}
for name, (audio, mfcc) in test_cases.items():
    save_dict[f"{name}_audio"] = audio
    save_dict[f"{name}_mfcc"]  = mfcc
np.savez(npz_path, **save_dict)
print(f"Wrote: {npz_path}  ({len(test_cases)} test cases)")

# Also write a C header with test vectors for host test binary
hdr_lines = [
    "/* AUTO-GENERATED by generate_mfcc_tables.py */",
    "#pragma once",
    '#include "mfcc.h"',
    "",
    f"#define TEST_AUDIO_LEN {AUDIO_LEN}",
    "",
]
for name, (audio, mfcc) in test_cases.items():
    uname = name.upper()
    hdr_lines.append(f"static const float test_audio_{name}[{AUDIO_LEN}] = {{")
    for i in range(0, AUDIO_LEN, 8):
        chunk = audio[i:i+8]
        hdr_lines.append("    " + ", ".join(fmt_float(v) for v in chunk) + ",")
    hdr_lines.append("};")
    hdr_lines.append("")
    hdr_lines.append(f"static const float test_mfcc_ref_{name}[{N_FRAMES * N_MFCC}] = {{")
    flat = mfcc.flatten()
    for i in range(0, len(flat), 8):
        chunk = flat[i:i+8]
        hdr_lines.append("    " + ", ".join(fmt_float(v) for v in chunk) + ",")
    hdr_lines.append("};")
    hdr_lines.append("")

tv_h_path = os.path.join(TEST_DIR, "test_vectors.h")
with open(tv_h_path, "w") as f:
    f.write("\n".join(hdr_lines) + "\n")
print(f"Wrote: {tv_h_path}")

print("\nDone. Tables ready for C build.")
