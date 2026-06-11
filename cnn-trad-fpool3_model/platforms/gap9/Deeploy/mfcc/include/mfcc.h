/*
 * MFCC feature extraction for DS-CNN-L keyword spotting.
 *
 * Matches the Float32 model's training pipeline:
 *   frame=480 (30ms), hop=320 (20ms), FFT=512, 40 mel (20-4000 Hz),
 *   Hann window (periodic), HTK mel scale, Slaney norm, magnitude (power=1),
 *   center=False, log+1e-6, DCT-II ortho → 10 coefficients.
 *
 * Pure C99, no dynamic allocation. Host-portable (no GAP9 dependencies).
 * For GAP9 deployment include via the DeeployTest CMake target.
 */

#pragma once
#include <stdint.h>

/* ── MFCC parameters ────────────────────────────────────────────────────── */
#define MFCC_SAMPLE_RATE   16000
#define MFCC_FRAME_SIZE    480
#define MFCC_FRAME_STEP    320
#define MFCC_FFT_SIZE      512
#define MFCC_N_FFT_BINS    257     /* FFT_SIZE/2 + 1 */
#define MFCC_N_MEL         40
#define MFCC_N_MFCC        10
#define MFCC_N_FRAMES      49
#define MFCC_AUDIO_LEN     16000  /* 1 second at 16 kHz */
#define MFCC_LOG_EPS       1e-6f

/* Output flat size: N_FRAMES * N_MFCC = 490 floats, row-major (frame-major).
 * output[frame * MFCC_N_MFCC + coeff]  for frame=0..48, coeff=0..9          */
#define MFCC_OUTPUT_SIZE   (MFCC_N_FRAMES * MFCC_N_MFCC)   /* 490 */

/* ── Precomputed tables (defined in mfcc_tables.c) ──────────────────────── */
extern const float mfcc_hann_window[MFCC_FRAME_SIZE];
extern const float mfcc_mel_filterbank[MFCC_N_MEL][MFCC_N_FFT_BINS];
extern const float mfcc_dct_matrix[MFCC_N_MFCC][MFCC_N_MEL];

/* ── Public API ─────────────────────────────────────────────────────────── */

/*
 * mfcc_compute() — compute MFCC features from raw 16 kHz float32 audio.
 *
 *   audio    : float32 PCM, exactly MFCC_AUDIO_LEN (16000) samples,
 *              amplitude range [-1, +1] (will be peak-normalised internally).
 *   output   : caller-allocated buffer of MFCC_OUTPUT_SIZE (490) floats.
 *              Layout: output[frame * MFCC_N_MFCC + coeff], frame-major.
 *              For Deeploy (1,49,10,1): reshape as (49,10) then add batch/channel dims.
 */
void mfcc_compute(const float *audio, float *output);

/*
 * mfcc_compute_int16() — same but accepts int16 PCM (e.g. from PDM/I2S mic).
 *   Converts to float32 by dividing by 32768.0 before computing.
 */
void mfcc_compute_int16(const int16_t *audio, float *output);
