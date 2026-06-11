/*
 * MFCC feature extraction — implementation.
 * See mfcc.h for parameter documentation.
 *
 * FFT: radix-2 DIT, in-place, 512-point real input.
 *      Convention: X[k] = sum_{n=0}^{N-1} x[n] * exp(-2*pi*j*k*n/N)  (no norm).
 *      Matches numpy.fft.rfft / torch.fft.rfft (normalized=False).
 */

#include "mfcc.h"
#include <math.h>
#include <string.h>
#include <stdint.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ── FFT helpers ────────────────────────────────────────────────────────── */

/* Bit-reversal permutation for length N (must be power-of-2). */
static void bit_reverse(float *re, float *im, int N)
{
    for (int i = 1, j = 0; i < N; i++) {
        int bit = N >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j) {
            float t;
            t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t;
        }
    }
}

/*
 * fft_inplace() — radix-2 DIT FFT, N must be power of 2.
 * Twiddle factors computed directly (sin/cosf per butterfly) for accuracy.
 * For 512 points: 9 stages × 256 butterflies = 2304 ops — fast enough.
 */
static void fft_inplace(float *re, float *im, int N)
{
    bit_reverse(re, im, N);

    for (int len = 2; len <= N; len <<= 1) {
        float ang_step = -2.0f * (float)M_PI / (float)len;
        int half = len >> 1;
        for (int i = 0; i < N; i += len) {
            for (int j = 0; j < half; j++) {
                float ang  = ang_step * (float)j;
                float tw_r = cosf(ang);
                float tw_i = sinf(ang);

                float ur = re[i + j];
                float ui = im[i + j];
                float vr = re[i + j + half] * tw_r - im[i + j + half] * tw_i;
                float vi = re[i + j + half] * tw_i + im[i + j + half] * tw_r;

                re[i + j]        = ur + vr;
                im[i + j]        = ui + vi;
                re[i + j + half] = ur - vr;
                im[i + j + half] = ui - vi;
            }
        }
    }
}

/* ── Static scratch buffers (no heap allocation) ────────────────────────── */
/* These are reused across calls — not thread-safe, but fine on GAP9/FC.     */
static float s_re[MFCC_FFT_SIZE];
static float s_im[MFCC_FFT_SIZE];
static float s_mag[MFCC_N_FFT_BINS];
static float s_mel[MFCC_N_MEL];
static float s_frame[MFCC_AUDIO_LEN];   /* normalised audio copy */

/* ── Core implementation ────────────────────────────────────────────────── */

void mfcc_compute(const float *audio, float *output)
{
    /* 1. Normalise by the maximum positive sample value — matches the training
     *    pipeline (get_dataset.py: wav / tf.reduce_max(wav)) and the STM32/
     *    MAX78000 frontend (max_sample = max over positive samples only). */
    float peak = 0.0f;
    for (int i = 0; i < MFCC_AUDIO_LEN; i++) {
        if (audio[i] > peak) peak = audio[i];
    }
    float scale = (peak > 0.0f) ? (1.0f / peak) : 1.0f;
    for (int i = 0; i < MFCC_AUDIO_LEN; i++) {
        s_frame[i] = audio[i] * scale;
    }

    /* 2–6. Per-frame processing */
    for (int t = 0; t < MFCC_N_FRAMES; t++) {
        int start = t * MFCC_FRAME_STEP;

        /* 2. Apply Hann window, zero-pad to FFT_SIZE */
        memset(s_re, 0, sizeof(s_re));
        memset(s_im, 0, sizeof(s_im));
        for (int n = 0; n < MFCC_FRAME_SIZE; n++) {
            s_re[n] = s_frame[start + n] * mfcc_hann_window[n];
        }
        /* s_re[FRAME_SIZE..FFT_SIZE-1] remain 0 (zero-padding) */

        /* 3. In-place FFT */
        fft_inplace(s_re, s_im, MFCC_FFT_SIZE);

        /* 4. Magnitude spectrum: |X[k]| for k=0..256 */
        for (int k = 0; k < MFCC_N_FFT_BINS; k++) {
            s_mag[k] = sqrtf(s_re[k] * s_re[k] + s_im[k] * s_im[k]);
        }

        /* 5. Mel filterbank: mel[i] = sum_k (fb[i][k] * mag[k]) */
        for (int i = 0; i < MFCC_N_MEL; i++) {
            float sum = 0.0f;
            const float *row = mfcc_mel_filterbank[i];
            for (int k = 0; k < MFCC_N_FFT_BINS; k++) {
                sum += row[k] * s_mag[k];
            }
            s_mel[i] = sum;
        }

        /* 6. Log + epsilon */
        for (int i = 0; i < MFCC_N_MEL; i++) {
            s_mel[i] = logf(s_mel[i] + MFCC_LOG_EPS);
        }

        /* 7. DCT-II ortho: mfcc[c] = sum_i (dct[c][i] * log_mel[i]) */
        float *out_frame = output + t * MFCC_N_MFCC;
        for (int c = 0; c < MFCC_N_MFCC; c++) {
            float sum = 0.0f;
            const float *row = mfcc_dct_matrix[c];
            for (int i = 0; i < MFCC_N_MEL; i++) {
                sum += row[i] * s_mel[i];
            }
            out_frame[c] = sum;
        }
    }
}

void mfcc_compute_int16(const int16_t *audio, float *output)
{
    /* Convert int16 PCM → float32 in [-1, 1] then run standard pipeline. */
    static float s_audio_f32[MFCC_AUDIO_LEN];
    for (int i = 0; i < MFCC_AUDIO_LEN; i++) {
        s_audio_f32[i] = (float)audio[i] / 32768.0f;
    }
    mfcc_compute(s_audio_f32, output);
}
