/**
 * DS-CNN MFCC frontend — adapted for MAX78000 (no kws20_model_io.h dependency).
 * All output elements are plain float.
 */

#include "ds_cnn_frontend.h"

#include <math.h>
#include <stddef.h>

#include "ds_cnn_frontend_tables.h"

#define DS_CNN_FRONTEND_AUDIO_SAMPLES 16000u
#define DS_CNN_FRONTEND_OUTPUT_ELEMS  (DS_CNN_FRONTEND_SPECTROGRAM_FRAMES * DS_CNN_FRONTEND_MFCC_BINS)
#define DS_CNN_FRONTEND_LOG_EPSILON   1.0e-6f
#define DS_CNN_FRONTEND_MFCC_SCALE    20.0f  /* matches kws12_mfcc.py MFCC_SCALE */

int ds_cnn_frontend_compute(const int16_t *audio,
                            uint32_t sample_count,
                            float *out,
                            uint32_t out_count)
{
    float windowed[DS_CNN_FRONTEND_FRAME_SIZE];
    float spectrum[DS_CNN_FRONTEND_FFT_BINS];
    float mel[DS_CNN_FRONTEND_MEL_BINS];
    /* Input scale: int16 -> [-1,1] via /32768, matching the training pipeline
       (kws12_mfcc.py). The previous per-clip peak normalization (1/max_sample)
       scaled every clip up to peak ~1.0 — far louder than training — so the
       log-energy term (MFCC coeff 0) landed near 0 instead of the trained ~-1,
       corrupting that input channel and biasing live predictions. */
    const float norm_scale = 1.0f / 32768.0f;

    if (audio == NULL || out == NULL) {
        return 0;
    }
    if (sample_count < DS_CNN_FRONTEND_AUDIO_SAMPLES) {
        return 0;
    }
    if (out_count < DS_CNN_FRONTEND_OUTPUT_ELEMS) {
        return 0;
    }

    for (uint32_t frame = 0; frame < DS_CNN_FRONTEND_SPECTROGRAM_FRAMES; frame++) {
        uint32_t frame_start = frame * DS_CNN_FRONTEND_FRAME_STEP;

        for (uint32_t i = 0; i < DS_CNN_FRONTEND_FRAME_SIZE; i++) {
            windowed[i] = ((float)audio[frame_start + i] * norm_scale) *
                          ds_cnn_frontend_window[i];
        }

        for (uint32_t bin = 0; bin < DS_CNN_FRONTEND_FFT_BINS; bin++) {
            float coeff     = ds_cnn_frontend_goertzel_coeff[bin];
            float cos_omega = ds_cnn_frontend_goertzel_cos[bin];
            float sin_omega = ds_cnn_frontend_goertzel_sin[bin];
            float s1 = 0.0f;
            float s2 = 0.0f;

            for (uint32_t i = 0; i < DS_CNN_FRONTEND_FRAME_SIZE; i++) {
                float s0 = windowed[i] + coeff * s1 - s2;
                s2 = s1;
                s1 = s0;
            }
            for (uint32_t i = DS_CNN_FRONTEND_FRAME_SIZE; i < DS_CNN_FRONTEND_FFT_SIZE; i++) {
                float s0 = coeff * s1 - s2;
                s2 = s1;
                s1 = s0;
            }

            {
                float real = s1 - s2 * cos_omega;
                float imag = s2 * sin_omega;
                spectrum[bin] = sqrtf(real * real + imag * imag);
            }
        }

        for (uint32_t mel_bin = 0; mel_bin < DS_CNN_FRONTEND_MEL_BINS; mel_bin++) {
            float mel_sum = 0.0f;

            for (uint32_t bin = 0; bin < DS_CNN_FRONTEND_FFT_BINS; bin++) {
                mel_sum += spectrum[bin] *
                           ds_cnn_frontend_mel_matrix[bin * DS_CNN_FRONTEND_MEL_BINS + mel_bin];
            }

            mel[mel_bin] = logf(mel_sum + DS_CNN_FRONTEND_LOG_EPSILON);
        }

        for (uint32_t coeff_idx = 0; coeff_idx < DS_CNN_FRONTEND_MFCC_BINS; coeff_idx++) {
            float mfcc = 0.0f;

            for (uint32_t mel_bin = 0; mel_bin < DS_CNN_FRONTEND_MEL_BINS; mel_bin++) {
                mfcc += mel[mel_bin] *
                        ds_cnn_frontend_dct_matrix[mel_bin * DS_CNN_FRONTEND_MFCC_BINS + coeff_idx];
            }

            float scaled = mfcc / DS_CNN_FRONTEND_MFCC_SCALE;
            if (scaled >  1.0f) scaled =  1.0f;
            if (scaled < -1.0f) scaled = -1.0f;
            out[frame * DS_CNN_FRONTEND_MFCC_BINS + coeff_idx] = scaled;
        }
    }

    return 1;
}
