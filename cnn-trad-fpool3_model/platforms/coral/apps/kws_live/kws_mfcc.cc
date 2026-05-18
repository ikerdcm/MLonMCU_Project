#include "kws_mfcc.h"

#include <cmath>
#include <cstdint>
#include <algorithm>

#include "kws_mfcc_tables.h"

bool KwsMfccCompute(const int16_t* audio, int8_t* out_int8) {
    // Normalize by peak amplitude to match training preprocessing.
    int16_t peak = 1;
    for (uint32_t i = 0; i < MFCC_AUDIO_SAMPLES; ++i) {
        int16_t a = audio[i] < 0 ? -audio[i] : audio[i];
        if (a > peak) peak = a;
    }
    const float norm_scale = 1.0f / (float)peak;

    float windowed[MFCC_FRAME_SIZE];
    float spectrum[MFCC_FFT_BINS];
    float mel_buf[MFCC_MEL_BINS];

    for (uint32_t frame = 0; frame < MFCC_FRAME_COUNT; ++frame) {
        const uint32_t frame_start = frame * MFCC_FRAME_STEP;

        // Apply Hann window
        for (uint32_t i = 0; i < MFCC_FRAME_SIZE; ++i)
            windowed[i] = ((float)audio[frame_start + i] * norm_scale) *
                          kMfccWindow[i];

        // Goertzel DFT — identical algorithm to STM32 frontend
        for (uint32_t bin = 0; bin < MFCC_FFT_BINS; ++bin) {
            const float coeff = kMfccGoertzelCoeff[bin];
            float s1 = 0.0f, s2 = 0.0f;
            for (uint32_t i = 0; i < MFCC_FRAME_SIZE; ++i) {
                float s0 = windowed[i] + coeff * s1 - s2;
                s2 = s1; s1 = s0;
            }
            // Zero-pad to FFT_SIZE
            for (uint32_t i = MFCC_FRAME_SIZE; i < MFCC_FFT_SIZE; ++i) {
                float s0 = coeff * s1 - s2;
                s2 = s1; s1 = s0;
            }
            float real = s1 - s2 * kMfccGoertzelCos[bin];
            float imag = s2 * kMfccGoertzelSin[bin];
            spectrum[bin] = sqrtf(real * real + imag * imag);
        }

        // Mel filterbank
        for (uint32_t m = 0; m < MFCC_MEL_BINS; ++m) {
            float sum = 0.0f;
            for (uint32_t b = 0; b < MFCC_FFT_BINS; ++b)
                sum += spectrum[b] * kMfccMelMatrix[b * MFCC_MEL_BINS + m];
            mel_buf[m] = logf(sum + 1.0e-6f);
        }

        // DCT → MFCC coefficients → quantize to int8
        for (uint32_t c = 0; c < MFCC_COEFF_COUNT; ++c) {
            float mfcc = 0.0f;
            for (uint32_t m = 0; m < MFCC_MEL_BINS; ++m)
                mfcc += mel_buf[m] * kMfccDctMatrix[m * MFCC_COEFF_COUNT + c];

            float q = mfcc / MFCC_INPUT_SCALE + MFCC_INPUT_ZERO_POINT + 0.5f;
            if (q < -128.0f) q = -128.0f;
            if (q >  127.0f) q =  127.0f;
            out_int8[frame * MFCC_COEFF_COUNT + c] = (int8_t)q;
        }
    }
    return true;
}
