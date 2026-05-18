#ifndef KWS_MFCC_H
#define KWS_MFCC_H

#include <cstdint>

// Compute MFCC features from 16kHz int16 PCM audio and quantize to int8.
// audio       : exactly MFCC_AUDIO_SAMPLES (16000) samples, normalized by max
// out_int8    : MFCC_OUTPUT_ELEMS (490) int8 values, row-major [frame][coeff]
// Returns true on success.
bool KwsMfccCompute(const int16_t* audio, int8_t* out_int8);

#endif  // KWS_MFCC_H
