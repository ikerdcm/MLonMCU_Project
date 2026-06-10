#ifndef DS_CNN_FRONTEND_H
#define DS_CNN_FRONTEND_H

#include <stdint.h>

/**
 * Compute MFCC features from raw 16-bit PCM audio.
 *
 * @param audio        16-bit signed PCM samples at 16 kHz
 * @param sample_count Number of audio samples (must be >= 16000)
 * @param out          Output float buffer (must hold at least out_count floats)
 * @param out_count    Number of output feature elements (must be >= 490)
 * @return             1 on success, 0 on error
 */
int ds_cnn_frontend_compute(const int16_t *audio,
                            uint32_t sample_count,
                            float *out,
                            uint32_t out_count);

#endif /* DS_CNN_FRONTEND_H */
