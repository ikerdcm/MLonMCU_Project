#ifndef CNN_INFERENCE_H
#define CNN_INFERENCE_H

#include <stdint.h>

/*
 * Run one KWS inference using the CNN accelerator.
 *
 * mfcc_in  : float[490] from ds_cnn_frontend_compute(), values in [-1, 1]
 *            layout: mfcc_in[t * 10 + c], t = 0..48, c = 0..9
 * scores   : float[12] softmax probabilities (may be NULL to skip)
 *
 * Returns the predicted class index (0..11), or -1 on error.
 * After return, cnn_time holds the accelerator inference time in µs.
 */
int cnn_infer(const float *mfcc_in, float *scores);

#endif /* CNN_INFERENCE_H */
