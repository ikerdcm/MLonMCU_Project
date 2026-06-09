#ifndef DS_CNN_INFERENCE_H
#define DS_CNN_INFERENCE_H

/**
 * DS-CNN-L plain float32 inference engine.
 * No CMSIS-NN, no external dependencies.
 * Tensors are stored NCHW (channels first), no batch dimension.
 */

/**
 * Standard 2-D convolution + ReLU.
 *
 * @param in      Input tensor  [in_c, in_h, in_w]  (row-major)
 * @param in_c    Input channels
 * @param in_h    Input height
 * @param in_w    Input width
 * @param w       Weight tensor [out_c, in_c, kh, kw]
 * @param b       Bias vector   [out_c]
 * @param out     Output tensor [out_c, out_h, out_w]
 * @param out_c   Output channels
 * @param kh, kw  Kernel height/width
 * @param sh, sw  Stride height/width
 * @param pt,pb   Padding top/bottom
 * @param pl,pr   Padding left/right
 */
void ds_cnn_conv2d_relu(const float *in, int in_c, int in_h, int in_w,
                        const float *w, const float *b,
                        float *out, int out_c,
                        int kh, int kw, int sh, int sw,
                        int pt, int pb, int pl, int pr);

/**
 * Depthwise 2-D convolution + ReLU.
 * Each channel is convolved independently (groups == ch).
 *
 * @param in   Input tensor  [ch, in_h, in_w]
 * @param ch   Number of channels (== groups)
 * @param w    Weight tensor [ch, 1, kh, kw]
 * @param b    Bias vector   [ch]
 * @param out  Output tensor [ch, out_h, out_w]
 */
void ds_cnn_dwconv2d_relu(const float *in, int ch, int in_h, int in_w,
                          const float *w, const float *b,
                          float *out,
                          int kh, int kw, int sh, int sw,
                          int pt, int pb, int pl, int pr);

/**
 * Global / local average pooling (no padding).
 *
 * @param in    Input tensor  [ch, in_h, in_w]
 * @param ch    Channels
 * @param in_h  Input height
 * @param in_w  Input width
 * @param out   Output tensor [ch, out_h, out_w]
 * @param kh    Pool kernel height
 * @param kw    Pool kernel width
 * @param sh    Pool stride height
 * @param sw    Pool stride width
 */
void ds_cnn_avgpool2d(const float *in, int ch, int in_h, int in_w,
                      float *out,
                      int kh, int kw, int sh, int sw);

/**
 * Fully-connected layer + softmax.
 *
 * @param in    Input vector  [in_n]
 * @param in_n  Input size
 * @param w     Weight matrix [out_n, in_n]
 * @param b     Bias vector   [out_n]
 * @param out   Output vector [out_n]  (softmax probabilities)
 * @param out_n Output size
 */
void ds_cnn_dense_softmax(const float *in, int in_n,
                          const float *w, const float *b,
                          float *out, int out_n);

/**
 * Full DS-CNN-L forward pass.
 *
 * @param input_490   MFCC feature tensor, 49 frames x 10 bins = 490 floats,
 *                    laid out as [49][10] (row-major, frame-first).
 * @param scores_12   Output softmax scores, 12 classes.
 * @return            Predicted class index (argmax of scores).
 */
int ds_cnn_infer(const float *input_490, float *scores_12);

#endif /* DS_CNN_INFERENCE_H */
