/**
 * DS-CNN-L plain float32 inference engine.
 * Cortex-M4 / MAX78000 FTHR — no CMSIS-NN, no CNN accelerator.
 */

#include "ds_cnn_inference.h"
#include "ds_cnn_weights.h"

#include <math.h>
#include <string.h>
#include <stdint.h>

/* Two ping-pong activation buffers.  Max size = 64 ch * 25 H * 5 W = 8000 */
static float act_a[DS_CNN_MAX_ACT_SIZE];
static float act_b[DS_CNN_MAX_ACT_SIZE];

/* --------------------------------------------------------------------------
 * Conv2D + ReLU
 * w layout: [out_c][in_c][kh][kw]
 * -------------------------------------------------------------------------- */
void ds_cnn_conv2d_relu(const float *in, int in_c, int in_h, int in_w,
                        const float *w, const float *b,
                        float *out, int out_c,
                        int kh, int kw, int sh, int sw,
                        int pt, int pb, int pl, int pr)
{
    int out_h = (in_h + pt + pb - kh) / sh + 1;
    int out_w = (in_w + pl + pr - kw) / sw + 1;

    for (int oc = 0; oc < out_c; oc++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                float acc = b[oc];
                for (int ic = 0; ic < in_c; ic++) {
                    for (int kh_i = 0; kh_i < kh; kh_i++) {
                        int ih = oh * sh - pt + kh_i;
                        if (ih < 0 || ih >= in_h) continue;
                        for (int kw_i = 0; kw_i < kw; kw_i++) {
                            int iw = ow * sw - pl + kw_i;
                            if (iw < 0 || iw >= in_w) continue;
                            float in_val = in[ic * in_h * in_w + ih * in_w + iw];
                            float w_val  = w[((oc * in_c + ic) * kh + kh_i) * kw + kw_i];
                            acc += in_val * w_val;
                        }
                    }
                }
                /* ReLU */
                out[(oc * out_h + oh) * out_w + ow] = acc > 0.0f ? acc : 0.0f;
            }
        }
    }
}

/* --------------------------------------------------------------------------
 * Depthwise Conv2D + ReLU
 * w layout: [ch][1][kh][kw]  (ONNX groups=ch, in_ch_per_group=1)
 * -------------------------------------------------------------------------- */
void ds_cnn_dwconv2d_relu(const float *in, int ch, int in_h, int in_w,
                          const float *w, const float *b,
                          float *out,
                          int kh, int kw, int sh, int sw,
                          int pt, int pb, int pl, int pr)
{
    int out_h = (in_h + pt + pb - kh) / sh + 1;
    int out_w = (in_w + pl + pr - kw) / sw + 1;

    for (int c = 0; c < ch; c++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                float acc = b[c];
                for (int kh_i = 0; kh_i < kh; kh_i++) {
                    int ih = oh * sh - pt + kh_i;
                    if (ih < 0 || ih >= in_h) continue;
                    for (int kw_i = 0; kw_i < kw; kw_i++) {
                        int iw = ow * sw - pl + kw_i;
                        if (iw < 0 || iw >= in_w) continue;
                        float in_val = in[c * in_h * in_w + ih * in_w + iw];
                        float w_val  = w[(c * kh + kh_i) * kw + kw_i];
                        acc += in_val * w_val;
                    }
                }
                /* ReLU */
                out[(c * out_h + oh) * out_w + ow] = acc > 0.0f ? acc : 0.0f;
            }
        }
    }
}

/* --------------------------------------------------------------------------
 * Average Pooling (no padding)
 * -------------------------------------------------------------------------- */
void ds_cnn_avgpool2d(const float *in, int ch, int in_h, int in_w,
                      float *out,
                      int kh, int kw, int sh, int sw)
{
    int out_h = (in_h - kh) / sh + 1;
    int out_w = (in_w - kw) / sw + 1;
    float inv = 1.0f / (float)(kh * kw);

    for (int c = 0; c < ch; c++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                float acc = 0.0f;
                for (int kh_i = 0; kh_i < kh; kh_i++) {
                    int ih = oh * sh + kh_i;
                    for (int kw_i = 0; kw_i < kw; kw_i++) {
                        int iw = ow * sw + kw_i;
                        acc += in[c * in_h * in_w + ih * in_w + iw];
                    }
                }
                out[c * out_h * out_w + oh * out_w + ow] = acc * inv;
            }
        }
    }
}

/* --------------------------------------------------------------------------
 * Dense + Softmax
 * w layout: [out_n][in_n]  (row-major, same as Gemm transB=0)
 * -------------------------------------------------------------------------- */
void ds_cnn_dense_softmax(const float *in, int in_n,
                          const float *w, const float *b,
                          float *out, int out_n)
{
    float max_val;
    float sum;
    int i, j;

    /* Linear */
    for (i = 0; i < out_n; i++) {
        float acc = b[i];
        const float *row = w + i * in_n;
        for (j = 0; j < in_n; j++) {
            acc += row[j] * in[j];
        }
        out[i] = acc;
    }

    /* Softmax — numerically stable */
    max_val = out[0];
    for (i = 1; i < out_n; i++) {
        if (out[i] > max_val) max_val = out[i];
    }
    sum = 0.0f;
    for (i = 0; i < out_n; i++) {
        out[i] = expf(out[i] - max_val);
        sum += out[i];
    }
    if (sum > 0.0f) {
        float inv_sum = 1.0f / sum;
        for (i = 0; i < out_n; i++) {
            out[i] *= inv_sum;
        }
    }
}

/* --------------------------------------------------------------------------
 * Full DS-CNN-L forward pass
 *
 * Network topology (NCHW, no batch dim):
 *   input (1, 49, 10)  — MFCC, C=1, H=49 frames, W=10 bins
 *   CONV0: (64,1,10,4) stride(2,2) pad(4,1,5,1) -> (64,25,5)
 *   DW1:   (64,1,3,3)  groups=64 stride(1,1) pad(1,1,1,1) -> (64,25,5)
 *   PW2:   (64,64,1,1) stride(1,1) -> (64,25,5)
 *   DW3-PW12: same pattern x5
 *   AvgPool: kernel(24,5) stride(24,5) -> (64,1,1)
 *   Flatten: (64,)
 *   Dense+Softmax: (12,)
 * -------------------------------------------------------------------------- */
int ds_cnn_infer(const float *input_490, float *scores_12)
{
    float *cur  = act_a;   /* current output buffer  */
    float *prev = act_b;   /* previous output buffer */
    float *tmp;
    int best_idx;
    float best_val;

    /* ---- Reshape input: (49,10) -> (1,49,10), already row-major ---- */
    /* We use prev as scratch for the initial 1-channel tensor.
     * Since act_b has 8000 floats and input is 490, it fits. */
    memcpy(prev, input_490, 490 * sizeof(float));

    /* ---- CONV0: (1,49,10) -> (64,25,5) ---- */
    ds_cnn_conv2d_relu(prev, 1, 49, 10,
                       DS_CNN_CONV0_W, DS_CNN_CONV0_B,
                       cur, DS_CNN_CONV0_OUT_CH,
                       DS_CNN_CONV0_KH, DS_CNN_CONV0_KW,
                       DS_CNN_CONV0_SH, DS_CNN_CONV0_SW,
                       DS_CNN_CONV0_PT, DS_CNN_CONV0_PB,
                       DS_CNN_CONV0_PL, DS_CNN_CONV0_PR);
    /* cur = (64,25,5) = 8000 floats */
    tmp = prev; prev = cur; cur = tmp;

    /* ---- DW1: (64,25,5) -> (64,25,5) ---- */
    ds_cnn_dwconv2d_relu(prev, 64, 25, 5,
                         DS_CNN_DW1_W, DS_CNN_DW1_B,
                         cur,
                         DS_CNN_DW1_KH, DS_CNN_DW1_KW,
                         DS_CNN_DW1_SH, DS_CNN_DW1_SW,
                         DS_CNN_DW1_PT, DS_CNN_DW1_PB,
                         DS_CNN_DW1_PL, DS_CNN_DW1_PR);
    tmp = prev; prev = cur; cur = tmp;

    /* ---- PW2: (64,25,5) -> (64,25,5) ---- */
    ds_cnn_conv2d_relu(prev, 64, 25, 5,
                       DS_CNN_PW2_W, DS_CNN_PW2_B,
                       cur, DS_CNN_PW2_OUT_CH,
                       DS_CNN_PW2_KH, DS_CNN_PW2_KW,
                       DS_CNN_PW2_SH, DS_CNN_PW2_SW,
                       DS_CNN_PW2_PT, DS_CNN_PW2_PB,
                       DS_CNN_PW2_PL, DS_CNN_PW2_PR);
    tmp = prev; prev = cur; cur = tmp;

    /* ---- DW3 ---- */
    ds_cnn_dwconv2d_relu(prev, 64, 25, 5,
                         DS_CNN_DW3_W, DS_CNN_DW3_B,
                         cur,
                         DS_CNN_DW3_KH, DS_CNN_DW3_KW,
                         DS_CNN_DW3_SH, DS_CNN_DW3_SW,
                         DS_CNN_DW3_PT, DS_CNN_DW3_PB,
                         DS_CNN_DW3_PL, DS_CNN_DW3_PR);
    tmp = prev; prev = cur; cur = tmp;

    /* ---- PW4 ---- */
    ds_cnn_conv2d_relu(prev, 64, 25, 5,
                       DS_CNN_PW4_W, DS_CNN_PW4_B,
                       cur, DS_CNN_PW4_OUT_CH,
                       DS_CNN_PW4_KH, DS_CNN_PW4_KW,
                       DS_CNN_PW4_SH, DS_CNN_PW4_SW,
                       DS_CNN_PW4_PT, DS_CNN_PW4_PB,
                       DS_CNN_PW4_PL, DS_CNN_PW4_PR);
    tmp = prev; prev = cur; cur = tmp;

    /* ---- DW5 ---- */
    ds_cnn_dwconv2d_relu(prev, 64, 25, 5,
                         DS_CNN_DW5_W, DS_CNN_DW5_B,
                         cur,
                         DS_CNN_DW5_KH, DS_CNN_DW5_KW,
                         DS_CNN_DW5_SH, DS_CNN_DW5_SW,
                         DS_CNN_DW5_PT, DS_CNN_DW5_PB,
                         DS_CNN_DW5_PL, DS_CNN_DW5_PR);
    tmp = prev; prev = cur; cur = tmp;

    /* ---- PW6 ---- */
    ds_cnn_conv2d_relu(prev, 64, 25, 5,
                       DS_CNN_PW6_W, DS_CNN_PW6_B,
                       cur, DS_CNN_PW6_OUT_CH,
                       DS_CNN_PW6_KH, DS_CNN_PW6_KW,
                       DS_CNN_PW6_SH, DS_CNN_PW6_SW,
                       DS_CNN_PW6_PT, DS_CNN_PW6_PB,
                       DS_CNN_PW6_PL, DS_CNN_PW6_PR);
    tmp = prev; prev = cur; cur = tmp;

    /* ---- DW7 ---- */
    ds_cnn_dwconv2d_relu(prev, 64, 25, 5,
                         DS_CNN_DW7_W, DS_CNN_DW7_B,
                         cur,
                         DS_CNN_DW7_KH, DS_CNN_DW7_KW,
                         DS_CNN_DW7_SH, DS_CNN_DW7_SW,
                         DS_CNN_DW7_PT, DS_CNN_DW7_PB,
                         DS_CNN_DW7_PL, DS_CNN_DW7_PR);
    tmp = prev; prev = cur; cur = tmp;

    /* ---- PW8 ---- */
    ds_cnn_conv2d_relu(prev, 64, 25, 5,
                       DS_CNN_PW8_W, DS_CNN_PW8_B,
                       cur, DS_CNN_PW8_OUT_CH,
                       DS_CNN_PW8_KH, DS_CNN_PW8_KW,
                       DS_CNN_PW8_SH, DS_CNN_PW8_SW,
                       DS_CNN_PW8_PT, DS_CNN_PW8_PB,
                       DS_CNN_PW8_PL, DS_CNN_PW8_PR);
    tmp = prev; prev = cur; cur = tmp;

    /* ---- DW9 ---- */
    ds_cnn_dwconv2d_relu(prev, 64, 25, 5,
                         DS_CNN_DW9_W, DS_CNN_DW9_B,
                         cur,
                         DS_CNN_DW9_KH, DS_CNN_DW9_KW,
                         DS_CNN_DW9_SH, DS_CNN_DW9_SW,
                         DS_CNN_DW9_PT, DS_CNN_DW9_PB,
                         DS_CNN_DW9_PL, DS_CNN_DW9_PR);
    tmp = prev; prev = cur; cur = tmp;

    /* ---- PW10 ---- */
    ds_cnn_conv2d_relu(prev, 64, 25, 5,
                       DS_CNN_PW10_W, DS_CNN_PW10_B,
                       cur, DS_CNN_PW10_OUT_CH,
                       DS_CNN_PW10_KH, DS_CNN_PW10_KW,
                       DS_CNN_PW10_SH, DS_CNN_PW10_SW,
                       DS_CNN_PW10_PT, DS_CNN_PW10_PB,
                       DS_CNN_PW10_PL, DS_CNN_PW10_PR);
    tmp = prev; prev = cur; cur = tmp;

    /* ---- DW11 ---- */
    ds_cnn_dwconv2d_relu(prev, 64, 25, 5,
                         DS_CNN_DW11_W, DS_CNN_DW11_B,
                         cur,
                         DS_CNN_DW11_KH, DS_CNN_DW11_KW,
                         DS_CNN_DW11_SH, DS_CNN_DW11_SW,
                         DS_CNN_DW11_PT, DS_CNN_DW11_PB,
                         DS_CNN_DW11_PL, DS_CNN_DW11_PR);
    tmp = prev; prev = cur; cur = tmp;

    /* ---- PW12 ---- */
    ds_cnn_conv2d_relu(prev, 64, 25, 5,
                       DS_CNN_PW12_W, DS_CNN_PW12_B,
                       cur, DS_CNN_PW12_OUT_CH,
                       DS_CNN_PW12_KH, DS_CNN_PW12_KW,
                       DS_CNN_PW12_SH, DS_CNN_PW12_SW,
                       DS_CNN_PW12_PT, DS_CNN_PW12_PB,
                       DS_CNN_PW12_PL, DS_CNN_PW12_PR);
    tmp = prev; prev = cur; cur = tmp;
    /* prev = (64,25,5) after PW12 */

    /* ---- AvgPool: kernel(24,5) stride(24,5) -> (64,1,1) ---- */
    ds_cnn_avgpool2d(prev, 64, 25, 5,
                     cur,
                     24, 5, 24, 5);
    tmp = prev; prev = cur; cur = tmp;
    /* prev = (64,1,1) = 64 floats */

    /* ---- Dense + Softmax: 64 -> 12 ---- */
    ds_cnn_dense_softmax(prev, DS_CNN_DENSE_IN,
                         DS_CNN_DENSE_W, DS_CNN_DENSE_B,
                         scores_12, DS_CNN_DENSE_OUT);

    /* ---- Argmax ---- */
    best_idx = 0;
    best_val = scores_12[0];
    for (int i = 1; i < DS_CNN_DENSE_OUT; i++) {
        if (scores_12[i] > best_val) {
            best_val = scores_12[i];
            best_idx = i;
        }
    }

    return best_idx;
}
