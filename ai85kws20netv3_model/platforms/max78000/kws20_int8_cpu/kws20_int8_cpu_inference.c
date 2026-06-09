/*
 * True INT8 software inference for ai85kws20netv3 on MAX78000 Cortex-M4F.
 *
 * Weights: int8_t arrays from kws20_32bit_weights.h via PTQ per-layer scaling.
 *   w_i8 = round(w_float * 2^RSHIFT)   where RSHIFT = floor(log2(127/abs_max_w))
 *
 * Quantization formula per layer:
 *   out_i8 = clip((acc + 2^(RS-1)) >> RS, 0, 127)   [ReLU included]
 *   Layer 0 (voice_conv1): RS=7  → (acc+64)>>7
 *   Layer 2 (voice_conv3): RS=4  → (acc+ 8)>>4
 *   All other conv layers:  RS=5  → (acc+16)>>5
 *   FC layer: raw int32 accumulation → float32 (no shift, argmax preserves order)
 *
 * SRAM usage:
 *   _buf_a[12800] + _buf_b[12096] + _buf_pool[6048] = 30,944 bytes (~30 KB)
 */

#include "kws20_int8_cpu_inference.h"
#include "kws20_int8_cpu_weights.h"
#include <string.h>

static int8_t _buf_a[12800];
static int8_t _buf_b[12096];
static int8_t _buf_pool[6048];

/* Generic conv1d + ReLU with compile-time rshift */
#define CONV1D_RELU(rshift_val, half_val)                                    \
static void conv1d_relu_rs##rshift_val(                                      \
    const int8_t *in,  int C_in,  int T_in,                                  \
    const int8_t *w,   int C_out, int K, int pad,                            \
    int8_t *out)                                                              \
{                                                                             \
    int T_out = T_in + 2 * pad - K + 1;                                      \
    for (int co = 0; co < C_out; ++co) {                                     \
        const int8_t *wco = w + co * C_in * K;                               \
        for (int t = 0; t < T_out; ++t) {                                    \
            int32_t acc = 0;                                                  \
            for (int ci = 0; ci < C_in; ++ci) {                              \
                const int8_t *wci = wco + ci * K;                            \
                for (int k = 0; k < K; ++k) {                                \
                    int t_in = t + k - pad;                                   \
                    if (t_in >= 0 && t_in < T_in)                            \
                        acc += (int32_t)wci[k] * (int32_t)in[ci*T_in+t_in]; \
                }                                                             \
            }                                                                 \
            int32_t q = (acc + (half_val)) >> (rshift_val);                  \
            out[co * T_out + t] = (int8_t)(q <= 0 ? 0 : q > 127 ? 127 : q); \
        }                                                                     \
    }                                                                         \
}

CONV1D_RELU(7, 64)   /* voice_conv1: factor=128 */
CONV1D_RELU(5, 16)   /* most layers: factor=32  */
CONV1D_RELU(4,  8)   /* voice_conv3: factor=16  */

static void maxpool1d_k2(const int8_t *in, int C, int T_in, int8_t *out)
{
    int T_out = T_in / 2;
    for (int c = 0; c < C; ++c)
        for (int t = 0; t < T_out; ++t) {
            int8_t a = in[c * T_in + t * 2];
            int8_t b = in[c * T_in + t * 2 + 1];
            out[c * T_out + t] = (a > b) ? a : b;
        }
}

static void avgpool1d_k2(const int8_t *in, int C, int T_in, int8_t *out)
{
    int T_out = T_in / 2;
    for (int c = 0; c < C; ++c)
        for (int t = 0; t < T_out; ++t)
            out[c * T_out + t] = (int8_t)(
                ((int16_t)in[c * T_in + t * 2] + (int16_t)in[c * T_in + t * 2 + 1]) >> 1);
}

static void linear_i8_to_f32(
    const int8_t *in, int M,
    const int8_t *w,  int N,
    float *out)
{
    for (int n = 0; n < N; ++n) {
        int32_t acc = 0;
        const int8_t *wn = w + n * M;
        for (int m = 0; m < M; ++m)
            acc += (int32_t)wn[m] * (int32_t)in[m];
        out[n] = (float)acc;
    }
}

int kws20_int8_cpu_infer(const int8_t *input, float *scores)
{
    /* Layer 0: voice_conv1  128x128 → 100x128, k=1, pad=0, RS=7 */
    conv1d_relu_rs7(input,    128, 128,
                    voice_conv1_weight, 100, 1, 0,
                    _buf_a);

    /* Layer 1: voice_conv2  100x128 → 96x126, k=3, pad=0, RS=5 */
    conv1d_relu_rs5(_buf_a,   100, 128,
                    voice_conv2_weight, 96, 3, 0,
                    _buf_b);

    /* Layer 2: maxpool(2) + voice_conv3  96x63 → 64x63, k=3, pad=1, RS=4 */
    maxpool1d_k2(_buf_b, 96, 126, _buf_pool);
    conv1d_relu_rs4(_buf_pool, 96, 63,
                    voice_conv3_weight, 64, 3, 1,
                    _buf_a);

    /* Layer 3: voice_conv4  64x63 → 48x61, k=3, pad=0, RS=5 */
    conv1d_relu_rs5(_buf_a,   64, 63,
                    voice_conv4_weight, 48, 3, 0,
                    _buf_b);

    /* Layer 4: maxpool(2) + kws_conv1  48x30 → 64x30, k=3, pad=1, RS=5 */
    maxpool1d_k2(_buf_b, 48, 61, _buf_pool);
    conv1d_relu_rs5(_buf_pool, 48, 30,
                    kws_conv1_weight, 64, 3, 1,
                    _buf_a);

    /* Layer 5: kws_conv2  64x30 → 96x28, k=3, pad=0, RS=5 */
    conv1d_relu_rs5(_buf_a,   64, 30,
                    kws_conv2_weight, 96, 3, 0,
                    _buf_b);

    /* Layer 6: avgpool(2) + kws_conv3  96x14 → 100x14, k=3, pad=1, RS=5 */
    avgpool1d_k2(_buf_b, 96, 28, _buf_pool);
    conv1d_relu_rs5(_buf_pool, 96, 14,
                    kws_conv3_weight, 100, 3, 1,
                    _buf_a);

    /* Layer 7: maxpool(2) + kws_conv4  100x7 → 64x4, k=6, pad=1, RS=5 */
    maxpool1d_k2(_buf_a, 100, 14, _buf_pool);
    conv1d_relu_rs5(_buf_pool, 100, 7,
                    kws_conv4_weight, 64, 6, 1,
                    _buf_b);

    /* Layer 8: FC  256 → 21 (raw int32 → float, no shift) */
    linear_i8_to_f32(_buf_b, 256, fc_weight, 21, scores);

    return KWS_INT8_OK;
}

int kws20_int8_cpu_argmax(const float *scores, int n)
{
    int best = 0;
    for (int i = 1; i < n; ++i)
        if (scores[i] > scores[best])
            best = i;
    return best;
}
