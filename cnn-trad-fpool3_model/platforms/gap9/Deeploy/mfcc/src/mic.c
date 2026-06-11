/*
 * PDM microphone capture for GAP9 EVK (Vesper VM mic on SAI1).
 *
 * Pipeline:
 *   PDM (3.072 MHz) → SFU CIC R=64 → PCM 48 kHz int32 → L2 buffer
 *
 * The SFU graph (KWSMic.src) is pre-compiled into KWSMic_L2_Descr.c/h
 * by the cmake sfu_add_graphs() step.
 */

#include "pmsis.h"
#include "sfu_pmsis_runtime.h"
#include "KWSMic_L2_Descr.h"
#include "mic.h"

/* SAI pad numbers: SAI_X_SCK/WS/SDI/SDO = 48 + itf*4 + {0,1,2,3} */
#define SAI_BASE_PAD    48U
#define SAI_SCK(itf)    (SAI_BASE_PAD + (unsigned)(itf)*4U + 0U)
#define SAI_WS(itf)     (SAI_BASE_PAD + (unsigned)(itf)*4U + 1U)
#define SAI_SDI(itf)    (SAI_BASE_PAD + (unsigned)(itf)*4U + 2U)
#define SAI_SDO(itf)    (SAI_BASE_PAD + (unsigned)(itf)*4U + 3U)

static struct pi_device s_i2s_dev;
static pi_sfu_graph_t  *s_graph = NULL;

int mic_open(void)
{
    /* ── 1. Open SFU ─────────────────────────────────────────────────────── */
    pi_sfu_conf_t sfu_conf;
    pi_sfu_conf_init(&sfu_conf);
    if (pi_sfu_open(&sfu_conf)) {
        printf("[mic] SFU open failed\r\n");
        return -1;
    }

    s_graph = pi_sfu_graph_open(&SFU_RTD(KWSMic));
    if (!s_graph) {
        printf("[mic] SFU graph open failed\r\n");
        pi_sfu_close();
        return -1;
    }

    /* ── 2. Open I2S peripheral in PDM mode ──────────────────────────────── */
    struct pi_i2s_conf i2s_conf;
    pi_i2s_conf_init(&i2s_conf);
    i2s_conf.itf            = MIC_SAI_ITF;
    i2s_conf.frame_clk_freq = MIC_PDM_CLK_HZ;  /* PDM bit-clock */
    i2s_conf.mode           = PI_I2S_MODE_PDM;
    i2s_conf.pdm_direction  = 0b11;             /* SDI=RX, SDO=RX (Vesper) */
    i2s_conf.pdm_diff       = 0b00;             /* single-ended */

    pi_open_from_conf(&s_i2s_dev, &i2s_conf);
    if (pi_i2s_open(&s_i2s_dev)) {
        printf("[mic] I2S open failed\r\n");
        pi_sfu_graph_close(s_graph);
        pi_sfu_close();
        return -1;
    }

    /* ── 3. Configure SAI pads AFTER pi_i2s_open (matches SDK example) ───── */
    pi_pad_function_set(SAI_SCK(MIC_SAI_ITF), PI_PAD_FUNC0);
    pi_pad_function_set(SAI_WS(MIC_SAI_ITF),  PI_PAD_FUNC0);
    pi_pad_function_set(SAI_SDI(MIC_SAI_ITF), PI_PAD_FUNC0);
    pi_pad_function_set(SAI_SDO(MIC_SAI_ITF), PI_PAD_FUNC0);

    /* ── 4. Bind PDM interface to SFU graph node ─────────────────────────── */
    pi_sfu_pdm_itf_id_t pdm_id = { MIC_SAI_ITF, MIC_PDM_CHANNEL, 0 };
    if (pi_sfu_graph_pdm_bind(s_graph, SFU_Name(KWSMic, PdmIn1), &pdm_id) != 0) {
        printf("[mic] SFU PDM bind failed\r\n");
        pi_i2s_close(&s_i2s_dev);
        pi_sfu_graph_close(s_graph);
        pi_sfu_close();
        return -1;
    }

    /* ── 5. Start I2S + SFU once and keep them running between recordings.
     *       Loading the graph every call to mic_record() resets the CIC
     *       integrators to zero; the first output sample is ~0 instead of
     *       the ~100 M DC value, creating a −100 M transient after DC
     *       removal that dominates the abs-max normalisation and corrupts
     *       every MFCC frame.  By keeping the graph loaded continuously the
     *       CIC stays in steady state and only real audio drives the scale. */
    pi_i2s_ioctl(&s_i2s_dev, PI_I2S_IOCTL_START, NULL);
    pi_time_wait_us(30000);          /* mic membrane stabilisation */
    pi_sfu_graph_load(s_graph);
    pi_time_wait_us(10000);          /* CIC settling after graph load */

    return 0;
}

void mic_record(int32_t *buf, int n_samples_48k)
{
    /* Sentinel in the last sample: the uDMA writes the buffer sequentially,
     * so the last sample is overwritten exactly at transfer completion.
     * pi_time_wait_us() runs on a timer clock that drifts against the audio
     * sample clock; a pure wall-clock wait can return long before the DMA is
     * done.  The unwritten tail then holds stale data from the previous
     * recording and live words get truncated mid-vowel. */
    volatile int32_t *last = &buf[n_samples_48k - 1];
    const int32_t SENTINEL = (int32_t)0xDEADBEEF;
    *last = SENTINEL;

    /* Enqueue output buffer — SFU fills it while the graph is already
     * running.  Samples produced between iterations are silently discarded
     * by the SFU (no DMA active on MemOut); only samples produced while
     * this DMA is active land in buf[]. */
    pi_sfu_buffer_t sfu_buf;
    pi_sfu_buffer_init(&sfu_buf, buf, n_samples_48k, sizeof(int32_t));

    pi_sfu_mem_port_t *port = pi_sfu_mem_port_get(s_graph, SFU_Name(KWSMic, MemOut1));
    pi_sfu_enqueue(s_graph, port, &sfu_buf);

    /* Coarse wait for the nominal duration, then poll for true completion. */
    uint32_t duration_us = (uint32_t)n_samples_48k * 1000000UL / MIC_SAMPLE_RATE_HZ;
    pi_time_wait_us(duration_us);

    uint32_t extra_ms = 0;
    while (*last == SENTINEL) {
        pi_time_wait_us(1000);
        if (++extra_ms >= MIC_RECORD_TIMEOUT_MS) {
            uint32_t done = pi_sfu_nb_transferred_bytes_get(port);
            printf("[mic] WARNING: DMA incomplete after +%u ms (%u/%u bytes)\r\n",
                   (unsigned)extra_ms, (unsigned)done,
                   (unsigned)((uint32_t)n_samples_48k * sizeof(int32_t)));
            break;
        }
    }
    if (extra_ms > 0)
        printf("[mic] DMA lagged wall-clock wait by ~%u ms\r\n",
               (unsigned)extra_ms);
}

void mic_downsample_to_16k(const int32_t *buf_48k, float *audio_16k)
{
    /* 3:1 decimation with DC removal.
     * PDM→CIC output has a large DC offset (integrators run around 50% duty
     * cycle).  Subtract the mean so the signal is zero-centered before MFCC,
     * matching the zero-mean training audio from Google Speech Commands. */
    int64_t dc_acc = 0;
    for (int i = 0; i < MFCC_AUDIO_LEN; i++) {
        dc_acc += buf_48k[i * 3];
    }
    float dc = (float)dc_acc / (float)MFCC_AUDIO_LEN;

    for (int i = 0; i < MFCC_AUDIO_LEN; i++) {
        audio_16k[i] = (float)buf_48k[i * 3] - dc;
    }
}

void mic_close(void)
{
    pi_i2s_ioctl(&s_i2s_dev, PI_I2S_IOCTL_STOP, NULL);
    if (s_graph) {
        pi_sfu_graph_unload(s_graph);
        pi_sfu_graph_close(s_graph);
        s_graph = NULL;
    }
    pi_i2s_close(&s_i2s_dev);
    pi_sfu_close();
}
