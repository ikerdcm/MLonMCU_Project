/*
 * DS-CNN-L Float32 live keyword-spotting on GAP9 EVK.
 *
 * Pipeline:
 *   PDM mic (Vesper, SAI1) → SFU CIC → 48 kHz PCM int32
 *   → 3:1 decimation → 16 kHz float32 (2-second window)
 *   → multi-alignment 1-second extraction around the speech peak
 *   → MFCC (490 coefficients) per alignment — raw, no gating/stretching
 *   → Deeploy DS-CNN-L Float32 inference per alignment (8-core cluster)
 *   → softmax-average over alignments → argmax + confidence threshold
 *   → keyword label printed via UART
 *
 * Build:
 *   python deeployRunner_gap9.py -t Tests/Models/DSCNNL -s board -D LIVE_INFERENCE=ON
 *
 * Feature strategy — match the training distribution exactly:
 *   get_dataset.py computes MFCC over the complete, untouched 1-second clip:
 *   peak-normalise → MFCC.  No VAD, no gating, no stretching.  80 % of the
 *   training clips additionally carry real background noise mixed in at up to
 *   0.1 amplitude (kws_util.py defaults), so room noise around the keyword
 *   (c0 ≈ −15…−25) is IN distribution.  Synthetic silence frames (c0 −87.4)
 *   only ever appear in training as zero-padding at clip ends.  Therefore the
 *   live frontend feeds the raw MFCC of the extracted window with no
 *   post-processing whatsoever.
 *
 * Alignment strategy:
 *   GSC keywords sit roughly centred in their 1-second clip (the training
 *   code's time-shift augmentation is hardcoded off), so the model is
 *   alignment-sensitive.  We extract N_WINDOWS windows that place the
 *   amplitude peak at different frames around the GSC-typical position and
 *   average the softmax outputs (ensemble over alignments).
 *
 * Latency per recognition cycle: 2 s recording + N_WINDOWS × (MFCC + ~120 ms
 * inference) ≈ 2.5 s.
 */

#include <math.h>
#include <string.h>

#include "CycleCounter.h"
#include "Network.h"
#include "dory_mem.h"
#include "mic.h"
#include "mfcc.h"
#include "pmsis.h"

#define SLAVESTACKSIZE  3800

/* VAD threshold on abs-peak of DC-removed 16 kHz signal — used only as a
 * trigger to skip empty cycles, never to modify features.
 * Observed:  background ~3–15 M,  speech ~100–1000 M. */
#define VAD_THRESHOLD 300000000.0f

/* Ensemble window alignments: number of MFCC frames before the amplitude
 * peak.  In GSC the keyword is roughly centred, so its vowel peak typically
 * falls at ~280–520 ms (frames 14–26).  Three alignments ±120 ms around
 * frame 20 cover that range. */
static const int EXTRACT_PRE[] = { 14, 20, 26 };
#define N_WINDOWS ((int)(sizeof(EXTRACT_PRE) / sizeof(EXTRACT_PRE[0])))

/* Detection threshold on the ensemble-averaged softmax probability. */
#define CONF_THRESHOLD 0.60f

/* 2-second audio constants */
#define AUDIO_2S_48K  (2 * MIC_SAMPLES_1S)    /* 96000 samples @ 48 kHz */
#define AUDIO_2S_16K  (2 * MFCC_AUDIO_LEN)    /* 32000 samples @ 16 kHz */
#define N_FRAMES_2S   (2 * MFCC_N_FRAMES)     /* 98 frames */

/* ── Label table (must match training order) ─────────────────────────────── */
static const char *KWS_LABELS[] = {
    "down", "go", "left", "no", "off", "on",
    "right", "stop", "up", "yes", "silence", "unknown"
};
#define KWS_NUM_CLASSES ((int)(sizeof(KWS_LABELS) / sizeof(KWS_LABELS[0])))

/* ── Globals ──────────────────────────────────────────────────────────────── */
struct pi_device cluster_dev;
static uint32_t  total_cycles = 0;

/* Static L2 audio buffers */
/* 48 kHz capture buffer — 2 seconds: 96000 × 4 = 384 KB */
static int32_t s_audio_48k[AUDIO_2S_48K];
/* 16 kHz buffer — 2 seconds: 32000 × 4 = 128 KB */
static float   s_audio_16k[AUDIO_2S_16K];
/* MFCC output: 490 × 4 = 2 KB */
static float   s_mfcc[MFCC_OUTPUT_SIZE];

/* ── Cluster task wrappers ────────────────────────────────────────────────── */
void InitNetworkWrapper(void *args)
{
    (void)args;
    InitNetwork(pi_core_id(), pi_cl_cluster_nb_cores());
}

void RunNetworkWrapper(void *args)
{
    (void)args;
    ResetTimer();
    StartTimer();
    RunNetwork(pi_core_id(), pi_cl_cluster_nb_cores());
    total_cycles = getCycles();
    StopTimer();
}

/* ── Helpers ──────────────────────────────────────────────────────────────── */
static int argmax_float(const float *v, int n)
{
    int best = 0;
    for (int i = 1; i < n; i++)
        if (v[i] > v[best]) best = i;
    return best;
}

/* ── main ─────────────────────────────────────────────────────────────────── */
int main(void)
{
    /* ── Cluster + frequency setup ──────────────────────────────────────── */
    struct pi_cluster_conf conf;
    pi_cluster_conf_init(&conf);
    conf.id = 0;
    pi_open_from_conf(&cluster_dev, &conf);
    if (pi_cluster_open(&cluster_dev))
        return -1;

    pi_freq_set(PI_FREQ_DOMAIN_FC, 240000000);
    pi_freq_set(PI_FREQ_DOMAIN_CL, 240000000);

#ifndef NOFLASH
    mem_init();
    open_fs();
#endif

    /* ── Initialise neural network ──────────────────────────────────────── */
    printf("Initializing network...\r\n");
    struct pi_cluster_task cluster_task;
    pi_cluster_task(&cluster_task, InitNetworkWrapper, NULL);
    cluster_task.slave_stack_size = SLAVESTACKSIZE;
    pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);
    printf("Network ready.\r\n");

    /* ── Open microphone ────────────────────────────────────────────────── */
    if (mic_open()) {
        printf("ERROR: mic_open() failed\r\n");
        return -1;
    }
    printf("Microphone ready.\r\n");

    /* ── Inference loop ─────────────────────────────────────────────────── */
    int inference_count = 0;
    while (1) {
        /* 1. Signal that we are listening, then capture 2 seconds. */
        printf("-- Listening --\r\n");
        mic_record(s_audio_48k, AUDIO_2S_48K);

        /* 2. DC-remove + 3:1 decimate 48 kHz → 16 kHz for the full 2 seconds.
              mic_downsample_to_16k() only handles 1 second; do it inline. */
        {
            int64_t dc_acc = 0;
            for (int i = 0; i < AUDIO_2S_16K; i++)
                dc_acc += s_audio_48k[i * 3];
            float dc = (float)dc_acc / (float)AUDIO_2S_16K;
            for (int i = 0; i < AUDIO_2S_16K; i++)
                s_audio_16k[i] = (float)s_audio_48k[i * 3] - dc;
        }

        /* 3. VAD: abs-peak over the full 2-second buffer. */
        float vad_peak = 0.0f;
        for (int i = 0; i < AUDIO_2S_16K; i++) {
            float v = s_audio_16k[i];
            if (v < 0.0f) v = -v;
            if (v > vad_peak) vad_peak = v;
        }
        if (vad_peak < VAD_THRESHOLD)
            continue;

        /* 3b. CIC saturation check: Q31 output clips near ±2^31.  Clipped
              vowels distort the spectrum, so flag recordings that get close
              to full scale (observed peaks up to ~2.0e9 ≈ 93 % FS). */
        {
            int clipped = 0;
            for (int i = 0; i < AUDIO_2S_16K; i++) {
                float v = s_audio_16k[i];
                if (v > 2.0e9f || v < -2.0e9f) clipped++;
            }
            if (clipped > 0)
                printf("[clip] %d samples near int32 full scale — "
                       "speak softer / increase CIC_Shift\r\n", clipped);
        }

        /* 4. Find the speech peak: frame with highest abs-max amplitude.
              Searching in MFCC frame steps (320 samples = 20 ms per frame). */
        int peak_frame_2s = 0;
        float peak_amp_2s = 0.0f;
        for (int t = 0; t < N_FRAMES_2S; t++) {
            int start = t * MFCC_FRAME_STEP;
            int end   = start + MFCC_FRAME_STEP;
            if (end > AUDIO_2S_16K) end = AUDIO_2S_16K;
            float frame_max = 0.0f;
            for (int i = start; i < end; i++) {
                float v = s_audio_16k[i];
                if (v < 0.0f) v = -v;
                if (v > frame_max) frame_max = v;
            }
            if (frame_max > peak_amp_2s) {
                peak_amp_2s = frame_max;
                peak_frame_2s = t;
            }
        }

        printf("--- Detected (peak=%.0f) peak_frame=%d (%.0fms in 2s buf) ---\r\n",
               vad_peak, peak_frame_2s, peak_frame_2s * 20.0f);

        /* 5. Ensemble over N_WINDOWS alignments: extract the 1-second window,
              compute the raw MFCC (peak-normalise → MFCC, exactly like
              training), run inference, accumulate softmax outputs. */
        float probs_avg[KWS_NUM_CLASSES];
        memset(probs_avg, 0, sizeof(probs_avg));
        int n_run = 0;
        int prev_start_sample = -1;

        for (int w = 0; w < N_WINDOWS; w++) {
            int start_frame = peak_frame_2s - EXTRACT_PRE[w];
            if (start_frame < 0) start_frame = 0;
            int start_sample = start_frame * MFCC_FRAME_STEP;
            /* Clamp so the 1-second window stays within the 2-second buffer. */
            if (start_sample + MFCC_AUDIO_LEN > AUDIO_2S_16K)
                start_sample = AUDIO_2S_16K - MFCC_AUDIO_LEN;

            /* Clamping can collapse alignments onto the same window when the
               peak sits near a buffer edge — skip duplicates. */
            if (start_sample == prev_start_sample)
                continue;
            prev_start_sample = start_sample;

            /* 5a. Raw MFCC of the full window — no gating, no stretching. */
            mfcc_compute(s_audio_16k + start_sample, s_mfcc);

#ifdef KWS_DEBUG_C0
            /* Full c0 trace — costs ~400 bytes of semihosting I/O per window,
               which stretches the deaf gap between recordings.  Enable only
               for frontend debugging. */
            printf("[c0all]");
            for (int t = 0; t < MFCC_N_FRAMES; t++) {
                printf(" %6.2f", s_mfcc[t * MFCC_N_MFCC]);
                if ((t % 7) == 6 && t != MFCC_N_FRAMES - 1)
                    printf("\r\n[c0all]");
            }
            printf("\r\n");
#endif

            /* 5b. Copy MFCC into network input buffer.
                  RunNetwork freed the buffer on the previous inference, so
                  re-allocate it before copying. */
            if (inference_count > 0) {
                for (uint32_t buf = 0; buf < DeeployNetwork_num_inputs; buf++) {
                    DeeployNetwork_inputs[buf] =
                        pi_l2_malloc(DeeployNetwork_inputs_bytes[buf]);
                }
            }
            if (DeeployNetwork_num_inputs > 0 &&
                DeeployNetwork_inputs[0] != NULL) {
                memcpy(DeeployNetwork_inputs[0], s_mfcc,
                       DeeployNetwork_inputs_bytes[0]);
            }

            /* 5c. Run inference on the cluster. */
            pi_cluster_task(&cluster_task, RunNetworkWrapper, NULL);
            cluster_task.slave_stack_size = SLAVESTACKSIZE;
            pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);
            inference_count++;

            /* 5d. Accumulate softmax probabilities, report this alignment. */
            if (DeeployNetwork_num_outputs > 0) {
                const float *out = (const float *)DeeployNetwork_outputs[0];
                int n = (int)(DeeployNetwork_outputs_bytes[0] / sizeof(float));
                if (n > KWS_NUM_CLASSES) n = KWS_NUM_CLASSES;
                for (int i = 0; i < n; i++)
                    probs_avg[i] += out[i];
                n_run++;
                int cls = argmax_float(out, n);
                printf("[win %d] peak@frame %2d  start=%4dms  best=%-8s p=%.4f"
                       "  [%u cycles]\r\n",
                       w, EXTRACT_PRE[w], start_sample / 16,
                       KWS_LABELS[cls], out[cls], total_cycles);
            }
        }
        if (n_run == 0)
            continue;

        /* 6. Decode the ensemble average. */
        for (int i = 0; i < KWS_NUM_CLASSES; i++)
            probs_avg[i] /= (float)n_run;

        int   cls  = argmax_float(probs_avg, KWS_NUM_CLASSES);
        float conf = probs_avg[cls];

        if (conf >= CONF_THRESHOLD)
            printf(">> %-8s  (class %d, avg score %.4f)\r\n",
                   KWS_LABELS[cls], cls, conf);
        else
            printf(">> no detection  (best: %s, avg score %.4f < %.2f)\r\n",
                   KWS_LABELS[cls], conf, CONF_THRESHOLD);
        for (int i = 0; i < KWS_NUM_CLASSES; i++)
            printf("   %-8s %.4f\r\n", KWS_LABELS[i], probs_avg[i]);
    }

    mic_close();
    return 0;
}
