/*
 * DS-CNN-L Float32 live keyword-spotting on GAP9 EVK.
 *
 * Pipeline:
 *   PDM mic (Vesper, SAI1) → SFU CIC → 48 kHz PCM int32
 *   → 3:1 decimation → 16 kHz float32 (2-second window)
 *   → peak-based 1-second extraction
 *   → MFCC (490 coefficients)
 *   → Deeploy DS-CNN-L Float32 inference (8-core cluster, 240 MHz)
 *   → argmax → keyword label printed via UART
 *
 * Build:
 *   python deeployRunner_gap9.py -t Tests/Models/DSCNNL -s board -D LIVE_INFERENCE=ON
 *
 * Recording strategy:
 *   Always record 2 seconds.  After decimation we find the speech-energy peak
 *   in the 2-second buffer, then extract a 1-second (MFCC_AUDIO_LEN) window
 *   starting EXTRACT_PRE_FRAMES before the peak.  This guarantees:
 *     • Keyword can arrive any time in the first 1.5 s without being cut off.
 *     • Post-keyword nasal / plosive tail is always captured in the tail of the
 *       2-second buffer.
 *   Latency per recognition cycle: 2 s recording + ~120 ms inference ≈ 2.1 s.
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

/* VAD threshold on abs-peak of DC-removed 16 kHz signal.
 * Observed:  background ~3–15 M,  speech ~100–1000 M. */
#define VAD_THRESHOLD 300000000.0f

/* Number of MFCC frames to include before the speech peak in the extracted
 * 1-second window.  5 frames = 100 ms of pre-context. */
#define EXTRACT_PRE_FRAMES 5

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
    int run = 0;
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

        /* 5. Extract 1-second (MFCC_AUDIO_LEN) window starting
              EXTRACT_PRE_FRAMES before the peak. */
        int extract_start_frame = peak_frame_2s - EXTRACT_PRE_FRAMES;
        if (extract_start_frame < 0) extract_start_frame = 0;
        int extract_start_sample = extract_start_frame * MFCC_FRAME_STEP;
        /* Clamp so the 1-second window stays within the 2-second buffer. */
        if (extract_start_sample + MFCC_AUDIO_LEN > AUDIO_2S_16K)
            extract_start_sample = AUDIO_2S_16K - MFCC_AUDIO_LEN;

        const float *audio_1s = s_audio_16k + extract_start_sample;

        /* Debug: show where in the 2-second recording the speech is. */
        printf("--- Detected (peak=%.0f) peak_frame=%d (%.0fms in 2s buf) "
               "extract_start=%.0fms ---\r\n",
               vad_peak,
               peak_frame_2s, peak_frame_2s * 20.0f,
               extract_start_sample / 16.0f);

        /* 6. Compute MFCC on the extracted 1-second window. */
        mfcc_compute(audio_1s, s_mfcc);

        /* 6b. Contiguous-region gating + duration normalisation.
         *
         *  PROBLEM: natural "on" is ~160-180 ms (6-9 MFCC frames), but GSC
         *  training clips have the keyword at ~400 ms (20 frames).  After
         *  global-average pooling, a 9-frame keyword contributes only 9/49 =
         *  18 % of the signal — the model "sees" mostly silence and cannot
         *  reliably classify.  Stretching to 25 frames raises that to 51 %.
         *
         *  Step 1 — contiguous region: expand from the c0 peak while c0 > -20.
         *            Zero all frames outside the contiguous region (eliminates
         *            isolated PDM background leakage).
         *  Step 2 — minimum check: skip inference if < MIN_FRAMES (false trig).
         *  Step 3 — time-stretch to TARGET_FRAMES (linear interp) and place
         *            starting at TARGET_START, filling the rest with silence.
         *            Applied only when the speech region is shorter than
         *            TARGET_FRAMES; longer regions are left untouched. */
        static const float s_silence_vec[MFCC_N_MFCC] = {
            -87.4f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f
        };
        /* static scratch for stretched frames — not on stack */
        static float s_stretched[25 * MFCC_N_MFCC];

        const float gate_thresh   = -20.0f;
        const int   MIN_FRAMES    = 4;
        const int   TARGET_FRAMES = 25;   /* ~500 ms, matches GSC duration */
        const int   TARGET_START  = 5;    /* 100 ms pre-keyword silence      */

        /* ── Step 1: contiguous region ─────────────────────────────────── */
        int speech_left, speech_right;
        {
            int pk = 0;
            float pk_c0 = s_mfcc[0];
            for (int t = 1; t < MFCC_N_FRAMES; t++) {
                if (s_mfcc[t * MFCC_N_MFCC] > pk_c0) {
                    pk_c0 = s_mfcc[t * MFCC_N_MFCC];
                    pk = t;
                }
            }
            speech_left = pk;
            while (speech_left > 0 &&
                   s_mfcc[(speech_left - 1) * MFCC_N_MFCC] > gate_thresh)
                speech_left--;
            speech_right = pk;
            while (speech_right < MFCC_N_FRAMES - 1 &&
                   s_mfcc[(speech_right + 1) * MFCC_N_MFCC] > gate_thresh)
                speech_right++;
            for (int t = 0; t < speech_left; t++)
                memcpy(s_mfcc + t * MFCC_N_MFCC, s_silence_vec,
                       MFCC_N_MFCC * sizeof(float));
            for (int t = speech_right + 1; t < MFCC_N_FRAMES; t++)
                memcpy(s_mfcc + t * MFCC_N_MFCC, s_silence_vec,
                       MFCC_N_MFCC * sizeof(float));
        }
        int n_speech = speech_right - speech_left + 1;
        printf("[speech_region] frames %d-%d (%d frames, %.0f-%.0fms)\r\n",
               speech_left, speech_right, n_speech,
               speech_left * 20.0f, speech_right * 20.0f);

        /* ── Step 2: minimum duration ───────────────────────────────────── */
        if (n_speech < MIN_FRAMES) {
            printf("[skip] too short (%d frames)\r\n", n_speech);
            continue;
        }

        /* ── Step 3: time-stretch ────────────────────────────────────────── */
        if (n_speech < TARGET_FRAMES) {
            /* Linear interpolation: N input frames → TARGET_FRAMES output. */
            for (int j = 0; j < TARGET_FRAMES; j++) {
                float pos  = (float)j * (float)(n_speech - 1)
                             / (float)(TARGET_FRAMES - 1);
                int   i0   = (int)pos;
                float alpha = pos - (float)i0;
                int   i1   = (i0 + 1 < n_speech) ? i0 + 1 : i0;
                const float *f0 = s_mfcc + (speech_left + i0) * MFCC_N_MFCC;
                const float *f1 = s_mfcc + (speech_left + i1) * MFCC_N_MFCC;
                float *dst = s_stretched + j * MFCC_N_MFCC;
                for (int c = 0; c < MFCC_N_MFCC; c++)
                    dst[c] = f0[c] * (1.0f - alpha) + f1[c] * alpha;
            }
            /* Clear window, then write stretched keyword at TARGET_START. */
            for (int t = 0; t < MFCC_N_FRAMES; t++)
                memcpy(s_mfcc + t * MFCC_N_MFCC, s_silence_vec,
                       MFCC_N_MFCC * sizeof(float));
            int copy = TARGET_FRAMES;
            if (TARGET_START + copy > MFCC_N_FRAMES)
                copy = MFCC_N_FRAMES - TARGET_START;
            for (int j = 0; j < copy; j++)
                memcpy(s_mfcc + (TARGET_START + j) * MFCC_N_MFCC,
                       s_stretched + j * MFCC_N_MFCC,
                       MFCC_N_MFCC * sizeof(float));
            printf("[stretched] %d→%d frames, placed at %d-%d\r\n",
                   n_speech, TARGET_FRAMES,
                   TARGET_START, TARGET_START + copy - 1);
        }

        /* DEBUG: all 49 c0 values + full MFCC near the peak frame. */
        {
            int peak_t = 0;
            float peak_c0 = s_mfcc[0];
            for (int t = 1; t < MFCC_N_FRAMES; t++) {
                if (s_mfcc[t * MFCC_N_MFCC] > peak_c0) {
                    peak_c0 = s_mfcc[t * MFCC_N_MFCC];
                    peak_t = t;
                }
            }
            printf("[peak_mfcc] frame=%d (%.1fms in extracted) c0=%.2f\r\n",
                   peak_t, peak_t * 20.0f, peak_c0);

            printf("[c0all]");
            for (int t = 0; t < MFCC_N_FRAMES; t++) {
                printf(" %6.2f", s_mfcc[t * MFCC_N_MFCC]);
                if ((t % 7) == 6 && t != MFCC_N_FRAMES - 1)
                    printf("\r\n[c0all]");
            }
            printf("\r\n");

            int t0 = peak_t - 2; if (t0 < 0) t0 = 0;
            int t1 = peak_t + 4; if (t1 >= MFCC_N_FRAMES) t1 = MFCC_N_FRAMES - 1;
            for (int t = t0; t <= t1; t++) {
                const float *f = s_mfcc + t * MFCC_N_MFCC;
                printf("[mfcc t=%02d] %6.2f %6.2f %6.2f %6.2f %6.2f"
                       " %6.2f %6.2f %6.2f %6.2f %6.2f\r\n",
                       t, f[0],f[1],f[2],f[3],f[4],f[5],f[6],f[7],f[8],f[9]);
            }
        }

        /* 7. Copy MFCC into network input buffer.
              RunNetwork freed the buffer on the previous run (run>0), so
              re-allocate it before copying. */
        if (run > 0) {
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

        /* 8. Run inference on the cluster */
        pi_cluster_task(&cluster_task, RunNetworkWrapper, NULL);
        cluster_task.slave_stack_size = SLAVESTACKSIZE;
        pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);

        /* 9. Decode output */
        if (DeeployNetwork_num_outputs > 0) {
            float *out = (float *)DeeployNetwork_outputs[0];
            int n      = (int)(DeeployNetwork_outputs_bytes[0] / sizeof(float));
            int cls    = argmax_float(out, n);
            float conf_val = out[cls];

            const char *label = (cls < KWS_NUM_CLASSES)
                                 ? KWS_LABELS[cls]
                                 : "?";
            printf(">> %-8s  (class %d, score %.4f)  [%u cycles]\r\n",
                   label, cls, conf_val, total_cycles);
            for (int i = 0; i < n && i < KWS_NUM_CLASSES; i++)
                printf("   %-8s %.4f\r\n", KWS_LABELS[i], out[i]);
        }

        run++;
    }

    mic_close();
    return 0;
}
