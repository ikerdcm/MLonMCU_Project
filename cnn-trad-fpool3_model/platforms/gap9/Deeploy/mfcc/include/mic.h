/*
 * PDM microphone capture via GAP9 SFU.
 *
 * Targets the Vesper VM microphone on the GAP9 EVK (SAI1, PDM channel 2).
 * SFU uses CIC R=64 to decimate 3.072 MHz PDM → 48 kHz PCM (int32 Q31).
 * mic_record() fills a caller-supplied int32 buffer at 48 kHz.
 *
 * Typical usage:
 *   int32_t  buf[MIC_SAMPLES_1S];          // 48 kHz, 1 second
 *   float    audio[MFCC_AUDIO_LEN];        // 16 kHz, 1 second
 *   if (mic_open()) return -1;
 *   while (1) {
 *       mic_record(buf, MIC_SAMPLES_1S);
 *       mic_downsample_to_16k(buf, audio);
 *       mfcc_compute(audio, mfcc_out);
 *       ...
 *   }
 *   mic_close();
 */

#pragma once
#include <stdint.h>
#include "mfcc.h"

/* Microphone hardware constants (Vesper on GAP9 EVK) */
#define MIC_PDM_CLK_HZ      3072000U    /* PDM bit-clock frequency */
#define MIC_SAMPLE_RATE_HZ  48000U      /* PCM output rate after CIC decimation */
#define MIC_SAMPLES_1S      48000       /* Samples per second at MIC_SAMPLE_RATE_HZ */
#define MIC_SAI_ITF         1           /* SAI interface for Vesper mic */
#define MIC_PDM_CHANNEL     2           /* PDM channel within SAI1 */

/* Extra time mic_record() may poll for DMA completion beyond the nominal
 * recording duration before giving up with a warning.  The wall-clock timer
 * drifts against the audio sample clock, so the DMA regularly finishes later
 * than the nominal wait. */
#define MIC_RECORD_TIMEOUT_MS  3000U

/*
 * mic_open() — open I2S peripheral + SFU graph.
 * Must be called once before mic_record().
 * Returns 0 on success, -1 on error.
 */
int mic_open(void);

/*
 * mic_record() — capture n_samples_48k samples at 48 kHz into buf (int32 Q31).
 * Blocks until the DMA transfer has actually completed (nominal duration plus
 * up to MIC_RECORD_TIMEOUT_MS of completion polling).
 * buf must be allocated by caller (n_samples_48k * sizeof(int32_t) bytes).
 */
void mic_record(int32_t *buf, int n_samples_48k);

/*
 * mic_downsample_to_16k() — decimate 48 kHz int32 buffer to 16 kHz float32.
 *
 *   buf_48k  : input, MIC_SAMPLES_1S int32 samples at 48 kHz
 *   audio_16k: output, MFCC_AUDIO_LEN float32 samples at 16 kHz
 *
 * Uses simple 3:1 decimation (every 3rd sample). mfcc_compute() peak-normalises
 * the output internally, so absolute scaling does not matter.
 */
void mic_downsample_to_16k(const int32_t *buf_48k, float *audio_16k);

/*
 * mic_close() — stop and release I2S + SFU resources.
 */
void mic_close(void);
