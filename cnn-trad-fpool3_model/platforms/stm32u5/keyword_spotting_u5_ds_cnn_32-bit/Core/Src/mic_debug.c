#include "main.h"
#include "mic_debug.h"
#include "kws20_live.h"
#include "../../Drivers/BSP/B-U585I-IOT02A/b_u585i_iot02a_audio.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define MIC_SAMPLE_RATE      AUDIO_FREQUENCY_16K
#define MIC_BYTES            32768u
#define PROBE_BYTES          64000u
#define PROBE_CHUNK_SAMPLES  128u
#define ST_EXAMPLE_BYTES     (16u * 1024u)
#define ST_EXAMPLE_HALF_BYTES   (ST_EXAMPLE_BYTES / 2u)
#define ST_EXAMPLE_HALF_SAMPLES (ST_EXAMPLE_HALF_BYTES / 2u)
#define ST_EXAMPLE_MAX_EVENTS   32u

static uint8_t mic_buffer[MIC_BYTES];
static uint8_t probe_buffer[PROBE_BYTES];
static uint8_t st_example_buffer[ST_EXAMPLE_BYTES];

static volatile uint32_t rec_half_done = 0;
static volatile uint32_t rec_full_done = 0;
static volatile uint32_t rec_error = 0;
static volatile uint32_t mic_debug_active = 0;

typedef struct {
    char kind;
    int16_t min_v;
    int16_t max_v;
    uint32_t avg_abs;
} timeline_event_t;

static volatile uint32_t st_example_timeline_active = 0;
static volatile uint32_t st_example_event_count = 0;
static timeline_event_t st_example_events[ST_EXAMPLE_MAX_EVENTS];

typedef struct {
    int16_t min_v;
    int16_t max_v;
    int32_t mean;
    uint32_t avg_abs;
    uint32_t chunk_min;
    uint32_t chunk_max;
    uint32_t chunk_avg;
} probe_stats_t;

static const char *device_desc(uint32_t device)
{
    return (device == AUDIO_IN_DEVICE_DIGITAL_MIC1)
               ? "DIGITAL_MIC1 / ADF1_Filter0 / PE9-PE10 / U6"
               : "DIGITAL_MIC2 / MDF1_Filter0 / PF10-PB1 / U7";
}

static timeline_event_t analyze_pcm_samples(const uint8_t *buffer, uint32_t sample_count, char kind)
{
    timeline_event_t event;
    int64_t sum_abs = 0;

    memset(&event, 0, sizeof(event));
    event.kind = kind;

    for (uint32_t i = 0; i < sample_count; i++) {
        uint16_t lo = buffer[2u * i + 0u];
        uint16_t hi = buffer[2u * i + 1u];
        int16_t v = (int16_t)((hi << 8) | lo);

        if (i == 0u) {
            event.min_v = v;
            event.max_v = v;
        }
        if (v < event.min_v) event.min_v = v;
        if (v > event.max_v) event.max_v = v;
        sum_abs += (v < 0) ? -v : v;
    }

    if (sample_count != 0u) {
        event.avg_abs = (uint32_t)(sum_abs / (int64_t)sample_count);
    }

    return event;
}

void BSP_AUDIO_IN_HalfTransfer_CallBack(uint32_t Instance)
{
    if (kws20_live_audio_half_callback(Instance) != 0) {
        return;
    }
    if (mic_debug_active == 0u) {
        return;
    }
    rec_half_done++;
    if ((st_example_timeline_active != 0u) && (st_example_event_count < ST_EXAMPLE_MAX_EVENTS)) {
        st_example_events[st_example_event_count] =
            analyze_pcm_samples(&st_example_buffer[0], ST_EXAMPLE_HALF_SAMPLES, 'H');
        st_example_event_count++;
    }
}

void BSP_AUDIO_IN_TransferComplete_CallBack(uint32_t Instance)
{
    if (kws20_live_audio_full_callback(Instance) != 0) {
        return;
    }
    if (mic_debug_active == 0u) {
        return;
    }
    rec_full_done++;
    if ((st_example_timeline_active != 0u) && (st_example_event_count < ST_EXAMPLE_MAX_EVENTS)) {
        st_example_events[st_example_event_count] =
            analyze_pcm_samples(&st_example_buffer[ST_EXAMPLE_HALF_BYTES], ST_EXAMPLE_HALF_SAMPLES, 'F');
        st_example_event_count++;
    }
}

void BSP_AUDIO_IN_Error_CallBack(uint32_t Instance)
{
    if (kws20_live_audio_error_callback(Instance) != 0) {
        return;
    }
    if (mic_debug_active == 0u) {
        return;
    }
    rec_error++;
}

static int mic_init(uint32_t device)
{
    BSP_AUDIO_Init_t AudioInit;

    AudioInit.Device        = device;
    AudioInit.SampleRate    = MIC_SAMPLE_RATE;
    AudioInit.BitsPerSample = AUDIO_RESOLUTION_16B;
    AudioInit.ChannelsNbr   = 1;
    AudioInit.Volume        = 100;

    if (BSP_AUDIO_IN_Init(0, &AudioInit) != BSP_ERROR_NONE) {
        printf("BSP_AUDIO_IN_Init FAILED\r\n");
        return -1;
    }

    return 0;
}



static void print_block_nonzero_stats(void)
{
    const uint32_t block = 1024u;
    printf("RAW nonzero per 1024-byte block:\r\n");

    for (uint32_t b = 0; b < MIC_BYTES; b += block) {
        uint32_t nz = 0;
        uint32_t end = b + block;
        if (end > MIC_BYTES) end = MIC_BYTES;

        for (uint32_t i = b; i < end; i++) {
            if (mic_buffer[i] != 0) nz++;
        }

        printf("block %lu..%lu nonzero=%lu\r\n",
               (unsigned long)b,
               (unsigned long)(end - 1u),
               (unsigned long)nz);
    }
}


static void print_offset_binary_interpretation(void)
{
    uint32_t n16 = MIC_BYTES / 2u;

    int8_t mn = 0;
    int8_t mx = 0;
    int64_t sum = 0;
    int64_t sum_abs = 0;
    uint32_t nonzero = 0;

    for (uint32_t i = 0; i < n16; i++) {
        uint16_t lo = mic_buffer[2u * i + 0u];
        uint16_t hi = mic_buffer[2u * i + 1u];
        uint16_t u = (uint16_t)((hi << 8) | lo);

        int32_t centered = (int32_t)u - 32768;
        int8_t q = (int8_t)(centered >> 8);

        if (i == 0) {
            mn = q;
            mx = q;
        }

        if (q < mn) mn = q;
        if (q > mx) mx = q;
        if (q != 0) nonzero++;

        sum += q;
        sum_abs += (q < 0) ? -q : q;
    }

    printf("OFFSET-BINARY u16->i8 samples: %lu\r\n", (unsigned long)n16);
    printf("OB i8 min: %d\r\n", mn);
    printf("OB i8 max: %d\r\n", mx);
    printf("OB i8 mean: %ld\r\n", (long)(sum / (int64_t)n16));
    printf("OB i8 avg_abs: %ld\r\n", (long)(sum_abs / (int64_t)n16));
    printf("OB i8 nonzero: %lu/%lu\r\n", (unsigned long)nonzero, (unsigned long)n16);

    printf("first 128 OB i8 samples:\r\n");
    for (uint32_t i = 0; i < 128 && i < n16; i++) {
        uint16_t lo = mic_buffer[2u * i + 0u];
        uint16_t hi = mic_buffer[2u * i + 1u];
        uint16_t u = (uint16_t)((hi << 8) | lo);
        int32_t centered = (int32_t)u - 32768;
        int8_t q = (int8_t)(centered >> 8);

        printf("%d", q);
        if (i != 127) printf(", ");
    }
    printf("\r\n");
}

static void print_int16_interpretation(void)
{
    uint32_t n16 = MIC_BYTES / 2u;

    int16_t mn = 0;
    int16_t mx = 0;
    int64_t sum = 0;
    int64_t sum_abs = 0;
    uint32_t sat_pos = 0;
    uint32_t sat_neg = 0;
    uint32_t nonzero = 0;

    for (uint32_t i = 0; i < n16; i++) {
        uint16_t lo = mic_buffer[2u * i + 0u];
        uint16_t hi = mic_buffer[2u * i + 1u];
        int16_t v = (int16_t)((hi << 8) | lo);

        if (i == 0) {
            mn = v;
            mx = v;
        }

        if (v < mn) mn = v;
        if (v > mx) mx = v;
        if (v != 0) nonzero++;
        if (v == 32767) sat_pos++;
        if (v == -32768) sat_neg++;

        sum += v;
        sum_abs += (v < 0) ? -v : v;
    }

    printf("INT16 little-endian samples: %lu\r\n", (unsigned long)n16);
    printf("INT16 min: %d\r\n", mn);
    printf("INT16 max: %d\r\n", mx);
    printf("INT16 mean: %ld\r\n", (long)(sum / (int64_t)n16));
    printf("INT16 avg_abs: %ld\r\n", (long)(sum_abs / (int64_t)n16));
    printf("INT16 nonzero: %lu\r\n", (unsigned long)nonzero);
    printf("INT16 sat_pos_32767: %lu\r\n", (unsigned long)sat_pos);
    printf("INT16 sat_neg_32768: %lu\r\n", (unsigned long)sat_neg);

    printf("first 64 int16>>8 samples:\r\n");
    for (uint32_t i = 0; i < 64 && i < n16; i++) {
        uint16_t lo = mic_buffer[2u * i + 0u];
        uint16_t hi = mic_buffer[2u * i + 1u];
        int16_t v = (int16_t)((hi << 8) | lo);
        int8_t q = (int8_t)(v >> 8);
        printf("%d", q);
        if (i != 63) printf(", ");
    }
    printf("\r\n");
}

static void print_byte_lane_stats(void)
{
    for (uint32_t lane = 0; lane < 2; lane++) {
        int8_t mn = (int8_t)mic_buffer[lane];
        int8_t mx = (int8_t)mic_buffer[lane];
        int64_t sum_abs = 0;
        uint32_t nonzero = 0;
        uint32_t count = 0;

        for (uint32_t i = lane; i < MIC_BYTES; i += 2u) {
            int8_t s = (int8_t)mic_buffer[i];
            if (s < mn) mn = s;
            if (s > mx) mx = s;
            if (s != 0) nonzero++;
            sum_abs += (s < 0) ? -s : s;
            count++;
        }

        printf("BYTE lane %lu signed min=%d max=%d avg_abs=%ld nonzero=%lu/%lu\r\n",
               (unsigned long)lane,
               mn,
               mx,
               (long)(sum_abs / (int64_t)count),
               (unsigned long)nonzero,
               (unsigned long)count);
    }
}

static void print_stats(void)
{
    uint8_t raw_min = mic_buffer[0];
    uint8_t raw_max = mic_buffer[0];

    int8_t s_min = (int8_t)mic_buffer[0];
    int8_t s_max = (int8_t)mic_buffer[0];

    int64_t sum = 0;
    int64_t sum_abs = 0;

    uint32_t raw_0 = 0;
    uint32_t raw_255 = 0;
    uint32_t signed_neg128 = 0;
    uint32_t signed_pos127 = 0;

    for (uint32_t i = 0; i < MIC_BYTES; i++) {
        uint8_t r = mic_buffer[i];
        int8_t s = (int8_t)r;

        if (r < raw_min) raw_min = r;
        if (r > raw_max) raw_max = r;

        if (s < s_min) s_min = s;
        if (s > s_max) s_max = s;

        if (r == 0) raw_0++;
        if (r == 255) raw_255++;
        if (s == -128) signed_neg128++;
        if (s == 127) signed_pos127++;

        sum += s;
        sum_abs += (s < 0) ? -s : s;
    }

    printf("MIC bytes: %lu\r\n", (unsigned long)MIC_BYTES);

    printf("RAW u8 min: %u\r\n", raw_min);
    printf("RAW u8 max: %u\r\n", raw_max);
    printf("RAW count 0: %lu\r\n", (unsigned long)raw_0);
    printf("RAW count 255: %lu\r\n", (unsigned long)raw_255);

    printf("SIGNED i8 min: %d\r\n", s_min);
    printf("SIGNED i8 max: %d\r\n", s_max);
    printf("SIGNED mean: %ld\r\n", (long)(sum / (int64_t)MIC_BYTES));
    printf("SIGNED avg_abs: %ld\r\n", (long)(sum_abs / (int64_t)MIC_BYTES));
    printf("SIGNED count -128: %lu\r\n", (unsigned long)signed_neg128);
    printf("SIGNED count 127: %lu\r\n", (unsigned long)signed_pos127);

    printf("first 64 signed i8 samples:\r\n");
    for (uint32_t i = 0; i < 64; i++) {
        printf("%d", (int8_t)mic_buffer[i]);
        if (i != 63) printf(", ");
    }
    printf("\r\n");

    printf("first 64 raw u8 samples:\r\n");
    for (uint32_t i = 0; i < 64; i++) {
        printf("%u", mic_buffer[i]);
        if (i != 63) printf(", ");
    }
    printf("\r\n");

    print_block_nonzero_stats();
    print_byte_lane_stats();
    print_int16_interpretation();
    print_offset_binary_interpretation();
}

void mic_debug_run_once(void)
{
    printf("\r\n==============================\r\n");
    printf("STM32U5 MIC DEBUG RECORD U8\r\n");
    printf("==============================\r\n");

    rec_half_done = 0;
    rec_full_done = 0;
    rec_error = 0;

    for (uint32_t i = 0; i < MIC_BYTES; i++) {
        mic_buffer[i] = 0;
    }

    if (mic_init(AUDIO_IN_DEVICE_DIGITAL_MIC1) != 0) {
        return;
    }

    mic_debug_active = 1;

    printf("Recording %lu bytes at 16 kHz...\r\n", (unsigned long)MIC_BYTES);
    printf("Speak now.\r\n");

    printf("Calling BSP_AUDIO_IN_Record...\r\n");
    int32_t rec_ret = BSP_AUDIO_IN_Record(0, mic_buffer, sizeof(mic_buffer));
    printf("BSP_AUDIO_IN_Record returned: %ld\r\n", (long)rec_ret);

    if (rec_ret != BSP_ERROR_NONE) {
        printf("BSP_AUDIO_IN_Record FAILED ret=%ld\r\n", (long)rec_ret);
        mic_debug_active = 0;
        return;
    }

    printf("Recording for 3000 ms at 16 kHz. Speak loudly now.\r\n");

    uint32_t start = HAL_GetTick();
    while ((HAL_GetTick() - start) < 3000u && rec_error == 0) {
        /* keep recording */
    }

    uint32_t elapsed = HAL_GetTick() - start;
    printf("Stopping record after %lu ms. half=%lu full=%lu err=%lu\r\n",
           (unsigned long)elapsed,
           (unsigned long)rec_half_done,
           (unsigned long)rec_full_done,
           (unsigned long)rec_error);

    BSP_AUDIO_IN_Stop(0);
    BSP_AUDIO_IN_DeInit(0);
    mic_debug_active = 0;

    printf("Recording stopped. half=%lu full=%lu err=%lu\r\n",
           (unsigned long)rec_half_done,
           (unsigned long)rec_full_done,
           (unsigned long)rec_error);

    print_stats();

    printf("MIC debug done.\r\n");
}

static probe_stats_t analyze_probe_buffer(const uint8_t *buffer, uint32_t bytes)
{
    probe_stats_t stats;
    uint32_t sample_count = bytes / 2u;
    int64_t sum = 0;
    int64_t sum_abs = 0;
    uint64_t chunk_sum_energy = 0;
    uint32_t chunk_count = 0;

    memset(&stats, 0, sizeof(stats));

    for (uint32_t i = 0; i < sample_count; i++) {
        uint16_t lo = buffer[2u * i + 0u];
        uint16_t hi = buffer[2u * i + 1u];
        int16_t v = (int16_t)((hi << 8) | lo);

        if (i == 0u) {
            stats.min_v = v;
            stats.max_v = v;
        }

        if (v < stats.min_v) stats.min_v = v;
        if (v > stats.max_v) stats.max_v = v;

        sum += v;
        sum_abs += (v < 0) ? -v : v;
    }

    if (sample_count != 0u) {
        stats.mean = (int32_t)(sum / (int64_t)sample_count);
        stats.avg_abs = (uint32_t)(sum_abs / (int64_t)sample_count);
    }

    for (uint32_t base = 0; base < sample_count; base += PROBE_CHUNK_SAMPLES) {
        uint32_t len = PROBE_CHUNK_SAMPLES;
        int32_t chunk_mean;
        int64_t chunk_sum = 0;
        uint32_t chunk_energy = 0;

        if ((base + len) > sample_count) {
            len = sample_count - base;
        }
        if (len == 0u) {
            break;
        }

        for (uint32_t i = 0; i < len; i++) {
            uint16_t lo = buffer[2u * (base + i) + 0u];
            uint16_t hi = buffer[2u * (base + i) + 1u];
            int16_t v = (int16_t)((hi << 8) | lo);
            chunk_sum += v;
        }

        chunk_mean = (int32_t)(chunk_sum / (int64_t)len);

        for (uint32_t i = 0; i < len; i++) {
            uint16_t lo = buffer[2u * (base + i) + 0u];
            uint16_t hi = buffer[2u * (base + i) + 1u];
            int16_t v = (int16_t)((hi << 8) | lo);
            int32_t d = (int32_t)v - chunk_mean;
            chunk_energy += (uint32_t)((d < 0) ? -d : d);
        }

        chunk_energy /= len;
        if (chunk_count == 0u) {
            stats.chunk_min = chunk_energy;
            stats.chunk_max = chunk_energy;
        }
        if (chunk_energy < stats.chunk_min) stats.chunk_min = chunk_energy;
        if (chunk_energy > stats.chunk_max) stats.chunk_max = chunk_energy;
        chunk_sum_energy += chunk_energy;
        chunk_count++;
    }

    if (chunk_count != 0u) {
        stats.chunk_avg = (uint32_t)(chunk_sum_energy / chunk_count);
    }

    return stats;
}

static void print_probe_stats(const char *label, probe_stats_t stats)
{
    printf("%s min=%d max=%d mean=%ld avg_abs=%lu chunk_min=%lu chunk_max=%lu chunk_avg=%lu\r\n",
           label,
           stats.min_v,
           stats.max_v,
           (long)stats.mean,
           (unsigned long)stats.avg_abs,
           (unsigned long)stats.chunk_min,
           (unsigned long)stats.chunk_max,
           (unsigned long)stats.chunk_avg);
}

static int capture_probe_window(uint32_t device,
                                const char *phase_name,
                                uint32_t duration_ms,
                                probe_stats_t *out_stats)
{
    int32_t rec_ret;
    uint32_t start;

    rec_half_done = 0;
    rec_full_done = 0;
    rec_error = 0;
    memset(probe_buffer, 0, sizeof(probe_buffer));

    if (mic_init(device) != 0) {
        return -1;
    }

    mic_debug_active = 1;
    printf("%s | %s | window=%lu ms\r\n",
           device_desc(device),
           phase_name,
           (unsigned long)duration_ms);
    printf("Calling BSP_AUDIO_IN_Record...\r\n");

    rec_ret = BSP_AUDIO_IN_Record(0, probe_buffer, sizeof(probe_buffer));
    printf("BSP_AUDIO_IN_Record returned: %ld\r\n", (long)rec_ret);
    if (rec_ret != BSP_ERROR_NONE) {
        mic_debug_active = 0;
        BSP_AUDIO_IN_DeInit(0);
        return -1;
    }

    start = HAL_GetTick();
    while ((HAL_GetTick() - start) < duration_ms && rec_error == 0u) {
    }

    BSP_AUDIO_IN_Stop(0);
    BSP_AUDIO_IN_DeInit(0);
    mic_debug_active = 0;

    printf("Capture finished. half=%lu full=%lu err=%lu elapsed=%lu ms\r\n",
           (unsigned long)rec_half_done,
           (unsigned long)rec_full_done,
           (unsigned long)rec_error,
           (unsigned long)(HAL_GetTick() - start));

    *out_stats = analyze_probe_buffer(probe_buffer, sizeof(probe_buffer));
    return 0;
}

static void print_probe_ratio(const char *device_label,
                              probe_stats_t silence_stats,
                              probe_stats_t active_stats)
{
    uint32_t avg_abs_ratio_x1000 = (silence_stats.avg_abs != 0u)
                                       ? (active_stats.avg_abs * 1000u) / silence_stats.avg_abs
                                       : 0u;
    uint32_t chunk_max_ratio_x1000 = (silence_stats.chunk_max != 0u)
                                         ? (active_stats.chunk_max * 1000u) / silence_stats.chunk_max
                                         : 0u;

    printf("%s response ratio: avg_abs_x1000=%lu chunk_max_x1000=%lu\r\n",
           device_label,
           (unsigned long)avg_abs_ratio_x1000,
           (unsigned long)chunk_max_ratio_x1000);
}

void mic_probe_run_once(void)
{
    probe_stats_t mic1_silence;
    probe_stats_t mic1_active;
    probe_stats_t mic2_silence;
    probe_stats_t mic2_active;

    printf("\r\n==============================\r\n");
    printf("STM32U5 DUAL MIC PROBE\r\n");
    printf("==============================\r\n");
    printf("Board: B-U585I-IOT02A\r\n");
    printf("Window order:\r\n");
    printf("  1) MIC1 silence\r\n");
    printf("  2) MIC1 speak / cover / tap\r\n");
    printf("  3) MIC2 silence\r\n");
    printf("  4) MIC2 speak / cover / tap\r\n");
    printf("Each window is 2000 ms.\r\n");
    HAL_Delay(1000);

    if (capture_probe_window(AUDIO_IN_DEVICE_DIGITAL_MIC1, "MIC1 silence", 2000u, &mic1_silence) != 0) {
        printf("MIC1 silence capture failed.\r\n");
        return;
    }
    print_probe_stats("MIC1 silence", mic1_silence);
    HAL_Delay(600);

    if (capture_probe_window(AUDIO_IN_DEVICE_DIGITAL_MIC1, "MIC1 speak / cover / tap", 2000u, &mic1_active) != 0) {
        printf("MIC1 active capture failed.\r\n");
        return;
    }
    print_probe_stats("MIC1 active ", mic1_active);
    print_probe_ratio("MIC1", mic1_silence, mic1_active);
    HAL_Delay(1000);

    if (capture_probe_window(AUDIO_IN_DEVICE_DIGITAL_MIC2, "MIC2 silence", 2000u, &mic2_silence) != 0) {
        printf("MIC2 silence capture failed.\r\n");
        return;
    }
    print_probe_stats("MIC2 silence", mic2_silence);
    HAL_Delay(600);

    if (capture_probe_window(AUDIO_IN_DEVICE_DIGITAL_MIC2, "MIC2 speak / cover / tap", 2000u, &mic2_active) != 0) {
        printf("MIC2 active capture failed.\r\n");
        return;
    }
    print_probe_stats("MIC2 active ", mic2_active);
    print_probe_ratio("MIC2", mic2_silence, mic2_active);

    printf("Dual mic probe done.\r\n");
}

void st_audio_record_example_run_once(void)
{
    BSP_AUDIO_Init_t AudioInit;
    int32_t ret;
    uint32_t half_ms = (ST_EXAMPLE_HALF_SAMPLES * 1000u) / AUDIO_FREQUENCY_11K;

    printf("\r\n==============================\r\n");
    printf("ST AUDIO RECORD EXAMPLE RUN\r\n");
    printf("==============================\r\n");
    printf("Board: B-U585I-IOT02A\r\n");
    printf("Config: MIC1 / ADF1_Filter0 / 11025 Hz / 16-bit / mono / 16 KiB buffer / 3000 ms\r\n");
    printf("Speak plan: 1 s silence, 1 s speech, 1 s silence\r\n");

    rec_half_done = 0;
    rec_full_done = 0;
    rec_error = 0;
    st_example_event_count = 0;
    memset(st_example_events, 0, sizeof(st_example_events));
    memset(st_example_buffer, 0, sizeof(st_example_buffer));

    AudioInit.Device = AUDIO_IN_DEVICE_DIGITAL_MIC1;
    AudioInit.SampleRate = AUDIO_FREQUENCY_11K;
    AudioInit.BitsPerSample = AUDIO_RESOLUTION_16B;
    AudioInit.ChannelsNbr = 1;
    AudioInit.Volume = 100;

    ret = BSP_AUDIO_IN_Init(0, &AudioInit);
    printf("BSP_AUDIO_IN_Init returned: %ld\r\n", (long)ret);
    if (ret != BSP_ERROR_NONE) {
        return;
    }

    mic_debug_active = 1;
    st_example_timeline_active = 1;
    printf("Calling BSP_AUDIO_IN_Record with %lu bytes...\r\n",
           (unsigned long)sizeof(st_example_buffer));
    ret = BSP_AUDIO_IN_Record(0, st_example_buffer, sizeof(st_example_buffer));
    printf("BSP_AUDIO_IN_Record returned: %ld\r\n", (long)ret);
    if (ret != BSP_ERROR_NONE) {
        mic_debug_active = 0;
        BSP_AUDIO_IN_DeInit(0);
        return;
    }

    HAL_Delay(3000);

    ret = BSP_AUDIO_IN_Stop(0);
    st_example_timeline_active = 0;
    printf("BSP_AUDIO_IN_Stop returned: %ld\r\n", (long)ret);
    ret = BSP_AUDIO_IN_DeInit(0);
    printf("BSP_AUDIO_IN_DeInit returned: %ld\r\n", (long)ret);
    mic_debug_active = 0;

    printf("Callbacks: half=%lu full=%lu err=%lu\r\n",
           (unsigned long)rec_half_done,
           (unsigned long)rec_full_done,
           (unsigned long)rec_error);

    {
        probe_stats_t stats = analyze_probe_buffer(st_example_buffer, sizeof(st_example_buffer));
        print_probe_stats("ST example buffer", stats);
    }

    printf("Timeline per DMA half-buffer (~%lu ms each):\r\n", (unsigned long)half_ms);
    for (uint32_t i = 0; i < st_example_event_count; i++) {
        uint32_t start_ms = i * half_ms;
        uint32_t end_ms = start_ms + half_ms;
        printf("  %c[%02lu] %4lu..%4lu ms avg_abs=%lu min=%d max=%d\r\n",
               st_example_events[i].kind,
               (unsigned long)i,
               (unsigned long)start_ms,
               (unsigned long)end_ms,
               (unsigned long)st_example_events[i].avg_abs,
               st_example_events[i].min_v,
               st_example_events[i].max_v);
    }

    printf("ST audio record example run done.\r\n");
}

void mic2_input_test_run_once(void)
{
    probe_stats_t stats;
    uint32_t round = 0;

    printf("\r\n==============================\r\n");
    printf("MIC2 ONLY INPUT TEST\r\n");
    printf("==============================\r\n");
    printf("Path: DIGITAL_MIC2 / MDF1_Filter0 / PF10 CLK / PB1 DATA\r\n");
    printf("Each round records 2000 ms from MIC2 only.\r\n");
    printf("Speak, stay quiet, clap, or scream. Watch avg_abs and chunk_max.\r\n");

    while (1)
    {
        round++;

        printf("\r\n--- MIC2 capture round %lu ---\r\n", (unsigned long)round);
        printf("Recording MIC2 for 2000 ms now...\r\n");

        if (capture_probe_window(AUDIO_IN_DEVICE_DIGITAL_MIC2,
                                 "MIC2 input test",
                                 2000u,
                                 &stats) != 0)
        {
            printf("MIC2 capture failed.\r\n");
            HAL_Delay(1000);
            continue;
        }

        print_probe_stats("MIC2 input", stats);

        printf("Interpretation: avg_abs=%lu chunk_max=%lu\r\n",
               (unsigned long)stats.avg_abs,
               (unsigned long)stats.chunk_max);

        HAL_Delay(1000);
    }
}
