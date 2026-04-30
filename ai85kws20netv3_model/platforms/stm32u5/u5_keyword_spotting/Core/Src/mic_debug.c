#include "main.h"
#include "mic_debug.h"
#include "../../Drivers/BSP/B-U585I-IOT02A/b_u585i_iot02a_audio.h"

#include <stdint.h>
#include <stdio.h>

#define MIC_SAMPLE_RATE AUDIO_FREQUENCY_16K
#define MIC_BYTES       32768u

static uint8_t mic_buffer[MIC_BYTES];

static volatile uint32_t rec_half_done = 0;
static volatile uint32_t rec_full_done = 0;
static volatile uint32_t rec_error = 0;

void BSP_AUDIO_IN_HalfTransfer_CallBack(uint32_t Instance)
{
    (void)Instance;
    rec_half_done++;
}

void BSP_AUDIO_IN_TransferComplete_CallBack(uint32_t Instance)
{
    (void)Instance;
    rec_full_done++;
}

void BSP_AUDIO_IN_Error_CallBack(uint32_t Instance)
{
    (void)Instance;
    rec_error++;
}

static int mic_init(void)
{
    BSP_AUDIO_Init_t AudioInit;

    AudioInit.Device        = AUDIO_IN_DEVICE_DIGITAL_MIC1;
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

    if (mic_init() != 0) {
        return;
    }

    printf("Recording %lu bytes at 16 kHz...\r\n", (unsigned long)MIC_BYTES);
    printf("Speak now.\r\n");

    printf("Calling BSP_AUDIO_IN_Record...\r\n");
    int32_t rec_ret = BSP_AUDIO_IN_Record(0, mic_buffer, sizeof(mic_buffer));
    printf("BSP_AUDIO_IN_Record returned: %ld\r\n", (long)rec_ret);

    if (rec_ret != BSP_ERROR_NONE) {
        printf("BSP_AUDIO_IN_Record FAILED ret=%ld\r\n", (long)rec_ret);
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

    printf("Recording stopped. half=%lu full=%lu err=%lu\r\n",
           (unsigned long)rec_half_done,
           (unsigned long)rec_full_done,
           (unsigned long)rec_error);

    print_stats();

    printf("MIC debug done.\r\n");
}
