#include "main.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "../../Drivers/BSP/B-U585I-IOT02A/b_u585i_iot02a_audio.h"

#define ST_AUDIO_BYTES        (16u * 1024u)
#define ST_AUDIO_SAMPLES      (ST_AUDIO_BYTES / 2u)
#define ST_AUDIO_CHUNKS       16u
#define ST_AUDIO_CHUNK_SAMPLES (ST_AUDIO_SAMPLES / ST_AUDIO_CHUNKS)

static int16_t st_audio_buffer[ST_AUDIO_SAMPLES];

static int32_t abs32_local(int32_t x)
{
    return (x < 0) ? -x : x;
}

static void print_chunk_energy(const int16_t *samples, uint32_t sample_count)
{
    printf("Chunk energy, int16 PCM interpretation:\r\n");

    for (uint32_t c = 0; c < ST_AUDIO_CHUNKS; c++)
    {
        uint32_t start = c * ST_AUDIO_CHUNK_SAMPLES;
        uint32_t end = start + ST_AUDIO_CHUNK_SAMPLES;

        int64_t sum = 0;
        int16_t min_v = samples[start];
        int16_t max_v = samples[start];

        for (uint32_t i = start; i < end; i++)
        {
            int16_t v = samples[i];
            sum += v;

            if (v < min_v) min_v = v;
            if (v > max_v) max_v = v;
        }

        int32_t mean = (int32_t)(sum / (int64_t)ST_AUDIO_CHUNK_SAMPLES);

        uint64_t sum_abs_centered = 0;
        uint32_t max_abs_centered = 0;

        for (uint32_t i = start; i < end; i++)
        {
            int32_t centered = (int32_t)samples[i] - mean;
            uint32_t a = (uint32_t)abs32_local(centered);

            sum_abs_centered += a;
            if (a > max_abs_centered) max_abs_centered = a;
        }

        uint32_t avg_abs_centered = (uint32_t)(sum_abs_centered / ST_AUDIO_CHUNK_SAMPLES);

        printf("chunk %02lu: min=%ld max=%ld mean=%ld avg_abs_centered=%lu max_abs_centered=%lu\r\n",
               (unsigned long)c,
               (long)min_v,
               (long)max_v,
               (long)mean,
               (unsigned long)avg_abs_centered,
               (unsigned long)max_abs_centered);
    }
}

static void print_first_samples(const int16_t *samples)
{
    printf("first 64 int16 samples:\r\n");

    for (uint32_t i = 0; i < 64u; i++)
    {
        printf("%ld", (long)samples[i]);

        if (i != 63u)
        {
            printf(", ");
        }
    }

    printf("\r\n");
}

void audio_st_minimal_run_once(void)
{
    BSP_AUDIO_Init_t AudioInit;
    int32_t ret;

    printf("\r\n==============================\r\n");
    printf("MINIMAL ST BSP AUDIO RECORD\r\n");
    printf("==============================\r\n");
    printf("MIC1, 11025 Hz, 16-bit, mono, 16 KiB buffer, 3000 ms\r\n");
    printf("Plan: first run silence, second run speak loudly.\r\n");

    memset(st_audio_buffer, 0, sizeof(st_audio_buffer));

    AudioInit.Device        = AUDIO_IN_DEVICE_DIGITAL_MIC1;
    AudioInit.SampleRate    = AUDIO_FREQUENCY_11K;
    AudioInit.BitsPerSample = AUDIO_RESOLUTION_16B;
    AudioInit.ChannelsNbr   = 1;
    AudioInit.Volume        = 100;

    ret = BSP_AUDIO_IN_Init(0, &AudioInit);
    printf("BSP_AUDIO_IN_Init returned: %ld\r\n", (long)ret);
    if (ret != BSP_ERROR_NONE)
    {
        printf("AUDIO INIT FAILED\r\n");
        return;
    }

    printf("Calling BSP_AUDIO_IN_Record...\r\n");

    ret = BSP_AUDIO_IN_Record(0, (uint8_t *)st_audio_buffer, ST_AUDIO_BYTES);
    printf("BSP_AUDIO_IN_Record returned: %ld\r\n", (long)ret);
    if (ret != BSP_ERROR_NONE)
    {
        printf("AUDIO RECORD FAILED\r\n");
        BSP_AUDIO_IN_DeInit(0);
        return;
    }

    printf("Recording now for 3000 ms...\r\n");
    HAL_Delay(3000);

    ret = BSP_AUDIO_IN_Stop(0);
    printf("BSP_AUDIO_IN_Stop returned: %ld\r\n", (long)ret);

    ret = BSP_AUDIO_IN_DeInit(0);
    printf("BSP_AUDIO_IN_DeInit returned: %ld\r\n", (long)ret);

    print_first_samples(st_audio_buffer);
    print_chunk_energy(st_audio_buffer, ST_AUDIO_SAMPLES);

    printf("MINIMAL ST BSP AUDIO RECORD DONE\r\n");
}
