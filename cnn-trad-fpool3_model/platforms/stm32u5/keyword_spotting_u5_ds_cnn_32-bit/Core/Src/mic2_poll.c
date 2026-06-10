#include "main.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "../../Drivers/BSP/B-U585I-IOT02A/b_u585i_iot02a_audio.h"

#define MIC2_POLL_BYTES        (8u * 1024u)
#define MIC2_POLL_SAMPLES      (MIC2_POLL_BYTES / 2u)
#define MIC2_POLL_MS           250u

#define MIC2_THRESHOLD         15u
#define MIC2_PRINT_EVERY_ROUND 1u

static int16_t mic2_poll_buffer[MIC2_POLL_SAMPLES];

static int32_t abs32_local(int32_t x)
{
    return (x < 0) ? -x : x;
}

static uint32_t mic2_compute_loudness(const int16_t *samples,
                                      uint32_t sample_count,
                                      int16_t *out_min,
                                      int16_t *out_max,
                                      int32_t *out_mean,
                                      uint32_t *out_max_abs_centered)
{
    int64_t sum = 0;
    int16_t min_v = samples[0];
    int16_t max_v = samples[0];

    for (uint32_t i = 0; i < sample_count; i++)
    {
        int16_t v = samples[i];

        if (v < min_v) min_v = v;
        if (v > max_v) max_v = v;

        sum += v;
    }

    int32_t mean = (int32_t)(sum / (int64_t)sample_count);

    uint64_t sum_abs_centered = 0;
    uint32_t max_abs_centered = 0;

    for (uint32_t i = 0; i < sample_count; i++)
    {
        int32_t centered = (int32_t)samples[i] - mean;
        uint32_t a = (uint32_t)abs32_local(centered);

        sum_abs_centered += a;

        if (a > max_abs_centered)
        {
            max_abs_centered = a;
        }
    }

    if (out_min != NULL) *out_min = min_v;
    if (out_max != NULL) *out_max = max_v;
    if (out_mean != NULL) *out_mean = mean;
    if (out_max_abs_centered != NULL) *out_max_abs_centered = max_abs_centered;

    return (uint32_t)(sum_abs_centered / (uint64_t)sample_count);
}

static int mic2_capture_once(uint32_t *out_avg_abs,
                             uint32_t *out_max_abs,
                             int16_t *out_min,
                             int16_t *out_max,
                             int32_t *out_mean)
{
    BSP_AUDIO_Init_t AudioInit;
    int32_t ret;

    memset(mic2_poll_buffer, 0, sizeof(mic2_poll_buffer));

    AudioInit.Device        = AUDIO_IN_DEVICE_DIGITAL_MIC2;
    AudioInit.SampleRate    = AUDIO_FREQUENCY_16K;
    AudioInit.BitsPerSample = AUDIO_RESOLUTION_16B;
    AudioInit.ChannelsNbr   = 1;
    AudioInit.Volume        = 100;

    ret = BSP_AUDIO_IN_Init(0, &AudioInit);
    if (ret != BSP_ERROR_NONE)
    {
        printf("BSP_AUDIO_IN_Init MIC2 failed: %ld\r\n", (long)ret);
        return -1;
    }

    ret = BSP_AUDIO_IN_Record(0, (uint8_t *)mic2_poll_buffer, MIC2_POLL_BYTES);
    if (ret != BSP_ERROR_NONE)
    {
        printf("BSP_AUDIO_IN_Record MIC2 failed: %ld\r\n", (long)ret);
        BSP_AUDIO_IN_DeInit(0);
        return -1;
    }

    HAL_Delay(MIC2_POLL_MS);

    ret = BSP_AUDIO_IN_Stop(0);
    if (ret != BSP_ERROR_NONE)
    {
        printf("BSP_AUDIO_IN_Stop MIC2 failed: %ld\r\n", (long)ret);
    }

    ret = BSP_AUDIO_IN_DeInit(0);
    if (ret != BSP_ERROR_NONE)
    {
        printf("BSP_AUDIO_IN_DeInit MIC2 failed: %ld\r\n", (long)ret);
    }

    uint32_t max_abs = 0;
    uint32_t avg_abs = mic2_compute_loudness(mic2_poll_buffer,
                                             MIC2_POLL_SAMPLES,
                                             out_min,
                                             out_max,
                                             out_mean,
                                             &max_abs);

    if (out_avg_abs != NULL) *out_avg_abs = avg_abs;
    if (out_max_abs != NULL) *out_max_abs = max_abs;

    return 0;
}

void mic2_poll_run_forever(void)
{
    uint32_t round = 0;

    printf("\r\n==============================\r\n");
    printf("MIC2 POLLING LOUDNESS TEST\r\n");
    printf("==============================\r\n");
    printf("Path: DIGITAL_MIC2 / MDF1_Filter0 / PF10 CLK / PB1 DATA\r\n");
    printf("Window: %lu ms, threshold: %lu\r\n",
           (unsigned long)MIC2_POLL_MS,
           (unsigned long)MIC2_THRESHOLD);
    printf("Quiet should be low, loud speech/clap should cross threshold.\r\n");

    while (1)
    {
        uint32_t avg_abs = 0;
        uint32_t max_abs = 0;
        int16_t min_v = 0;
        int16_t max_v = 0;
        int32_t mean = 0;

        round++;

        if (mic2_capture_once(&avg_abs, &max_abs, &min_v, &max_v, &mean) != 0)
        {
            printf("round %lu capture failed\r\n", (unsigned long)round);
            HAL_Delay(500);
            continue;
        }

#if MIC2_PRINT_EVERY_ROUND
        printf("round=%lu loudness=%lu max_abs=%lu min=%ld max=%ld mean=%ld",
               (unsigned long)round,
               (unsigned long)avg_abs,
               (unsigned long)max_abs,
               (long)min_v,
               (long)max_v,
               (long)mean);
#endif

        if (avg_abs >= MIC2_THRESHOLD)
        {
            printf("  >>> SOUND DETECTED <<<");
        }

        printf("\r\n");

        HAL_Delay(250);
    }
}
