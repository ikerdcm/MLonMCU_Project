#include "main.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "../../Drivers/BSP/B-U585I-IOT02A/b_u585i_iot02a_audio.h"

#define PROBE_AUDIO_BYTES   (16u * 1024u)
#define PROBE_AUDIO_SAMPLES (PROBE_AUDIO_BYTES / 2u)
#define PIN_SAMPLE_LOOPS    300000u

__attribute__((aligned(32)))
static int16_t probe_audio_buf[PROBE_AUDIO_SAMPLES];

static uint32_t pin_read_fast(GPIO_TypeDef *port, uint16_t pin)
{
    return ((port->IDR & pin) != 0u) ? 1u : 0u;
}

static void sample_pin_activity(const char *name, GPIO_TypeDef *port, uint16_t pin)
{
    uint32_t first = pin_read_fast(port, pin);
    uint32_t last = first;
    uint32_t ones = 0;
    uint32_t transitions = 0;

    for (volatile uint32_t i = 0; i < PIN_SAMPLE_LOOPS; i++)
    {
        uint32_t v = pin_read_fast(port, pin);

        if (v != 0u)
        {
            ones++;
        }

        if (v != last)
        {
            transitions++;
            last = v;
        }
    }

    printf("%s: first=%lu last=%lu ones=%lu/%lu transitions=%lu\r\n",
           name,
           (unsigned long)first,
           (unsigned long)last,
           (unsigned long)ones,
           (unsigned long)PIN_SAMPLE_LOOPS,
           (unsigned long)transitions);
}

static void dump_gpio_regs(const char *name, GPIO_TypeDef *port)
{
    printf("%s MODER=0x%08lX AFRL=0x%08lX AFRH=0x%08lX IDR=0x%08lX\r\n",
           name,
           (unsigned long)port->MODER,
           (unsigned long)port->AFR[0],
           (unsigned long)port->AFR[1],
           (unsigned long)port->IDR);
}

static void dump_filter_regs(const char *name, MDF_TypeDef *block, MDF_Filter_TypeDef *flt)
{
    printf("%s BLOCK GCR=0x%08lX CKGCR=0x%08lX OR=0x%08lX\r\n",
           name,
           (unsigned long)block->GCR,
           (unsigned long)block->CKGCR,
           (unsigned long)block->OR);

    printf("%s FLT SITFCR=0x%08lX BSMXCR=0x%08lX DFLTCR=0x%08lX DFLTCICR=0x%08lX DFLTRSFR=0x%08lX DFLTISR=0x%08lX\r\n",
           name,
           (unsigned long)flt->SITFCR,
           (unsigned long)flt->BSMXCR,
           (unsigned long)flt->DFLTCR,
           (unsigned long)flt->DFLTCICR,
           (unsigned long)flt->DFLTRSFR,
           (unsigned long)flt->DFLTISR);
}

static void print_audio_stats(const char *label)
{
    int16_t min_v = probe_audio_buf[0];
    int16_t max_v = probe_audio_buf[0];
    int64_t sum = 0;
    int64_t sum_abs_centered = 0;
    uint32_t sat_pos = 0;
    uint32_t sat_neg = 0;
    uint32_t zero = 0;

    for (uint32_t i = 0; i < PROBE_AUDIO_SAMPLES; i++)
    {
        int16_t v = probe_audio_buf[i];

        if (v < min_v) min_v = v;
        if (v > max_v) max_v = v;
        if (v == 32767) sat_pos++;
        if (v == -32768) sat_neg++;
        if (v == 0) zero++;

        sum += v;
    }

    int32_t mean = (int32_t)(sum / (int64_t)PROBE_AUDIO_SAMPLES);

    for (uint32_t i = 0; i < PROBE_AUDIO_SAMPLES; i++)
    {
        int32_t d = (int32_t)probe_audio_buf[i] - mean;
        if (d < 0) d = -d;
        sum_abs_centered += d;
    }

    printf("%s audio: min=%ld max=%ld mean=%ld avg_abs_centered=%lu sat+=%lu sat-=%lu zero=%lu/%lu\r\n",
           label,
           (long)min_v,
           (long)max_v,
           (long)mean,
           (unsigned long)(sum_abs_centered / (int64_t)PROBE_AUDIO_SAMPLES),
           (unsigned long)sat_pos,
           (unsigned long)sat_neg,
           (unsigned long)zero,
           (unsigned long)PROBE_AUDIO_SAMPLES);

    printf("%s first 32 int16:\r\n", label);
    for (uint32_t i = 0; i < 32u; i++)
    {
        printf("%ld", (long)probe_audio_buf[i]);
        if (i != 31u) printf(", ");
    }
    printf("\r\n");
}

static void probe_one_mic(const char *label,
                          uint32_t device,
                          GPIO_TypeDef *clk_port,
                          uint16_t clk_pin,
                          GPIO_TypeDef *data_port,
                          uint16_t data_pin,
                          MDF_TypeDef *block,
                          MDF_Filter_TypeDef *filter)
{
    BSP_AUDIO_Init_t AudioInit;
    int32_t ret;

    printf("\r\n------------------------------\r\n");
    printf("PROBE %s\r\n", label);
    printf("------------------------------\r\n");

    memset(probe_audio_buf, 0, sizeof(probe_audio_buf));

    printf("Before init pin activity:\r\n");
    sample_pin_activity("CLK before", clk_port, clk_pin);
    sample_pin_activity("DAT before", data_port, data_pin);

    AudioInit.Device        = device;
    AudioInit.SampleRate    = AUDIO_FREQUENCY_11K;
    AudioInit.BitsPerSample = AUDIO_RESOLUTION_16B;
    AudioInit.ChannelsNbr   = 1;
    AudioInit.Volume        = 100;

    ret = BSP_AUDIO_IN_Init(0, &AudioInit);
    printf("BSP_AUDIO_IN_Init returned: %ld\r\n", (long)ret);
    if (ret != BSP_ERROR_NONE)
    {
        printf("INIT FAILED for %s\r\n", label);
        return;
    }

    printf("After init registers:\r\n");
    dump_filter_regs(label, block, filter);
    dump_gpio_regs("GPIOB", GPIOB);
    dump_gpio_regs("GPIOE", GPIOE);
    dump_gpio_regs("GPIOF", GPIOF);

    printf("After init pin activity:\r\n");
    sample_pin_activity("CLK after init", clk_port, clk_pin);
    sample_pin_activity("DAT after init", data_port, data_pin);

    ret = BSP_AUDIO_IN_Record(0, (uint8_t *)probe_audio_buf, PROBE_AUDIO_BYTES);
    printf("BSP_AUDIO_IN_Record returned: %ld\r\n", (long)ret);
    if (ret != BSP_ERROR_NONE)
    {
        printf("RECORD FAILED for %s\r\n", label);
        BSP_AUDIO_IN_DeInit(0);
        return;
    }

    HAL_Delay(100);

    printf("While recording registers:\r\n");
    dump_filter_regs(label, block, filter);

    printf("While recording pin activity:\r\n");
    sample_pin_activity("CLK recording", clk_port, clk_pin);
    sample_pin_activity("DAT recording", data_port, data_pin);

    HAL_Delay(1000);

    ret = BSP_AUDIO_IN_Stop(0);
    printf("BSP_AUDIO_IN_Stop returned: %ld\r\n", (long)ret);

    ret = BSP_AUDIO_IN_DeInit(0);
    printf("BSP_AUDIO_IN_DeInit returned: %ld\r\n", (long)ret);

    print_audio_stats(label);
}

void audio_hw_probe_run_once(void)
{
    printf("\r\n==============================\r\n");
    printf("STM32U5 AUDIO HARDWARE PROBE\r\n");
    printf("==============================\r\n");
    printf("This checks if MIC clock and data pins toggle.\r\n");

    probe_one_mic("MIC1 / ADF1 / PE9 CLK / PE10 DATA",
                  AUDIO_IN_DEVICE_DIGITAL_MIC1,
                  AUDIO_ADF1_CCK0_GPIO_PORT,
                  AUDIO_ADF1_CCK0_GPIO_PIN,
                  AUDIO_ADF1_SDINx_GPIO_PORT,
                  AUDIO_ADF1_SDINx_GPIO_PIN,
                  ADF1,
                  ADF1_Filter0);

    HAL_Delay(500);

    probe_one_mic("MIC2 / MDF1 / PF10 CLK / PB1 DATA",
                  AUDIO_IN_DEVICE_DIGITAL_MIC2,
                  AUDIO_MDF1_CCK1_GPIO_PORT,
                  AUDIO_MDF1_CCK1_GPIO_PIN,
                  AUDIO_MDF1_SDIN0_GPIO_PORT,
                  AUDIO_MDF1_SDIN0_GPIO_PIN,
                  MDF1,
                  MDF1_Filter0);

    printf("\r\nAUDIO HARDWARE PROBE DONE\r\n");
}
