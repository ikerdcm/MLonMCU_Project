/**
 * DS-CNN-L keyword spotting — MAX78000 FTHR, v1 (CNN accelerator, INT8).
 * MFCC computed in software on Cortex-M4; inference on CNN accelerator.
 */

#include <stdio.h>
#include <stdint.h>

#include "mxc.h"
#include "mxc_device.h"
#include "mxc_sys.h"
#include "icc.h"
#include "uart.h"
#include "board.h"
#include "led.h"

#include "cnn.h"
#include "kws20_mode_config.h"
#include "kws20_test.h"
#include "kws20_measure.h"
#include "kws20_live.h"
#include "kws20_eval.h"

/* Required by cnn.h (CNN ISR uses this to store inference time in µs) */
volatile uint32_t cnn_time;

#define CON_BAUD 115200

static int console_uart_init(uint32_t baud)
{
    mxc_uart_regs_t *con_uart = MXC_UART_GET_UART(CONSOLE_UART);

    NVIC_ClearPendingIRQ(MXC_UART_GET_IRQ(CONSOLE_UART));
    NVIC_DisableIRQ(MXC_UART_GET_IRQ(CONSOLE_UART));
    NVIC_SetPriority(MXC_UART_GET_IRQ(CONSOLE_UART), 1);
    NVIC_EnableIRQ(MXC_UART_GET_IRQ(CONSOLE_UART));

    return MXC_UART_Init(con_uart, baud, MXC_UART_IBRO_CLK);
}

int main(void)
{
#if defined(BOARD_FTHR_REVA)
    MXC_Delay(200000);
#endif

    MXC_ICC_Enable(MXC_ICC0);

    MXC_SYS_ClockSourceEnable(MXC_SYS_CLOCK_IPO);
    MXC_SYS_Clock_Select(MXC_SYS_CLOCK_IPO);
    SystemCoreClockUpdate();

    if (console_uart_init(CON_BAUD) != E_NO_ERROR)
        while (1) {}

    printf("\r\n\r\n");
    printf("==============================\r\n");
    printf("DS-CNN-L KWS — MAX78000 FTHR\r\n");
    printf("v1: CNN accelerator, INT8\r\n");
    printf("HCLK = %lu Hz\r\n", (unsigned long)SystemCoreClock);
    printf("==============================\r\n");

    /* One-time CNN accelerator initialisation */
    cnn_enable(MXC_S_GCR_PCLKDIV_CNNCLKSEL_PCLK, MXC_S_GCR_PCLKDIV_CNNCLKDIV_DIV1);
    cnn_init();
    cnn_load_weights();
    cnn_load_bias();
    cnn_configure();

#if defined(KWS20_CFG_ENABLE_EVAL) && KWS20_CFG_ENABLE_EVAL
    kws20_eval_run_once();   /* device-in-the-loop accuracy eval; never returns */
#elif KWS20_CFG_ENABLE_MEASURE
#if KWS20_CFG_MEASURE_LIVE
    kws20_live_run_once();   /* never returns */
#else
#if !defined(KWS20_CFG_ENABLE_TEST) || KWS20_CFG_ENABLE_TEST
    kws20_test_run_once();
#endif
    kws20_measure_run_once();
    printf("\r\nDone. Spinning.\r\n");
    while (1) {}
#endif
#else
    kws20_live_run_once();   /* never returns */
#endif
}
