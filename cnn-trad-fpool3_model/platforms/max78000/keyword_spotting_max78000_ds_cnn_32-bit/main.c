/**
 * DS-CNN-L keyword spotting — MAX78000 FTHR (Cortex-M4, no CNN accelerator).
 * Phase 1: CPU-only float32 inference benchmark.
 */

#include <stdio.h>
#include <stdint.h>

#include "mxc.h"
#include "mxc_device.h"
#include "mxc_sys.h"
#include "icc.h"
#include "uart.h"
#include "board.h"

#include "kws20_mode_config.h"
#include "kws20_test.h"
#include "kws20_measure.h"
#include "kws20_live.h"
#include "kws20_eval.h"

#define CON_BAUD 115200

static int console_uart_init(uint32_t baud)
{
    mxc_uart_regs_t *con_uart = MXC_UART_GET_UART(CONSOLE_UART);
    int err;

    NVIC_ClearPendingIRQ(MXC_UART_GET_IRQ(CONSOLE_UART));
    NVIC_DisableIRQ(MXC_UART_GET_IRQ(CONSOLE_UART));
    NVIC_SetPriority(MXC_UART_GET_IRQ(CONSOLE_UART), 1);
    NVIC_EnableIRQ(MXC_UART_GET_IRQ(CONSOLE_UART));

    err = MXC_UART_Init(con_uart, baud, MXC_UART_IBRO_CLK);
    return err;
}

int main(void)
{
#if defined(BOARD_FTHR_REVA)
    /* Wait for PMIC 1.8V rail — ~180 ms after power-up */
    MXC_Delay(200000);
#endif

    /* Enable instruction cache */
    MXC_ICC_Enable(MXC_ICC0);

    /* Switch to IPO (100 MHz) for best CPU performance */
    MXC_SYS_ClockSourceEnable(MXC_SYS_CLOCK_IPO);
    MXC_SYS_Clock_Select(MXC_SYS_CLOCK_IPO);
    SystemCoreClockUpdate();

    /* Init UART for printf */
    if (console_uart_init(CON_BAUD) != E_NO_ERROR) {
        /* If UART fails we cannot print — just spin */
        while (1) {}
    }

    printf("\r\n\r\n");
    printf("==============================\r\n");
    printf("DS-CNN-L KWS — MAX78000 FTHR\r\n");
    printf("CPU-only (Phase 1), float32\r\n");
    printf("HCLK = %lu Hz\r\n", (unsigned long)SystemCoreClock);
    printf("==============================\r\n");

#if defined(KWS20_CFG_ENABLE_EVAL) && KWS20_CFG_ENABLE_EVAL
    kws20_eval_run_once();   /* device-in-the-loop accuracy eval; never returns */
#elif KWS20_CFG_ENABLE_MEASURE
#if KWS20_CFG_MEASURE_LIVE
    kws20_live_run_once();   /* never returns */
#else
    /* Correctness test: MFCC of "left" */
    kws20_test_run_once();
    /* Latency benchmark */
    kws20_measure_run_once();
    printf("\r\nDone. Spinning.\r\n");
    while (1) {}
#endif
#else
    kws20_live_run_once();   /* never returns */
#endif
}
