/*
 * DS-CNN-L v0 (32-bit float CPU) device-in-the-loop accuracy eval.
 * Same wire protocol as the v1 (accelerator) eval; runs the float ds_cnn_infer
 * and times it with a SW timer. See kws20_eval.h + tools/eval_accuracy_max.py.
 */

#include "kws20_eval.h"
#include "kws20_mode_config.h"
#include "ds_cnn_frontend.h"
#include "ds_cnn_inference.h"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "mxc.h"
#include "tmr.h"
#include "uart.h"
#include "board.h"

#define EVAL_WINDOW        16000u   /* 1 s @ 16 kHz audio window */
#define EVAL_INPUT_ELEMS   490u     /* 49 frames x 10 MFCC bins */
#define EVAL_OUT_CLASSES   12u

static int16_t eval_audio[EVAL_WINDOW];
static float   eval_mfcc[EVAL_INPUT_ELEMS];
static float   eval_scores[EVAL_OUT_CLASSES];

static int eval_read_byte(mxc_uart_regs_t *uart)
{
    int c;
    do {
        c = MXC_UART_ReadCharacter(uart);
    } while (c < 0);
    return c & 0xFF;
}

static void eval_read_line(mxc_uart_regs_t *uart, char *buf, int len)
{
    int i = 0;
    for (;;) {
        int c = eval_read_byte(uart);
        if (c == '\n') break;
        if (c == '\r') continue;
        if (i < len - 1) buf[i++] = (char)c;
    }
    buf[i] = '\0';
}

void kws20_eval_run_once(void)
{
    mxc_uart_regs_t *uart = MXC_UART_GET_UART(CONSOLE_UART);
    char hdr[64];

    printf("BENCH,event=eval_ready,classes=12,window=%u\r\n", (unsigned)EVAL_WINDOW);

    for (;;) {
        eval_read_line(uart, hdr, sizeof(hdr));

        if (strncmp(hdr, "EVAL ", 5) != 0) {
            continue;
        }
        char *q = hdr + 5;
        unsigned long idx = strtoul(q, &q, 10);
        unsigned long n   = strtoul(q, &q, 10);
        if (n != EVAL_WINDOW) {
            printf("BENCH,event=eval_error,idx=%lu,reason=nsamples\r\n", idx);
            continue;
        }

        /* Read n int16 little-endian straight into the audio buffer
           (MAX78000 is little-endian). */
        uint8_t *p = (uint8_t *)eval_audio;
        for (uint32_t b = 0; b < EVAL_WINDOW * 2u; b++) {
            p[b] = (uint8_t)eval_read_byte(uart);
        }

        memset(eval_mfcc, 0, sizeof(eval_mfcc));
        ds_cnn_frontend_compute(eval_audio, EVAL_WINDOW, eval_mfcc, EVAL_INPUT_ELEMS);

        MXC_TMR_SW_Start(MXC_TMR0);
        int pred = ds_cnn_infer(eval_mfcc, eval_scores);
        unsigned long cnn_us = (unsigned long)MXC_TMR_SW_Stop(MXC_TMR0);

        printf("BENCH,event=eval,idx=%lu,pred_idx=%d,cnn_us=%lu\r\n",
               idx, pred, cnn_us);
    }
}
