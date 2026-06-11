/*
 * DS-CNN-L v1 device-in-the-loop accuracy eval — CNN accelerator, INT8.
 * Host streams real test audio; the board runs frontend + CNN and returns the
 * prediction so the host can build an accuracy + confusion matrix. See
 * kws20_eval.h for the wire protocol and tools/eval_accuracy_max.py for the host.
 */

#include "kws20_eval.h"
#include "kws20_mode_config.h"
#include "ds_cnn_frontend.h"
#include "cnn_inference.h"
#include "cnn.h"   /* cnn_time (inference µs, set by the CNN ISR) */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "mxc.h"
#include "uart.h"
#include "board.h"

#define EVAL_WINDOW        16000u   /* 1 s @ 16 kHz audio window */
#define EVAL_INPUT_ELEMS   490u     /* 49 frames x 10 MFCC bins */

static int16_t eval_audio[EVAL_WINDOW];
static float   eval_mfcc[EVAL_INPUT_ELEMS];

/* Blocking read of one byte from the console UART. */
static int eval_read_byte(mxc_uart_regs_t *uart)
{
    int c;
    do {
        c = MXC_UART_ReadCharacter(uart);
    } while (c < 0);
    return c & 0xFF;
}

/* Read an ASCII line (CR stripped, '\n'-terminated) into buf. */
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

        /* Expect "EVAL <idx> <nsamples>"; ignore anything else (resync). */
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

        /* Read n int16 little-endian straight into the audio buffer (the
           MAX78000 is little-endian, so the raw byte order is the int16 order). */
        uint8_t *p = (uint8_t *)eval_audio;
        for (uint32_t b = 0; b < EVAL_WINDOW * 2u; b++) {
            p[b] = (uint8_t)eval_read_byte(uart);
        }

        memset(eval_mfcc, 0, sizeof(eval_mfcc));
        ds_cnn_frontend_compute(eval_audio, EVAL_WINDOW, eval_mfcc, EVAL_INPUT_ELEMS);
        int pred = cnn_infer(eval_mfcc, NULL);
        unsigned long cnn_us = (unsigned long)cnn_time;   /* accelerator µs */

        printf("BENCH,event=eval,idx=%lu,pred_idx=%d,cnn_us=%lu\r\n",
               idx, pred, cnn_us);
    }
}
