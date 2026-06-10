#ifndef KWS20_LIVE_H
#define KWS20_LIVE_H

#include <stdint.h>

void kws20_live_run_once(void);
void kws20_eval_run_once(void);   /* device-in-the-loop accuracy eval (testbench.py) */
int kws20_live_audio_half_callback(uint32_t instance);
int kws20_live_audio_full_callback(uint32_t instance);
int kws20_live_audio_error_callback(uint32_t instance);

#endif
