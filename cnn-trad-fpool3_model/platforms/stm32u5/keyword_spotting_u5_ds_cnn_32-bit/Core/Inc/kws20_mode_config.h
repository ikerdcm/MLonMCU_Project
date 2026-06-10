#ifndef KWS20_MODE_CONFIG_H
#define KWS20_MODE_CONFIG_H

/*
 * Measurement mode selection
 *
 * KWS20_CFG_ENABLE_MEASURE = 0
 *   Normal live app path, no BENCH output.
 *
 * KWS20_CFG_ENABLE_MEASURE = 1 and KWS20_CFG_MEASURE_LIVE = 0
 *   Offline measurement mode using the fixed test tensor.
 *
 * KWS20_CFG_ENABLE_MEASURE = 1 and KWS20_CFG_MEASURE_LIVE = 1
 *   Live measurement mode using microphone capture plus BENCH output.
 *
 * KWS20_CFG_LIVE_MEASURE_MINIMAL_OUTPUT = 1
 *   In live measurement mode, suppress non-essential status/prediction prints
 *   so the UART stream is mostly BENCH lines plus real errors.
 */
#define KWS20_CFG_APP_MODE_KWS_LIVE       0
#define KWS20_CFG_APP_MODE_MIC2_POLL      1
#define KWS20_CFG_APP_MODE_AUDIO_HW_PROBE 2
#define KWS20_CFG_APP_MODE_AUDIO_MINIMAL  3

#define KWS20_CFG_APP_MODE KWS20_CFG_APP_MODE_KWS_LIVE

#define KWS20_CFG_ENABLE_MEASURE 1
#define KWS20_CFG_MEASURE_LIVE   0
#define KWS20_CFG_LIVE_MEASURE_MINIMAL_OUTPUT 1
#define KWS20_CFG_MEASURE_RUNS 50

/* Device-in-the-loop accuracy eval mode (overrides measure/live when 1).
 * Host streams test audio over huart1; board runs frontend+model and returns
 * predictions + cnn_us. See testbench.py. */
#define KWS20_CFG_ENABLE_EVAL    1

/* --- Offline power-measurement mode (PPK2) --------------------------------
 * KWS20_CFG_POWER_MODE = 1: the offline path runs exactly POWER_RUNS inferences
 * at a fixed PERIOD_MS, after a one-time SETTLE_MS idle, so the trace contains
 * evenly spaced inference spikes.
 * Set to 0 to restore the normal 50-run latency benchmark.
 */
#define KWS20_CFG_POWER_MODE        1
#define KWS20_CFG_POWER_CONTINUOUS  1      /* loop forever for live PPK watching; 0 = one POWER_RUNS pass */
#define KWS20_CFG_POWER_SLEEP_IDLE  0      /* 1 = __WFI sleep between inferences; 0 = CPU never sleeps (run mode) */
#define KWS20_CFG_POWER_RUNS        6
#define KWS20_CFG_POWER_PERIOD_MS   5000u  /* 1 inference every 5 s */
#define KWS20_CFG_POWER_SETTLE_MS   1000u  /* settle 1 s before the first inference */

#endif
