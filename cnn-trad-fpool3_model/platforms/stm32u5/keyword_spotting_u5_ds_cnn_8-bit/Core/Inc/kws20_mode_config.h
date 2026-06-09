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

/* --- Offline power-measurement mode (PPK2) --------------------------------
 * KWS20_CFG_POWER_MODE = 1: the offline path runs exactly POWER_RUNS inferences
 * at a fixed PERIOD_MS, after a one-time SETTLE_MS idle, raising a marker GPIO
 * during each inference so a PPK2 digital channel delimits each inference window
 * exactly (no boot/init contamination, no manual-reset jitter).
 * Set to 0 to restore the normal 50-run latency benchmark.
 */
#define KWS20_CFG_POWER_MODE        1
#define KWS20_CFG_POWER_CONTINUOUS  1      /* loop forever for live PPK watching; 0 = one POWER_RUNS pass (official protocol) */
#define KWS20_CFG_POWER_SLEEP_IDLE  0      /* 1 = __WFI sleep between inferences (clean spikes); 0 = CPU never sleeps (run mode) */
#define KWS20_CFG_POWER_RUNS        6
#define KWS20_CFG_POWER_PERIOD_MS   5000u  /* official 6x5s protocol: 1 inference every 5 s -> spikes at t=1,6,11,16,21,26 s */
#define KWS20_CFG_POWER_SETTLE_MS   1000u  /* settle 1 s before the first inference */

/* Marker GPIO: a FREE pin broken out to a header (NOT an LED -- LED current
 * would contaminate the PPK2 trace). Default PB8 (Arduino header on
 * B-U585I-IOT02A). Relocate by editing only these three lines. */
#define KWS20_CFG_MARKER_PORT        GPIOB
#define KWS20_CFG_MARKER_PIN         GPIO_PIN_8
#define KWS20_CFG_MARKER_CLK_ENABLE  __HAL_RCC_GPIOB_CLK_ENABLE

#endif
