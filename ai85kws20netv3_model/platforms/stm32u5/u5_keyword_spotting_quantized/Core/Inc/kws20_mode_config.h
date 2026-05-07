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
#define KWS20_CFG_LIVE_MEASURE_MINIMAL_OUTPUT 0

#endif
