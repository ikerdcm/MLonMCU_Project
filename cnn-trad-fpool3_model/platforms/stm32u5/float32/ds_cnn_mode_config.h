#ifndef DS_CNN_MODE_CONFIG_H
#define DS_CNN_MODE_CONFIG_H

/*
 * DS-CNN-L measurement mode selection
 *
 * DS_CNN_CFG_ENABLE_MEASURE = 1, DS_CNN_CFG_MEASURE_LIVE = 0
 *   Offline measurement: runs inference on the fixed MFCC test tensor,
 *   prints BENCH CSV lines over UART. Use this for latency benchmarking.
 *
 * DS_CNN_CFG_ENABLE_MEASURE = 0
 *   No benchmark output (placeholder for future live-audio path).
 */
#define DS_CNN_CFG_ENABLE_MEASURE 1
#define DS_CNN_CFG_MEASURE_LIVE   0

#endif
