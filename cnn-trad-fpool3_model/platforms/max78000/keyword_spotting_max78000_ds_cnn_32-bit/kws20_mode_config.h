#ifndef KWS20_MODE_CONFIG_H
#define KWS20_MODE_CONFIG_H

/*
 * KWS20_CFG_ENABLE_MEASURE = 0  → live demo mode (no BENCH output)
 * KWS20_CFG_ENABLE_MEASURE = 1, KWS20_CFG_MEASURE_LIVE = 0 → offline benchmark
 * KWS20_CFG_ENABLE_MEASURE = 1, KWS20_CFG_MEASURE_LIVE = 1 → live benchmark with BENCH CSV
 */
#define KWS20_CFG_ENABLE_MEASURE 1
#define KWS20_CFG_MEASURE_LIVE   0
#define KWS20_CFG_MEASURE_RUNS   1

/* Device-in-the-loop accuracy eval mode (overrides measure/live when 1).
 * Host streams test audio over UART; board runs frontend+CNN and returns
 * predictions + cnn_us. See kws20_eval.h + tools/eval_accuracy_max.py. */
#define KWS20_CFG_ENABLE_EVAL    0

#endif /* KWS20_MODE_CONFIG_H */
