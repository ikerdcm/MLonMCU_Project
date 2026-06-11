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

/*
 * Idle-window power policy (only used when MEASURE_RUNS == 1).
 *   0 → CNN stays powered/configured across the LPM idle window.
 *       Clean per-inference peak (no weight reload in the measured spike),
 *       but the powered CNN array pins the idle floor high (~16 mA observed).
 *   1 → CNN is fully powered down (cnn_disable) during the idle window and
 *       reloaded right before each inference. Realistic low idle floor /
 *       battery-life number, but the measured spike now includes the weight
 *       reload, so it is NOT a clean inference-only peak.
 * Use 0 for energy-per-inference, 1 for idle/standby power. */
#define KWS20_CFG_DEEP_IDLE      1

/* Device-in-the-loop accuracy eval mode (overrides measure/live when 1).
 * Host streams test audio over UART; board runs frontend+CNN and returns
 * predictions. See kws20_eval.h + tools/eval_accuracy_max.py. */
#define KWS20_CFG_ENABLE_EVAL    0
#define KWS20_CFG_ENABLE_TEST    0

#endif /* KWS20_MODE_CONFIG_H */
