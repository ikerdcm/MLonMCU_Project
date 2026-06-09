#ifndef KWS20_MODE_CONFIG_H
#define KWS20_MODE_CONFIG_H

/* 0 = normal demo, 1 = measurement-oriented firmware output */
#define KWS20_CFG_ENABLE_MEASURE 1

/* 0 = offline / fixed test vector measurement, 1 = live microphone measurement */
#define KWS20_CFG_MEASURE_LIVE 1

/* 1 = suppress noisy live debug prints, keep BENCH + prediction output */
#define KWS20_CFG_MINIMAL_OUTPUT 1

/* 1 = emit one full signed INT8 model-input dump for live calibration */
#define KWS20_CFG_ENABLE_AI_INPUT_DUMP 0

/* 1 = print periodic microphone statistics during live acquisition */
#define KWS20_CFG_ENABLE_MIC_DEBUG 0

/* 0 = microphone live path, 1 = offline fixed-vector path */
#define KWS20_CFG_FORCE_OFFLINE 1

/*
 * Power measurement mode: sleep → infer → sleep → repeat.
 * CPU sleeps for POWER_IDLE_MS, wakes, runs POWER_BURST_INFERENCES inferences
 * on the fixed "FIVE" test vector, then sleeps again.
 * POWER_RUNS=0 means infinite loop (Nordic profiler capture).
 * This disables ENABLE_MIC_PROCESSING (see main.c line 99) and selects the
 * offline kws_five.h path, so no microphone is needed.
 */
#define KWS20_CFG_POWER_MEASURE_MODE     0
#define KWS20_CFG_POWER_IDLE_MS          2000
#define KWS20_CFG_POWER_BURST_INFERENCES 1
#define KWS20_CFG_POWER_RUNS             0

#endif
