#ifndef KWS20_MODE_CONFIG_H
#define KWS20_MODE_CONFIG_H

/* 0 = normal demo, 1 = measurement-oriented firmware output */
#define KWS20_CFG_ENABLE_MEASURE 1

/* 0 = offline / fixed test vector measurement, 1 = live microphone measurement */
#define KWS20_CFG_MEASURE_LIVE   1

/* 1 = suppress noisy live debug prints, keep BENCH + prediction output */
#define KWS20_CFG_MINIMAL_OUTPUT 1

/* 1 = emit full AI input dumps, 0 = keep dump logic compiled out */
#define KWS20_CFG_ENABLE_AI_INPUT_DUMP 0

/* 0 = microphone live path, 1 = offline fixed-vector path */
#define KWS20_CFG_FORCE_OFFLINE 0

#endif
