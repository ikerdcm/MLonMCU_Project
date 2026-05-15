#ifndef KWS20_MEASURE_H
#define KWS20_MEASURE_H

/**
 * Run the offline inference benchmark.
 * Calls ds_cnn_infer() KWS20_CFG_MEASURE_RUNS times with a zero input,
 * measures latency using the DWT cycle counter, and prints BENCH CSV lines
 * to stdout (UART via retargeted printf).
 */
void kws20_measure_run_once(void);

#endif /* KWS20_MEASURE_H */
