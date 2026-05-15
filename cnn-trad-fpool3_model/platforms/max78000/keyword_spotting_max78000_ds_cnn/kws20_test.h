#ifndef KWS20_TEST_H
#define KWS20_TEST_H

/**
 * Run the offline correctness test.
 * Feeds the MFCC of "left" into ds_cnn_infer() and prints [PASS] / [FAIL].
 */
void kws20_test_run_once(void);

#endif /* KWS20_TEST_H */
