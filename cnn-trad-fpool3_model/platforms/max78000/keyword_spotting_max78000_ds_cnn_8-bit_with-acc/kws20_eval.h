#ifndef KWS20_EVAL_H
#define KWS20_EVAL_H

/* Device-in-the-loop accuracy eval. Receives 16 kHz int16 audio clips streamed
   over the console UART, runs the SAME frontend (ds_cnn_frontend_compute) + CNN
   the live path uses, and reports each prediction for the host to tally into an
   accuracy + confusion matrix (see tools/eval_accuracy_max.py). Never returns.

   Protocol (host -> board, 115200), one clip at a time:
     host : "EVAL <idx> <nsamples>\n"  then  nsamples * int16 little-endian
     board: "BENCH,event=eval,idx=<idx>,pred_idx=<p>\r\n"
*/
void kws20_eval_run_once(void);

#endif /* KWS20_EVAL_H */
