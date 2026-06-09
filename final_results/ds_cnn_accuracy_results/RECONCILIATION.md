# Reconciliation note — DS-CNN-L accuracy attributions

Verified on branch `DS-CNN_Optimization_Iker` (2026-06-06) with
`training/eval_quantized_model.py` on the Speech Commands test set (n=4890).
The accuracy **numbers** in `README.md` / `platform_accuracy_mapping.csv` look
correct, but the **"Model File" attributions are mislabeled** — corrected here.

## Verified eval (this branch)
| File | input | input scale/zp | test acc |
|---|---|---|---|
| `ds_cnn_l.tflite` | int8 | 0.5847 / 83 | **92.7%** (4533/4890) |
| `ds_cnn_l_static_v2.tflite` | int8 | 0.5847 / 83 | 92.0% (4497/4890) |
| `ds_cnn_l_float.tflite` | float32 | — | 92.4% (4516/4890) |
| `ds_cnn_l_static.tflite` | int8 | **0.0368 / −9** | **23.6%** (1155/4890) 🚩 broken |

## Corrections
- The mapping's **INT8 92.72%** is **`ds_cnn_l.tflite`** (confirmed: 4533 ≈ their
  4534 / 4890), **not `ds_cnn_l_static.tflite`**. The latter has a different input
  scale (0.0368 vs 0.5847) and is the **broken 4-block / random-calibration** model
  = **23.6%** (see `cnn-trad-fpool3_model/DS-CNN_session_coral_v2.html`).
- **Coral does not deploy 92.72% today**: the measured 1.81 ms run used
  `ds_cnn_l_static_edgetpu.tflite` (compiled from the broken file). The correct
  real-calibration model is `ds_cnn_l_static_v2_edgetpu.tflite` (≈92.0%) — a latency
  re-measure with it is pending.
- The **float32 93.03%** source is *unverified on this branch* (`ds_cnn_l.tflite` is
  int8, not float, and gives 92.7%); left untouched rather than guessed.

## Per-board caveat
`MAX78000` (ai8x) and `STM32U5` (X-CUBE-AI) quantize **independently** of these
TFLite files, so a single 92.72% across all boards is an *assumption*, not a
per-board on-device measurement. See `Experiments/RESULTS_LEDGER.md` (footnote ᵍ)
and `cnn-trad-fpool3_model/models/MODELS.md`.
