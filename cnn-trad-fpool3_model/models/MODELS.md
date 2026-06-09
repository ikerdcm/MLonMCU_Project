# DS-CNN-L model artifacts — versioning & lineage

Versioning matches the per-board experiment convention: **v0 = float baseline,
v1 = the quantized (INT8) model** — i.e. the v0→v1 change *is just quantization*
(same as MAX78000 / STM32U5, where the INT8 model is v1).

The INT8 (v1) step has two cuts. The first was broken; keep it as the
before/after evidence, **do not deploy it**. The `_v2` suffix is a *re-quantization
fix of v1, not a new optimization stage.*

## Float (v0)
| File | Notes |
|---|---|
| `../training/trained_models/ds_cnn_l_finetuned.keras` | FP32 training source — 6-block, 95.98% val / **92.7% test** |
| `ds_cnn_l_float.tflite` | FP32 static TFLite (→ Coral M7 CPU bench `kws_*_cpu`) — **92.4% test** |
| `ds_cnn_l.onnx` | FP32 ONNX (→ STM32U5 X-CUBE-AI) |

## INT8 (v1 = quantization)
| File | Cut | Arch | Calibration | Test acc | Status |
|---|---|---|---|---|---|
| `ds_cnn_l_static.tflite` + `../platforms/coral/model/ds_cnn_l_static_edgetpu.tflite` | **v1.0 (broken)** | 4-block | random | **23.6%** 🚩 | keep as before/after; **do not deploy** |
| `ds_cnn_l_static_v2.tflite` + `../platforms/coral/model/ds_cnn_l_static_v2_edgetpu.tflite` | **v1 (canonical)** | 6-block | real MFCC | **92.0%** ✓ | the real INT8 — deploy this |
| `ds_cnn_l.tflite` | source | 6-block | real MFCC | — | INT8 *dynamic* batch; source for the static export |

> ⚠️ The measured "int8-accel 1.81 ms" run used the **broken v1.0** edgetpu model
> (the apps still load `ds_cnn_l_static_edgetpu.tflite`). To make the canonical v1
> the deployed/headline INT8, point `kws_bench`/`kws_live` at the `_v2_edgetpu`
> model and re-measure (latency ≈ unchanged, accuracy 23.6% → 92.0%).

See [DS-CNN_session_coral_v2.html](../DS-CNN_session_coral_v2.html) for the fix
details and `Experiments/RESULTS_LEDGER.md` (footnote ᵍ) for the numbers.
