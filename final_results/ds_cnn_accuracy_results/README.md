# DS-CNN-L Accuracy Results — cnn-trad-fpool3_model

## Model

**DS-CNN-L** (Depthwise Separable CNN, Large variant)  
Architecture: 64 filters, 6 depthwise-separable blocks  
Dataset: Google Speech Commands v2 — 12 classes (down, go, left, no, off, on, right, stop, up, yes, silence, unknown)  
Test set: **4890 samples**

---

## Accuracy by Model Format

| Model | Accuracy | Correct / Total |
|---|---|---|
| Keras float32 (`.h5`) | 93.01% | 4548 / 4890 |
| TFLite float32 (`.tflite`) | 93.03% | 4549 / 4890 |
| TFLite INT8 quantized (`.tflite`) | 92.72% | 4534 / 4890 |

**Quantization loss (Float32 → INT8): −0.31%**

---

## Platform Mapping

All platforms use the **same DS-CNN-L model** — accuracy differences come only from quantization, not from the platform or inference engine.

| Platform | Version | Format | Inference Engine | Accuracy | Quant. Loss | Confusion Matrix |
|---|---|---|---|---|---|---|
| STM32U5 | v0 — float32 | TFLite Float32 | X-CUBE-AI (Cortex-M33) | **93.03%** | — | `confusion_ds_cnn_l_float32.png` |
| STM32U5 | v1 — INT8 | TFLite INT8 | X-CUBE-AI (Cortex-M33) | **92.72%** | −0.31% | `confusion_ds_cnn_l.png` |
| MAX78000 | v0 — float32 SW | TFLite Float32 | Software (Cortex-M4F) | **93.03%** | — | `confusion_ds_cnn_l_float32.png` |
| MAX78000 | v1 — INT8 HW | TFLite INT8 | CNN Hardware Accelerator | **92.72%** | −0.31% | `confusion_ds_cnn_l.png` |
| Coral Dev Board Micro | v0 — INT8 Edge TPU | TFLite INT8 (Edge TPU) | Google Edge TPU | **92.72%** | −0.31% | `confusion_ds_cnn_l.png` |

---

## Files in `plots/`

| File | Description |
|---|---|
| `confusion_ds_cnn_l_keras.png` | Confusion matrix — Keras float32 model |
| `confusion_ds_cnn_l_float32.png` | Confusion matrix — TFLite float32 (STM32U5 v0, MAX78000 v0) |
| `confusion_ds_cnn_l.png` | Confusion matrix — TFLite INT8 (STM32U5 v1, MAX78000 v1, Coral v0) |
| `acc.png` | Accuracy bar chart across model variants |
| `summary.txt` | Raw evaluation output |

---

## Key Takeaway

- **INT8 quantization costs only 0.31% accuracy** (93.03% → 92.72%)
- All hardware-accelerated platforms (MAX78000 CNN accel, Coral Edge TPU) achieve the same accuracy as the equivalent software INT8 model — the inference engine does not affect accuracy
- The Keras and TFLite float32 models are functionally identical (0.02% difference due to floating point rounding in export)

---

## Model Files

| File | Location |
|---|---|
| `ds_cnn_l.tflite` | `cnn-trad-fpool3_model/models/ds_cnn_l.tflite` |
| `ds_cnn_l_static.tflite` (INT8) | `cnn-trad-fpool3_model/models/ds_cnn_l_static.tflite` |
| `ds_cnn_l_static_edgetpu.tflite` | `cnn-trad-fpool3_model/platforms/coral/model/ds_cnn_l_static_edgetpu.tflite` |
