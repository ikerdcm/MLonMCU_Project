# DS-CNN cross-MCU experiment roadmap

Scope: the **DS-CNN** model (`cnn-trad-fpool3_model`) across Coral, MAX78000, STM32U5
(N6 exists only for the ai85 model so far). Two deliverables: **(A) which MCU wins at
a matched config**, and **(B) how far the champion (Coral) can be pushed**. Companion
to `cnn-trad-fpool3_model/DS-CNN_onboarding.html`.

---

## 0. Config-naming convention (read first)

The existing `Experiments/.../<model>_vN/` folders use `vN`, but **`vN` means a
different config on different boards** (Coral v0 = INT8, MAX v0 = FP32). For fair
cross-MCU comparison, identify experiments by **content**, not `vN`. Use this registry
and put it in each result folder's notes:

| Config id | Meaning | Precision | Runs on |
|---|---|---|---|
| `fp32-cpu` | float32 baseline | FP32 | Arm core (no accel) |
| `int8-accel` | INT8 on the board's accelerator | INT8 | MAX CNN-accel / Coral Edge-TPU |
| `int8-cpu` | INT8 on CPU (CMSIS-NN) | INT8 | STM32U5 |
| `int8-prune` | INT8 + **structured** pruning (smaller arch) | INT8 | accel/TPU/CPU |
| `int8-prune-distill` | above, accuracy recovered via distillation | INT8 | accel/TPU/CPU |
| `int4-cpu` | INT4 weight compression (X-CUBE-AI) | INT4 | **STM32U5 only** |

Keep folders as `<model>_<configid>` going forward (don't rename old `_vN` dirs — add a
one-line `config.txt` mapping `vN → configid`). Per the project rule, **every
optimization is a new version; never delete the old** (before/after is the whole point).

---

## 1. What's already measured (DS-CNN)

| Config | Coral | MAX78000 | STM32U5 |
|---|---|---|---|
| `fp32-cpu` | ❌ **gap** (needs M7 CPU build) | ✅ profiling | ✅ profiling+pwr |
| `int8-accel` / `int8-cpu` | ✅ profiling, ⚠️ power offline only | ✅ profiling+pwr | ✅ profiling+pwr |
| `int8-prune` | ❌ | ❌ | ❌ |
| `int8-prune-distill` | ❌ | ❌ | ❌ |

Each cell = General Profiling (latency, cycles, mem) **and** PWR Consumption
(energy/inf), in **offline** (fixed vector) and **online** (mic) variants.

Reference numbers (INT8): Coral ~1.8 ms (no energy yet) · MAX 0.16 ms / 23 µJ ·
STM32 64 ms / 4.6 mJ. FP32: MAX 2048 ms · STM32 224 ms.

---

## 2. Methodology

- **Cross-MCU (Story A): matched config.** Same `configid` on all boards → fair "which
  HW wins." This is parity, *not* stacking.
- **Within-MCU (Story B): cumulative ladder.** `fp32-cpu → int8-accel → int8-prune →
  int8-prune-distill` on Coral = one accuracy-vs-(latency/energy) tradeoff curve. This
  is the main "max out the champion" narrative; matches the cumulative `vN` convention.
- **Isolated ablation: optional, champion only.** Apply each technique alone to one
  baseline to *attribute* credit ("prune bought X ms, distill bought Y %"). Expensive —
  do for Coral only, if time. Don't do isolated across all boards (combinatorial blowup).

### Reality checks (avoid wasted experiments)
1. **Distillation is not a latency stage** — same architecture → same speed/energy. It
   only belongs *after* compression, to recover accuracy. Plot it on the **accuracy** axis.
2. **Pruning on Edge-TPU must be structured** (fewer channels/filters); unstructured
   sparsity doesn't speed up the dense INT8 TPU. So "Coral + prune" ≈ smaller arch
   (overlaps the 4-block / 32-filter idea, onboarding §7.4).
3. **INT4 ≠ Coral** — Edge-TPU is INT8-only; `int4-cpu` lives on **STM32U5** only.
4. **Don't chase accuracy.** 6-block vs 4-block was +0.3 % (15/4890) — *below* the
   ±0.4 % std-error at n=4890, i.e. noise. Choose models by latency/energy at accuracy
   parity. (So for Coral, prefer the **4-block**, not the 6-block `v2`.)

---

## 3. Prioritized next actions

1. **Coral `fp32-cpu` build + Coral `int8-accel` online power.** Completes Story A's FP32
   row *and* adds the TPU-vs-CPU curve (the most compelling single Coral result). Build
   `kws_live` without the Edge-TPU op (plain TFLite-micro on the M7); capture energy.
2. **Coral `int8-prune` (structured: 4-block / fewer filters).** The onboarding §7.4
   point, now framed as the pruning stage of Story B (~1.2 ms, lower energy, ≈parity acc).
3. **`int8-prune-distill` on Coral.** Distill from the 6-block (or FP32) teacher into the
   pruned student to recover any accuracy lost in step 2. Top of the Coral ladder.
4. *(optional)* Isolated ablation on Coral; **`int4-cpu` on STM32U5** (from the 6-block,
   which has more redundancy to tolerate 4-bit) vs its `int8-cpu`.

### Metrics to record every run
latency (cnn_us, p50/p95) · energy/inference (µJ) · INT8 test accuracy
(`training/eval_quantized_model.py`) · flash + SRAM/on-chip memory · offline & online.

---

## 4. Where things live
- Results: `Experiments/{General Profiling,PWR Consumption}/<Board>/<model>_<configid>/{offline,online}/`
- Power analysis tool: `Experiments/PWR Consumption/tools/analyze_power_peaks.py`
- DS-CNN training/quant/eval: `cnn-trad-fpool3_model/training/`
- Coral model pipeline: `cnn-trad-fpool3_model/platforms/coral/scripts/` (`convert_static_int8.py`, `fix_static_shape.py`, `compile_edgetpu.sh`)
- Full context: `cnn-trad-fpool3_model/DS-CNN_onboarding.html`
