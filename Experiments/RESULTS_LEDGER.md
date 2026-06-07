# DS-CNN cross-MCU results ledger (living document)

**Single source of truth for the headline numbers of every DS-CNN experiment.**
Update this file *every time* a new measurement is captured — one row per
`(Board × config × mode)`. Keep it in sync with the raw data under
`Experiments/{General Profiling,PWR Consumption}/…` and the per-platform
`measurements/` folders. Companion to [ROADMAP.md](ROADMAP.md) and
`cnn-trad-fpool3_model/DS-CNN_onboarding.html`.

Model: **DS-CNN-L** (`cnn-trad-fpool3_model`), 3,937,717 MACs/inf, input `[1,49,10,1]`, 12 classes.
Last updated: **2026-06-06**.

---

## Config registry (what each `vN` folder means)

`vN` means a *different config on each board* — always refer to experiments by **config id**, not `vN`.

| Config id | Precision | Runs on | Folder mapping |
|---|---|---|---|
| `fp32-cpu` | FP32 | Arm core, no accel | Coral `v0` *(offline measured)* · MAX `v0` · U5 `v0` |
| `int8-accel` | INT8 | MAX CNN-accel / Coral Edge-TPU | Coral `v1` · MAX `v1` |
| `int8-cpu` | INT8 | STM32U5 CMSIS-NN | U5 `v1` |
| `int8-prune` | INT8 | structured prune (smaller arch) | — *(none yet)* |
| `int8-prune-distill` | INT8 | prune + distillation recovery | — *(none yet)* |
| `int4-cpu` | INT4 | **STM32U5 only** (X-CUBE-AI) | — *(none yet)* |

---

## Headline metrics

Latency = pure CNN inference (`cnn_latency_us`, excludes the 1 s audio window).
Energy/inf from `energy_estimate` block unless noted. Flash = `.text` KiB; SRAM = `data+bss` (static).
**Model acc** = host TFLite reference accuracy (`eval_quantized_model.py`, n=4890). **MCU acc** = on-device test-set accuracy (device-in-the-loop). ✱ = not yet measured. See ʰ for the model-vs-MCU distinction.

Model column: **Coral & U5 run the identical DS-CNN-L**; **MAX runs the ai8x `kws-ds-cnn-l-kws12` variant** (conv1d, ~1.34M MACs) ᵏ — not the same model.

> **Reading the Coral rows — latency is set by BLOCK COUNT, accuracy by training/calibration.**
> On the Edge-TPU every **4-block** model runs **≈1.8 ms**, every **6-block** **≈2.27 ms** (filters and calibration don't change speed). So the four ~1.8 ms 4-block rows are the **same speed tier**; they differ only in accuracy:
> - `ds_cnn_l_static` **23.6 %** — the *original deploy*, broken **random calibration**. The 23.6 % is a **host-eval artifact** (mis-scaled input, ᵍ); it actually works on-device. Kept for before/after.
> - `kws_ref` **91.7 %** — the **same 4-block, correctly calibrated** (what the broken one should have been). Best latency+accuracy point.
> - `f64b4` **89.7 %** — a 4-block obtained by **actually pruning** the 6-block + fine-tune (the rigorous prune).
>
> **Bottom line:** a calibrated 4-block is **~1.8 ms @ ~90–92 %** vs the full 6-block **2.27 ms @ 92 %** — ~20 % faster at ≈parity. "Best in latency" just = "it's a 4-block".

| Board | Config | Model | Mode | Latency avg (ms) | p95 (ms) | Energy/inf (µJ) | Flash (KiB) | SRAM (KiB) | Model acc (%) | MCU acc (%) | Status | Source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Coral** | `fp32-cpu` | `ds_cnn_l_float` | offline | **417.1** | 417.3 | — ᶠ | 259.0 | n/a ᵇ | **92.4** | **91.7** ⁱ | ✅ prof + model-acc + MCU-acc (energy pending) | `Coral/…_v0/offline/` |
| **Coral** | `fp32-cpu` | `ds_cnn_l_float` | online | — | — | — | — | n/a ᵇ | 92.4 | ✱ | 🔨 code ready, capture pending | `apps/kws_live_cpu` → `Coral/…_v0/online/` |
| **Coral** | `int8-accel` | `ds_cnn_l_static` (**4-blk**, random-calib) | offline | 1.807 | 1.873 | ≈2358 ᵃ | 236.8 | n/a ᵇ | 23.6 ᵍ | ✱ | ⚠️ v1 original deploy — broken calibration | `Coral/…_v1/offline/` ᶜ |
| **Coral** | `int8-accel` | `kws_ref` (**4-blk**, real-calib) | offline | 1.81 | 1.87 | ≈2358 ᵃ | 236.8 | n/a ᵇ | **91.7** ᵐ | ✱ ᵐ | ✅ **v1 done right** — same 4-blk, real calibration | `training/.../kws_ref_model.tflite` |
| **Coral** | `int8-accel` | `ds_cnn_l_static_v2` (**6-blk**) | offline | **2.27** ʲ | 2.281 | — (v2 pending) | 236.8 | n/a ᵇ | **92.0** ᵍ | **91.0** ⁱ | ✅ prof + MCU-acc (energy pending) | re-measured 2026-06-07 ʲ |
| **Coral** | `int8-accel` | `ds_cnn_l_static_v2` (**6-blk**) | online | — | — | — | — | n/a ᵇ | 92.0 | ✱ | ❌ **gap (priority 1)** | online power not captured |
| **Coral** | `int8-prune` | **f64b4** (4-blk, 64f) | offline | **1.88** ⁿ | — | — | — | n/a ᵇ | **89.7** | **91.0** ⁿ | ✅ **prune winner** (real, warm-start+ft) | `models/…_pruned_f64b4_…` |
| **Coral** | `int8-prune` | f32b6 (6-blk, 32f) | offline | 2.25 ⁿ | — | — | — | n/a ᵇ | 76.7 | 78.5 ⁿ | real prune (32-filter) | `models/…_pruned_f32b6_…` |
| **Coral** | `int8-prune` | f32b4 (4-blk, 32f) | offline | 1.82 ⁿ | — | — | — | n/a ᵇ | 65.1 | 68.8 ⁿ | real prune (32f+4blk) | `models/…_pruned_f32b4_…` |
| **Coral** | `int8-prune-distill` | **f64b4** + KD (4-blk) | offline | 1.88 | — | — | — | n/a ᵇ | **90.8** ᵒ | ≈90.8 ᵒ | ✅ **ladder top** (+1.1 vs prune) | `models/…_distilled_f64b4_…` |
| **Coral** | `int8-prune-distill` | f32b4 + KD (4-blk,32f) | offline | 1.82 | — | — | — | n/a ᵇ | 76.5 ᵒ | — | rescue (+11.4 vs prune) ᵒ | `models/…_distilled_f32b4_…` |
| **MAX78000** ᵏ | `fp32-cpu` | `kws-ds-cnn-l-kws12` | offline | 2048.5 | 2048.5 | 101639 | 170.1 | 68.5 | 92.4 | ✱ | ✅ prof + pwr | `MAX78000/…_v0/offline/` |
| **MAX78000** ᵏ | `int8-accel` | `kws-ds-cnn-l-kws12` | offline | **0.161** | 0.161 | **23.1** | 101.0 | 7.0 | 92.0 | ✱ | ✅ prof + pwr | `MAX78000/…_v1/offline/` |
| **MAX78000** ᵏ | `int8-accel` | `kws-ds-cnn-l-kws12` | online | 0.162 | 0.162 | ✱ (CSV only) | 150.6 | 69.9 ᵈ | 92.0 | ✱ | ✅ prof + ⚠️ pwr | `MAX78000/…_v1/online/` |
| **STM32U5** | `fp32-cpu` | `ds_cnn_l (onnx)` | offline | 224.5 | 224.5 | 16159 | 172.7 | 94.5 | 92.4 | ✱ | ✅ prof + pwr | `U5/…_v0/offline/` |
| **STM32U5** | `fp32-cpu` | `ds_cnn_l (onnx)` | online | 224.6 | 224.6 | 16161 | 228.6 | 193.9 ᵈ | 92.4 | ✱ | ✅ prof + pwr | `U5/…_v0/online/` |
| **STM32U5** | `int8-cpu` | `ds_cnn_l (onnx)` | offline | 64.06 | 64.06 | 4610 ᵉ | 105.8 | 70.3 | 92.0 | ✱ | ✅ prof | `U5/…_v1/offline/` |
| **STM32U5** | `int8-cpu` | `ds_cnn_l (onnx)` | online | 64.08 | 64.09 | 4927 ᵉ | 158.5 | 151.4 ᵈ | 92.0 | ✱ | ✅ prof + ⚠️ pwr | `U5/…_v1/online/` |

**Legend:** ✅ done · ⚠️ partial · ❌ missing · ✱ value not yet captured · — n/a.

**Notes**
- **ᵃ Coral energy is measured differently** — integrated from the power-peak analysis (`on_peak_analysis/peak_report.json`, V=5.0). ≈**2358 µJ total** per inference window (≈2.86 ms peak) vs ≈**516 µJ excess** over the ~635 mW idle baseline. The MAX/U5 numbers are `power_w × latency` from a single avg current, so treat cross-board energy as *approximate* until methodology is unified.
- **ᵇ Coral SRAM:** the tensor arena lives in SDRAM (`data+bss` ≈ 17.6 MB in the ELF), so on-chip SRAM is not the comparable axis. Flash `.text` (236.8 KiB) is the meaningful footprint.
- **ᶜ Coral profiling** was copied into the `Experiments/` GP tree on 2026-06-06 from `cnn-trad-fpool3_model/platforms/coral/measurements/offline_measurements/kws_metrics_20260519_151706/summary.json` (normalized).
- **ᵈ online builds** include the mic + MFCC path, so their flash/SRAM are larger than the offline (model-only) build — don't compare online memory to offline.
- **ᵉ STM32U5/MAX `energy_estimate`** in `summary.json` is `power_w × latency` from an *assumed avg board current* (U5 offline uses 21.94 mA for both fp32 and int8 — i.e. not an independent per-config power measurement). Real power CSV traces exist only for some online runs under `PWR Consumption/`. Backfilled into the tree on 2026-06-06.
- **ᶠ Coral `fp32-cpu` offline** captured 2026-06-06 (`kws_bench_cpu`, 46-run, std 0.16 ms, pred=left ✓; via `tools/run_offline_measurement_cpu.sh`). **Energy pending** (needs the power-meter capture, like int8). Headline: Edge-TPU **1.81 ms** vs M7 CPU **417 ms ≈ 230× faster** (9.4 vs 2179 MOPS).
- **ᵍ ⚠️ The measured int8-accel model is broken (random calibration).** Test-set accuracy (`eval_quantized_model.py`, 2026-06-06): FP32 `ds_cnn_l_float` = **92.4 %**; INT8 `ds_cnn_l_static` (the model behind the 1.81 ms run) = **23.6 %**; INT8 `ds_cnn_l_static_v2` (real-data recalibration, from `ds_cnn_l_finetuned`) = **92.0 %**. So the deployed int8 (v1) lost ~69 pts to bad calibration (onboarding §7.1). The corrected int8 (**v2, 92.0 %**) recovers FP32-parity accuracy at the same INT8 architecture → needs a latency re-measure (expect ≈1.81 ms) to become the real int8-accel comparison point. FP32-vs-INT8 *parity* comparison = 92.4 % vs 92.0 %. Colleague's `main` `final_results` reports INT8 **92.72 %** = `ds_cnn_l.tflite` (confirmed 4533/4890), **mislabeled** there as `ds_cnn_l_static.tflite`; see `final_results/ds_cnn_accuracy_results/RECONCILIATION.md`.
- **ʰ Model acc vs MCU acc.** *Model acc* = host TFLite reference (FP32 `ds_cnn_l_float` 92.4 %, INT8 v2 92.0 %); the same reference model is assumed across boards. *MCU acc* = the accuracy the *deployed* board actually achieves on the test set — separate because each board quantizes independently (Coral Edge-TPU, MAX ai8x, U5 X-CUBE-AI) so it can differ. **No board has MCU acc yet:** the offline firmware runs latency on one fixed vector, not the test set; measuring it needs a **device-in-the-loop** mode (host streams each test feature → board infers → returns pred → host tallies), which doesn't exist, and MAX/U5 also need their (Linux) build toolchains. Colleague's `main` reports FP32 93.0 % / INT8 92.7 % for a different trained lineage (see RECONCILIATION.md).
- **ⁱ Coral on-device accuracy (device-in-the-loop), 2026-06-07.** `apps/kws_eval` ran a 144-sample stratified test subset (12/class) on the **v2 Edge-TPU** model: **131/144 = 90.97 %**, *identical* to the host v2 on the same subset (also 131/144) → **the Edge-TPU is numerically faithful** to the host int8 model. Subset estimate (±~5 %); full-set host v2 = 92.0 %. Subset embedded via `scripts/make_eval_set.py` (kept small to fit internal flash); scale to the full set by loading from LittleFS. MAX/U5 need this on their Linux toolchains. **fp32-cpu (v0)** measured the same way with `apps/kws_eval_cpu` (M7 CPU float, loads the subset `.bin` from LittleFS): **132/144 = 91.66 %** ≈ host fp32 → CPU float path faithful. Both Coral configs: on-device ≈ host (Edge-TPU and M7-CPU reproduce the host model).
- **ʲ Coral int8-accel latency re-measured on the corrected v2 (6-block) model**, 2026-06-07: avg **2.27 ms** (p95 2.281). The earlier **1.81 ms was the broken v1 (4-block)** model → the correct model is ~25 % slower (more layers, 16 vs 12 Edge-TPU ops). `kws_bench` now loads `ds_cnn_l_static_v2_edgetpu.tflite`. **Energy (≈2358 µJ) is still the v1 capture** — re-measure for v2 pending. (The bench's embedded "left" vector is still v1-scaled, so its pred label is meaningless; latency is value-independent.)
- **ᵏ ⚠️ MAX78000 runs a DIFFERENT model — not the shared DS-CNN-L.** Its ai8x accelerator can't run the TFLite, so it deploys `kws-ds-cnn-l-kws12` (ai8x model zoo, `ai85kwsmfccnet`): **6 conv1d layers, ~1.34M MACs** (`cnn.h`), vs the Coral/U5 TFLite DS-CNN-L (depthwise-separable 2D, **3.94M MACs**). So MAX is ≈3× fewer MACs + a different conv structure → its **0.16 ms latency and energy are *not* a same-model comparison**, and its true accuracy is its own ai8x QAT number (the 92 % in MAX `Model acc` cells is the TFLite value, *not* MAX's actual — to be replaced by ai8x eval). **Coral & U5 do share the identical DS-CNN-L.** For Story A, treat MAX as "best KWS this HW can do" rather than "same model, different HW."
- **ᵒ `int8-prune-distill` (v4), 2026-06-07** — knowledge distillation from the **6-block FP32 teacher** (35,532 p) into the pruned student (`distill_branch.py`: KL on T=4-softened probs + CE, α=0.3, 25 ep). Recovers accuracy lost to prune+int8: **`f64b4` 89.7 → 90.8 % int8 (+1.1)** — now within 1.2 pt of the full 6-block (92.0 %) **at 1.88 ms (~17 % faster)** → the **best Coral model**; **`f32b4` 65.1 → 76.5 % (+11.4)** — distillation helps the heavily-pruned most, but at the same ~1.8 ms it's still ≪ `f64b4` → reconfirms *drop blocks, keep filters*. Latency unchanged (same architecture); MCU acc ≈ host (Edge-TPU faithful, ⁱ — distilled models not re-flashed). **Completes the Coral ladder: fp32-cpu → int8-accel → int8-prune → int8-prune-distill.**
- **ᵐ `kws_ref` = "v1 done right" (the calibration before/after).** The deployed v1 (`ds_cnn_l_static`, 4-block) is the `kws_ref_model` quantized with **random** calibration → 23.6 % (host artifact, ᵍ). `kws_ref_model.tflite` is the **same 4-block, same weights**, quantized with **real** MFCC calibration (scale 0.5847/83) → **91.7 %** (4482/4890). Identical architecture → identical latency/energy (~1.8 ms / ≈2358 µJ; calibration is value-independent) — calibration is the *only* difference, so it sits with the broken row under `int8-accel`, **not** pruning. MCU acc ✱ = not separately deployed, but ≈91.7 % (Edge-TPU faithful, ⁱ). (A real *static* recalibrated-4-block edgetpu deploy needs a Keras-3 fix in `convert_static_int8.py` — `TFSMLayer` for the legacy SavedModel.) The actual pruning (6-block → smaller) is the ⁿ branches; `f64b4` is its 4-block result (89.7 %).
- **ⁿ Real structured-prune branches (flavor b), Coral v3** — warm-started from the 6-block v2 (block-truncate / channel-slice), fine-tuned (`training/prune_branch.py`, 25 ep), int8-quantized (real calib) + Edge-TPU compiled (all 12/16 ops on-TPU). On-device 2026-06-07 via `kws_eval` (144-sample subset, mean Invoke latency). **MCU acc ≈ host int8 acc** (Edge-TPU faithful). Model-acc column = host int8 **test** (full 4890); MCU-acc = the 144-subset. **Key finding: Edge-TPU latency is block-count-bound, not filter-bound** — 4-block ≈ 1.8–1.9 ms, 6-block ≈ 2.25 ms *regardless of 32 vs 64 filters* (overhead-dominated). Cutting filters (32) gives ~no speedup but large accuracy loss (narrow models are quantization-sensitive). **Winner: `f64b4` (4-block, 64-filter) = 1.88 ms @ 89.7 %** vs full 6-block 2.27 ms @ 92.0 % → ~17 % faster at −2.3 %. **Prune rule for Coral: drop blocks, keep filters.** The 32-filter branches (`f32b6` 76.7 %, `f32b4` 65.1 %) need **distillation/QAT** to be viable → that's the `int8-prune-distill` (v4) stage; carry **f64b4** forward as the primary, keep all three.

---

## SOTA comparison (Application)

DS-CNN is *the* reference KWS architecture (Zhang "Hello Edge" 2017 [1]; the MLPerf-Tiny KWS reference [3]). Our models map directly onto it:

| Model | Params | Test acc (Speech Commands) | Notes |
|---|---|---|---|
| Hello Edge DS-CNN S/M/L | up to ~0.5 M | **94.4 / 94.9 / 95.4 %** (float, v1) | best-in-class *float*, larger nets — Zhang 2017 [1] |
| MLPerf-Tiny DS-CNN reference | **24,908** | ~90 % int8 / ~92.7 % float | the standard tinyML benchmark net [3] |
| **Ours — `kws_ref`/`f64b4` (4-blk)** | **24,908** | **91.7 % int8** (host) · 89.7 % (`f64b4`) · 91.0 % on-device | **== the MLPerf-Tiny reference architecture** |
| **Ours — DS-CNN-L (6-blk)** | 35,532 | **92.0 % int8 · 92.4 % float** (3.94 M MACs) | larger variant |

**Takeaway:** our `f64b4` is *literally the MLPerf-Tiny DS-CNN reference* (identical 24,908 params) and reaches ~90 % int8 — on par with the reference; the 6-block hits 92 %. The ~3-pt gap to Hello Edge's 95.4 % is a *larger float* net on v1 data. **Novelty vs the papers: none deploy+profile the *same* DS-CNN across three MCU classes (Edge-TPU / CNN-accelerator / CPU) with on-device accuracy** — that cross-platform tradeoff + the Coral overhead-ceiling finding is the contribution.

*Refs:* [1] Zhang et al., *Hello Edge* (arXiv 1711.07128). [2] Sorensen et al., DS-CNN KWS, EURASIP 2020. [3] MLPerf Tiny (arXiv 2106.07597 / MLCommons). [4] ADI, *Keyword Spotting on MAX78000*.

## Hardware-efficiency summary (deployed INT8, offline)

| Metric | Coral (Edge-TPU) | MAX78000 (CNN accel) ᵏ | STM32U5 (CMSIS-NN) |
|---|---|---|---|
| Inference latency | 1.81 ms (4-blk) / 2.27 ms (6-blk) | **0.16 ms** | 64.1 ms |
| Clock | 800 MHz | 100 MHz | 160 MHz |
| Throughput | 2179 MMAC/s | ~8400 MMAC/s (eff.) | 61 MMAC/s |
| MAC / cycle | 2.72 | ~84 (massively parallel) | 0.38 |
| **Energy / inference** | ≈2358 µJ ᵃ | **23 µJ** | 4610 µJ |
| **MAC / Joule (≈MAC·s⁻¹·W⁻¹)** | ~1.7 G ᵃ | **58 G** | 0.85 G |
| Active power | ~0.7 W (5 V) | 0.14 W | 0.07 W |
| Flash (`.text`) | 237 KiB | 101 KiB | 106 KiB |
| Static SRAM | arena in SDRAM ᵇ | **7.0 KiB** | 70 KiB |

ᵏ MAX runs the smaller ai8x net (1.34 M MACs) vs Coral/U5's 3.94 M, so MAC-based metrics aren't 1:1 — **energy/inference is the fair cross-board axis.**

- **Energy: MAX78000 is ~100× more efficient/inference than Coral, ~200× vs U5** (it finishes 11–400× sooner). Note U5 has the *lowest power* (0.07 W) yet *highest energy/inf* — power ≠ efficiency; time dominates.
- **End-to-end is dominated by the 1 s audio window:** end-to-end ≈ 1000.2 / 1001.8 / 1064 ms (MAX / Coral / U5) = acquisition (1000 ms) + inference. **Inference is 0.02–6 % of the pipeline** → user-perceived latency is the 1 s window on all three (this is why Coral's 1.8 ms "ceiling" is invisible).
- **Memory 32→8 bit (U5, cleanest report):** weights **≈139 KB (fp32) → 34.7 KB (int8)** = **4×**; activations 18.3 KB int8. INT4 (`--compression 4bit`) would ~halve weights again → completes the 32→8→4 axis (TODO).

## How to update (per new measurement)

1. Capture the run (dashboard or `tools/run_{offline,live}_measurement.sh`) → `summary.json` (+ power CSV).
2. Drop the artifacts into `Experiments/{General Profiling,PWR Consumption}/<Board>/cnn-trad-fpool3_model_<vN>/<mode>/`.
3. Run `training/eval_quantized_model.py` for the INT8 test accuracy.
4. Add/replace the row here. **Never overwrite a previous config's row** — every optimization is a new row (new `vN`/config id), so before/after stays visible.
5. If you add a new config id, register it in the table above.

## Coral `fp32-cpu` — capture status & how-to

**Offline General-Profiling: DONE** (2026-06-06, board on this Mac @ `/dev/tty.usbmodem1301`).
Still pending: **online** live, **energy** (power meter), **FP32 accuracy** (eval).

```sh
cd cnn-trad-fpool3_model/platforms/coral
python3 scripts/convert_float_tflite.py          # (re)gen float model — needs TF, no board
./scripts/build_and_flash_bench_cpu.sh           # offline bench  (or build_and_flash_live_cpu.sh)
./tools/run_offline_measurement_cpu.sh --port /dev/tty.usbmodem1301   # → summary.json (latency+mem)
# online: ./scripts/build_and_flash_live_cpu.sh then ./tools/run_live_measurement.sh --port …
# energy: capture the current trace + Experiments/PWR Consumption/tools/analyze_power_peaks.py
# accuracy: cd ../../training && python3 eval_quantized_model.py \
#             --tfl_file_name ../models/ds_cnn_l_float.tflite --target_set test
```

Results go to `Experiments/.../Coral/cnn-trad-fpool3_model_v0/{offline,online}/`, then fill the rows above.

## Open gaps (tracked)

- [x] Coral `fp32-cpu` **offline** profiling captured (417.1 ms, 2026-06-06)
- [ ] Coral `fp32-cpu` **online** live capture (`kws_live_cpu`) — *priority 1*
- [ ] Coral `fp32-cpu` **energy** (power meter, offline+online) + **FP32 accuracy** (eval) — *priority 1*
- [ ] Coral `int8-accel` **online** power — *priority 1*
- [x] Copy Coral `int8-accel` offline `summary.json` into the `Experiments/` GP tree (normalized 2026-06-06)
- [x] Backfill STM32U5 `int8-cpu` profiling JSON (offline + online) into the `Experiments/` tree (2026-06-06)
- [x] Coral accuracy captured (2026-06-06): FP32 92.4 % · int8 v1 (random-calib) 23.6 % · int8 v2 (recalibrated) 92.0 % — see ᵍ
- [x] Re-measured int8-accel latency on corrected v2 model (2026-06-07): **2.27 ms** (vs broken-v1 1.81 ms, 4-block→6-block); `kws_bench`/`kws_live` now load v2 ʲ. *(v2 energy re-capture still pending.)*
- [ ] INT8/FP32 test accuracy for the **MAX78000 / STM32U5** rows (still ✱) — same `eval_quantized_model.py` procedure
- [ ] Coral `int8-prune` (structured / 4-block) — *priority 2*
- [ ] Coral `int8-prune-distill` — *priority 3*
