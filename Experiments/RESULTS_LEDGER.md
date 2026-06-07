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

| Board | Config | Model | Mode | Latency avg (ms) | p95 (ms) | Energy/inf (µJ) | Flash (KiB) | SRAM (KiB) | Model acc (%) | MCU acc (%) | Status | Source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Coral** | `fp32-cpu` | `ds_cnn_l_float` | offline | **417.1** | 417.3 | — ᶠ | 259.0 | n/a ᵇ | **92.4** | **91.7** ⁱ | ✅ prof + model-acc + MCU-acc (energy pending) | `Coral/…_v0/offline/` |
| **Coral** | `fp32-cpu` | `ds_cnn_l_float` | online | — | — | — | — | n/a ᵇ | 92.4 | ✱ | 🔨 code ready, capture pending | `apps/kws_live_cpu` → `Coral/…_v0/online/` |
| **Coral** | `int8-accel` | `ds_cnn_l_static` (**4-blk**) | offline | 1.807 | 1.873 | ≈2358 ᵃ | 236.8 | n/a ᵇ | 23.6 ᵍ | ✱ | ⚠️ original deploy, random-calib | `Coral/…_v1/offline/` ᶜ |
| **Coral** | `int8-accel` | `ds_cnn_l_static_v2` (**6-blk**) | offline | **2.27** ʲ | 2.281 | — (v2 pending) | 236.8 | n/a ᵇ | **92.0** ᵍ | **91.0** ⁱ | ✅ prof + MCU-acc (energy pending) | re-measured 2026-06-07 ʲ |
| **Coral** | `int8-accel` | `ds_cnn_l_static_v2` | online | — | — | — | — | n/a ᵇ | 92.0 | ✱ | ❌ **gap (priority 1)** | online power not captured |
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

---

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
