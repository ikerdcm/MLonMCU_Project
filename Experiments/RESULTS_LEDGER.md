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
Accuracy = INT8 test accuracy via `training/eval_quantized_model.py` (✱ = not yet captured).

| Board | Config | Mode | Latency avg (ms) | p95 (ms) | Energy/inf (µJ) | Flash (KiB) | SRAM (KiB) | Acc (%) | Status | Source |
|---|---|---|---|---|---|---|---|---|---|---|
| **Coral** | `fp32-cpu` | offline | **417.1** | 417.3 | — ᶠ | 259.0 | n/a ᵇ | **92.4** | ✅ prof + acc (energy pending) | `Coral/…_v0/offline/` |
| **Coral** | `fp32-cpu` | online | — | — | — | — | n/a ᵇ | 92.4 | 🔨 code ready, capture pending | `apps/kws_live_cpu` → `Coral/…_v0/online/` |
| **Coral** | `int8-accel` | offline | **1.807** | 1.873 | **≈2358** ᵃ | 236.8 | n/a ᵇ | **23.6** ᵍ | ✅ prof + ⚠️ pwr · ⚠️ acc | `Coral/…_v1/offline/` ᶜ |
| **Coral** | `int8-accel` | online | — | — | — | — | n/a ᵇ | ✱ | ❌ **gap (priority 1)** | online power not captured |
| **MAX78000** | `fp32-cpu` | offline | 2048.5 | 2048.5 | 101639 | 170.1 | 68.5 | ✱ | ✅ prof + pwr | `MAX78000/…_v0/offline/` |
| **MAX78000** | `int8-accel` | offline | **0.161** | 0.161 | **23.1** | 101.0 | 7.0 | ✱ | ✅ prof + pwr | `MAX78000/…_v1/offline/` |
| **MAX78000** | `int8-accel` | online | 0.162 | 0.162 | ✱ (CSV only) | 150.6 | 69.9 ᵈ | ✱ | ✅ prof + ⚠️ pwr | `MAX78000/…_v1/online/` |
| **STM32U5** | `fp32-cpu` | offline | 224.5 | 224.5 | 16159 | 172.7 | 94.5 | ✱ | ✅ prof + pwr | `U5/…_v0/offline/` |
| **STM32U5** | `fp32-cpu` | online | 224.6 | 224.6 | 16161 | 228.6 | 193.9 ᵈ | ✱ | ✅ prof + pwr | `U5/…_v0/online/` |
| **STM32U5** | `int8-cpu` | offline | 64.06 | 64.06 | 4610 ᵉ | 105.8 | 70.3 | ✱ | ✅ prof | `U5/…_v1/offline/` |
| **STM32U5** | `int8-cpu` | online | 64.08 | 64.09 | 4927 ᵉ | 158.5 | 151.4 ᵈ | ✱ | ✅ prof + ⚠️ pwr | `U5/…_v1/online/` |

**Legend:** ✅ done · ⚠️ partial · ❌ missing · ✱ value not yet captured · — n/a.

**Notes**
- **ᵃ Coral energy is measured differently** — integrated from the power-peak analysis (`on_peak_analysis/peak_report.json`, V=5.0). ≈**2358 µJ total** per inference window (≈2.86 ms peak) vs ≈**516 µJ excess** over the ~635 mW idle baseline. The MAX/U5 numbers are `power_w × latency` from a single avg current, so treat cross-board energy as *approximate* until methodology is unified.
- **ᵇ Coral SRAM:** the tensor arena lives in SDRAM (`data+bss` ≈ 17.6 MB in the ELF), so on-chip SRAM is not the comparable axis. Flash `.text` (236.8 KiB) is the meaningful footprint.
- **ᶜ Coral profiling** was copied into the `Experiments/` GP tree on 2026-06-06 from `cnn-trad-fpool3_model/platforms/coral/measurements/offline_measurements/kws_metrics_20260519_151706/summary.json` (normalized).
- **ᵈ online builds** include the mic + MFCC path, so their flash/SRAM are larger than the offline (model-only) build — don't compare online memory to offline.
- **ᵉ STM32U5/MAX `energy_estimate`** in `summary.json` is `power_w × latency` from an *assumed avg board current* (U5 offline uses 21.94 mA for both fp32 and int8 — i.e. not an independent per-config power measurement). Real power CSV traces exist only for some online runs under `PWR Consumption/`. Backfilled into the tree on 2026-06-06.
- **ᶠ Coral `fp32-cpu` offline** captured 2026-06-06 (`kws_bench_cpu`, 46-run, std 0.16 ms, pred=left ✓; via `tools/run_offline_measurement_cpu.sh`). **Energy pending** (needs the power-meter capture, like int8). Headline: Edge-TPU **1.81 ms** vs M7 CPU **417 ms ≈ 230× faster** (9.4 vs 2179 MOPS).
- **ᵍ ⚠️ The measured int8-accel model is broken (random calibration).** Test-set accuracy (`eval_quantized_model.py`, 2026-06-06): FP32 `ds_cnn_l_float` = **92.4 %**; INT8 `ds_cnn_l_static` (the model behind the 1.81 ms run) = **23.6 %**; INT8 `ds_cnn_l_static_v2` (real-data recalibration, from `ds_cnn_l_finetuned`) = **92.0 %**. So the deployed int8 (v1) lost ~69 pts to bad calibration (onboarding §7.1). The corrected int8 (**v2, 92.0 %**) recovers FP32-parity accuracy at the same INT8 architecture → needs a latency re-measure (expect ≈1.81 ms) to become the real int8-accel comparison point. FP32-vs-INT8 *parity* comparison = 92.4 % vs 92.0 %. Colleague's `main` `final_results` reports INT8 **92.72 %** = `ds_cnn_l.tflite` (confirmed 4533/4890), **mislabeled** there as `ds_cnn_l_static.tflite`; see `final_results/ds_cnn_accuracy_results/RECONCILIATION.md`.

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
- [ ] **Re-measure int8-accel latency with the corrected v2 model** (`ds_cnn_l_static_v2_edgetpu`, 92.0 %) — the 1.81 ms run used the broken 23.6 % model — *high value, Coral*
- [ ] INT8/FP32 test accuracy for the **MAX78000 / STM32U5** rows (still ✱) — same `eval_quantized_model.py` procedure
- [ ] Coral `int8-prune` (structured / 4-block) — *priority 2*
- [ ] Coral `int8-prune-distill` — *priority 3*
