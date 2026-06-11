# ML-on-MCU Results Ledger

---

## DS-CNN (cnn-trad-fpool3)

Device-in-the-loop, one common GSC **test** audio set per board (`testbench.py`). One unique row per (board, model) — re-running replaces it. Model accuracy is the canonical model score; MCU accuracy is the on-device test-bench result. Flash/SRAM come from the ELF. Energy from PPK2 power meter (µJ/inference); — where not yet measured. **GAP9 is NOFLASH**: code and weights run entirely from L2 on-chip SRAM (no external Flash); Flash .text column shows the L2 code section (Deeploy ELF) or combined static L2 (NNTool/AT, no .text/.data split). SRAM column = data+bss in L2 (weights) for Deeploy; for NNTool (int8-ne16) = L1 cluster scratchpad used for activations (20.02 KiB) — static weights are already included in the Flash/L2 value.

| Timestamp | Board | Config | Model | N | Model accuracy % | MCU accuracy % | Lat avg (ms) | Lat p95 (ms) | Flash/.text / L2 (KiB) | SRAM / L1 scratch (KiB) | Energy/Inference (µJ) | Run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-06-10 17:45 | max | int8-accel | v1 | 144 | 92.0 | 67.36 | 0.162 | 0.162 | 139.5 | 38.0 | — | results/max_v1 |
| 2026-06-10 17:30 | max | fp32-cpu | v0 | 144 | 92.4 | 93.75 | 1988.121 | 1988.147 | 208.8 | 99.7 | — | results/max_v0 |
| 2026-06-10 18:45 | u5 | int8-cpu | v1 | 144 | 92.0 | 91.67 | 64.091 | 64.096 | 150.5 | 82.8 | — | results/u5_v1 |
| 2026-06-10 18:57 | u5 | fp32-cpu | v0 | 144 | 92.4 | 93.75 | 224.542 | 224.543 | 220.8 | 125.8 | — | results/u5_v0 |
| 2026-06-10 20:21 | coral | int8-accel | v1 | 144 | 92.0 | 91.67 | 2.346 | 2.372 | 283.4 | n/a | — | results/coral_v1 |
| 2026-06-10 20:25 | coral | fp32-cpu | v0 | 144 | 92.4 | 90.97 | 417.012 | 417.240 | 304.2 | n/a | — | results/coral_v0 |
| 2026-06-10 17:09 | gap9 | fp32-cluster | v0 | 144 | 92.4 | 91.62 | 116.806 | 116.868 | 46.48 | 134.08 | 3893.85 | results/gap_v0 |
| 2026-06-09 14:45 | gap9 | int8-cluster | v1 | 144 | 92.0 | 91.45 | 17.153 | 17.165 | 46.69 | 32.86 | 480.11 | results/gap_v1 |
| 2026-06-09 17:28 | gap9 | int8-ne16 | v2 | 144 | 92.0 | 91.23 | 0.786 | 0.786 | 36.38 (L2 total) | 20.02 (L1) | 34.32 | results/gap_v2 |

---

## AI85KWS20NetV3

Live-mic online inference measurements (no labeled testbench run; MCU accuracy — for all entries). Model accuracy from ai8x-training simulation on KWS_20 test set (21 classes, 11 005 samples). Energy from PPK2 (µJ/inference). STM32U5 @ 160 MHz, MAX78000 @ 50 MHz CNN / 100 MHz CPU.

| Timestamp | Board | Config | Model | N | Model accuracy % | MCU accuracy % | Lat avg (ms) | Lat p95 (ms) | Flash/.text / L2 (KiB) | SRAM / L1 scratch (KiB) | Energy/Inference (µJ) | Run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-06-08 15:16 | max | int8-cpu | v0_0 | 16 | 90.80 | 89.43 | 897.288 | 897.290 | 397.8 | 66.7 | 49541.78 | Experiments/MAX78000/ai85kws20netv3_v0_0 |
| 2026-05-05 16:48 | max | int8-accel | v0 | 11 | 90.80 | 91.25 | 1.849 | 1.849 | 381.1 | 37.7 | 283.39 | Experiments/MAX78000/ai85kws20netv3_v0 |
| 2026-05-08 22:44 | max | int8-accel-w90 | v1 | 7 | 85.10 | 83.47 | 1.711 | 1.711 | 357.5 | 37.8 | 250.94 | Experiments/MAX78000/ai85kws20netv3_v1 |
| 2026-05-04 19:22 | u5 | fp32-cpu | v0 | 6 | 91.17 | 91.23 | 397.242 | 397.242 | 724.1 | 238.1 | 31394.71 | Experiments/U5/ai85kws20netv3_v0 |
| 2026-05-06 18:22 | u5 | int8-cpu | v1 | 3 | 90.80 | 90.01 | 173.839 | 173.839 | 250.1 | 180.8 | 12751.97 | Experiments/U5/ai85kws20netv3_v1 |
| 2026-05-10 16:10 | u5 | int8-cpu-w90 | v2 | 13 | 85.10 | 84.27 | 152.922 | 152.922 | 226.7 | 176.8 | 11475.73 | Experiments/U5/ai85kws20netv3_v2 |



