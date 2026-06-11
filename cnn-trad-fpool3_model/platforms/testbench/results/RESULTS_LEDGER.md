# DS-CNN normalized test-bench ledger (v2)

Device-in-the-loop, one common GSC **test** audio set per board (`testbench.py`). One unique row per (board, model) — re-running replaces it. Model accuracy is the canonical model score; MCU accuracy is the on-device test-bench result. Flash/SRAM come from the ELF. Energy from PPK2 power meter (µJ/inference); — where not yet measured. GAP9 runs are offline latency-only (no accuracy testbench); MCU accuracy — accordingly.

| Timestamp | Board | Config | Model | N | Model accuracy % | MCU accuracy % | Lat avg (ms) | Lat p95 (ms) | Flash .text (KiB) | SRAM (KiB) | Energy/Inference (µJ) | Run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-06-10 17:45 | max | int8-accel | v1 | 144 | 92.0 | 67.36 | 0.162 | 0.162 | 139.5 | 38.0 | — | results/max_v1 |
| 2026-06-10 17:30 | max | fp32-cpu | v0 | 144 | 92.4 | 93.75 | 1988.121 | 1988.147 | 208.8 | 99.7 | — | results/max_v0 |
| 2026-06-10 18:45 | u5 | int8-cpu | v1 | 144 | 92.0 | 91.67 | 64.091 | 64.096 | 150.5 | 82.8 | — | results/u5_v1 |
| 2026-06-10 18:57 | u5 | fp32-cpu | v0 | 144 | 92.4 | 93.75 | 224.542 | 224.543 | 220.8 | 125.8 | — | results/u5_v0 |
| 2026-06-10 17:09 | gap9 | fp32-cluster | v0 | 144 | 92.4 | 91.62 | 116.806 | 116.868 | 46.48 | 134.08 | 3893.85 | results/gap_v0 |
| 2026-06-09 14:45 | gap9 | int8-cluster | v1 | 144 | 92.0 | 91.45 | 17.153 | 17.165 | 46.69 | 32.86 | 480.11 | results/gap_v1 |
| 2026-06-09 17:28 | gap9 | int8-ne16 | v2 | 144 | 92.0 | 91.23 | 0.786 | 0.786 | — | 36.38 | 34.32 | results/gap_v2 |




kreux