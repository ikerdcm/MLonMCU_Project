# DS-CNN normalized test-bench ledger (v2)

Device-in-the-loop, one common GSC **test** audio set per board (`testbench.py`). One unique row per (board, model) — re-running replaces it. Model accuracy is the canonical model score; MCU accuracy is the on-device test-bench result. Flash/SRAM come from the ELF. Energy still needs the power meter (separate).

| Timestamp | Board | Config | Model | N | Model accuracy % | MCU accuracy % | Lat avg (ms) | Lat p95 (ms) | Flash .text (KiB) | SRAM (KiB) | Run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-06-10 17:45 | max | int8-accel | v1 | 144 | 92.0 | 67.36 | 0.162 | 0.162 | 139.5 | 38.0 | results/max_v1 |
| 2026-06-10 17:30 | max | fp32-cpu | v0 | 144 | 92.4 | 93.75 | 1988.121 | 1988.147 | 208.8 | 99.7 | results/max_v0 |
| 2026-06-10 18:45 | u5 | int8-cpu | v1 | 144 | 92.0 | 91.67 | 64.091 | 64.096 | 150.5 | 82.8 | results/u5_v1 |
| 2026-06-10 18:57 | u5 | fp32-cpu | v0 | 144 | 92.4 | 93.75 | 224.542 | 224.543 | 220.8 | 125.8 | results/u5_v0 |
| 2026-06-10 20:21 | coral | int8-accel | v1 | 144 | 92.0 | 91.67 | 2.346 | 2.372 | 283.4 | n/a | results/coral_v1 |
| 2026-06-10 20:25 | coral | fp32-cpu | v0 | 144 | 92.4 | 90.97 | 417.012 | 417.240 | 304.2 | n/a | results/coral_v0 |
