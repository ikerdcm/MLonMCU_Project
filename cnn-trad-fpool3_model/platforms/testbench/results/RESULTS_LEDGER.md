# DS-CNN normalized test-bench ledger (v2)

Device-in-the-loop, one common GSC **test** audio set per board (`testbench.py`). One unique row per (board, model) — re-running replaces it. Accuracy + latency are on-device; flash/SRAM from the ELF. Energy still needs the power meter (separate).

| Timestamp | Board | Config | Model | N | Accuracy % | Lat avg (ms) | Lat p95 (ms) | Flash .text (KiB) | SRAM (KiB) | Run |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-06-10 17:45 | max | int8-accel | v1 | 144 | 67.36 | 0.162 | 0.162 | 139.5 | 38.0 | results/max_v1 |
| 2026-06-10 17:30 | max | fp32-cpu | v0 | 144 | 93.75 | 1988.121 | 1988.147 | 208.8 | 99.7 | results/max_v0 |
