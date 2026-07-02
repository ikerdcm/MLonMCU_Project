# MAX78000 ai85kws20netv3 Online Results

Source data: online `on.json` files from `Experiments/General Profiling/MAX78000`.
Power data: peak analysis folders from `Experiments/PWR Consumption/MAX78000`.

Variants:
- `v0`: baseline MAX78000 deployment (CNN hardware accelerator)
- `v1`: w90 pruned MAX78000 deployment (CNN hardware accelerator)
- `cpu`: INT8 CPU-only deployment (no CNN hardware accelerator)

Key online findings:
- CNN latency: v0 `1.849 ms`, v1 `1.711 ms`, cpu `897.288 ms`.
- Static SRAM: v0 `37.7 KiB`, v1 `37.8 KiB`, cpu `66.7 KiB`.
- Energy per inference: v0 `283.4 uJ`, v1 `250.9 uJ`, cpu `49541.8 uJ`.
- v1 vs v0 latency improvement: `7.48%`.
- cpu vs v0 latency factor: `485x` slower (no hardware accelerator).

Generated tables are in `tables/`; generated SVG plots are in `plots/`.
