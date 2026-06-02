# MAX78000 ai85kws20netv3 Online Results

Source data: online `on.json` files from `Experiments/General Profiling/MAX78000`.
Power data: online `on_peak_analysis` folders from `Experiments/PWR Consumption/MAX78000`.

Variants:
- `v0`: baseline MAX78000 deployment
- `v1`: w90 pruned MAX78000 deployment

Key online findings:
- CNN latency: v0 `1.849 ms`, v1 `1.711 ms`.
- Static SRAM: v0 `37.7 KiB`, v1 `37.8 KiB`.
- Energy per inference estimate: v0 `283.4 uJ`, v1 `250.9 uJ`.
- v1 vs v0 latency improvement: `7.48%`.
- v1 vs v0 energy improvement: `11.45%`.

Generated tables are in `tables/`; generated SVG plots are in `plots/`.
