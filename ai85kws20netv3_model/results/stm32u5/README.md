# STM32U5 ai85kws20netv3 Online Results

Source data: online `on.json` files from `Experiments/General Profiling/U5`.
Missing v0 AI-core fields are backfilled from the matching X-CUBE-AI `network_generate_report.txt` because that older firmware log did not emit `model_info` over UART.

Variants:
- `v0`: unquantized deployment
- `v1`: quantized deployment
- `v2`: pruned quantized deployment

Key online findings:
- CNN latency: v0 `397.24 ms`, v1 `173.84 ms`, v2 `152.92 ms`.
- Static SRAM: v0 `238.1 KiB`, v1 `180.8 KiB`, v2 `176.8 KiB`.
- Energy per inference estimate: v0 `31394.7 uJ`, v1 `12752.0 uJ`, v2 `11475.7 uJ`.
- v2 vs v1 latency improvement: `12.03%`.
- v2 vs v1 static SRAM improvement: `2.21%`.

Generated tables are in `tables/`; generated SVG plots are in `plots/`.
