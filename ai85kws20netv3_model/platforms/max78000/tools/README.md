# MAX78000 Measurement Tools

This folder is split into:

- `baseline/`: wrappers for the original `kws20_demo`
- `w90/`: wrappers for the width-reduced `kws20_demo_w90`
- top-level shared scripts:
  - `build_flash_max78000.sh`
  - `kws20_measure_metrics_max78000.py`
  - `run_kws20_accuracy.sh`
  - `run_kws20_accuracy_playback.py`
  - `make_kws20_manifest.py`
  - `extract_max78000_ai_dumps.py`

Variant-specific JSON configs remain at the top level because the shared scripts consume them directly.
The `baseline/` and `w90/` folders are the intended entry points.
