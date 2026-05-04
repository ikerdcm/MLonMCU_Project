# Experiments

This directory stores experiment results grouped by experiment type, platform, and model. Each model folder contains two subfolders: `online` and `offline`.

## Structure

```
Experiments/
  General Profiling/
    <platform>/
      <model>/
        online/
        offline/
  PWR Consumption/
    <platform>/
      <model>/
        online/
        offline/
```

Examples of valid values:
- `<platform>`: `MAX78000`, `N6`, `U5`
- `<model>`: `ai85kws20netv3_model_v0`, `ai85kws20netv3_model_v1`

Notes:
- `online/` holds data captured from live runs (with microphone activated)
- `offline/` holds data captured from prerecorded data

## PWR measurement guidelines (nRF Connect)

Use the following protocol for power measurements:
- Measurement duration: 30 seconds.
- Inferences per measurement: 6 total (one inference every 5 seconds).
- Export format: CSV.
- File naming: `<word>.csv` (e.g., `yes.csv`, `on.csv`).
- Store the exported CSV in the corresponding `PWR Consumption/<platform>/<model>/online/` folder.
