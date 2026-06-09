# PPK2 power-measurement automation (DS-CNN cross-MCU)

Automated power capture with the Nordic **Power Profiler Kit II** over USB
(`ppk2-api`), so we don't depend on the nRF Connect GUI. Works for any board.

## Setup
- **Hardware:** PPK2 **Ampere meter** in series with the MCU VDD (U5: across the
  **IDD jumper**, board self-powered). Marker GPIO → PPK2 logic **D0** (U5 = **PB8**).
  Common GND. For *flashing through* the PPK2, use **Source meter @3.3 V** (stable);
  Ampere mode for measuring (matches the colleague's ~22 mA U5 numbers).
- **⚠️ The nRF Connect / Power Profiler app must be CLOSED** — only one program can
  own the PPK2 at a time, or `ppk2-api` gets `Resource busy`.
- **venv** (recreate if `/tmp/ppk2venv` is gone): `python3 -m venv venv && venv/bin/pip install -r requirements.txt`

## Scripts
- `run_power_test.py <dur_s> <label>` — power DUT, capture, **auto-detect inference
  spikes from the current** (no marker needed once idle≪inference), print + save
  `npz`+`json` (energy/inference mean±std, idle/active power, latency).
- `ppk2_live.py` — local web UI (live current chart + stats) at `http://localhost:8077`.
- `ppk2_capture.py` — raw capture → npz (current + 8 logic channels).
- `analyze_u5_power.py <npz>` — offline analysis of a saved capture.
- `u5_serial_cap.py <port> <dur>` — read the BENCH UART (cross-check the inferences).

## Key findings (U5 int8)
- Idle (`HAL_Delay` spin) ≈ inference power (~21 mA) → inferences are **invisible**
  in current. Fix: **sleep-idle firmware** (`__WFI` between inferences) drops idle to
  ~10 mA so each inference is a clean spike. (`KWS20_CFG_POWER_*` in
  `keyword_spotting_u5_ds_cnn_8-bit/Core/Inc/kws20_mode_config.h`.)
- Measured U5 int8: **idle 10 mA/33 mW, inference 21.1 mA/69.8 mW, ~64 ms, ≈4766 µJ/inf**
  (matches ledger estimate 4610 µJ). BENCH serial confirms 6 inferences @ 64 µs-timed.

## TODO (next session)
- Official protocol: settle 1 s then 1 inference every 5 s → inferences at t = 1,6,11,16,21,26.
- **CSV export** (`Timestamp(ms),Current(uA)`, colleague's format) for plotting + `analyze_power_peaks.py`.
- Per-MCU marker pin + sleep-idle firmware; measure all DS-CNN configs (see RESULTS_LEDGER).
