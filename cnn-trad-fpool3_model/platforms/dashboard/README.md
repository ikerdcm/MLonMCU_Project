# MCU KWS Dashboard

Monitor live keyword-spotting inference from 1–3 MCUs at once — Coral Dev Board
Micro, STM32U5, MAX78000 — over their USB serial ports, in one normalized view.

The dashboard both **monitors** the live inference streams and **flashes** the
boards: a `Flash All` button (and per-MCU `Flash` buttons) run
[`../flash_all.sh`](../flash_all.sh) in the chosen mode (`live`/`offline`),
streaming build/flash output into a log pane. Each MCU is flashed through its own
debug probe (Coral USB VID/PID, MAX78000 CMSIS-DAP, STM32 ST-Link), so no
serial-port mapping is needed.

## How it works

Each MCU streams inference results over serial @115200. Two line formats are
supported and unified into one record:

| Format | Emitted by |
|---|---|
| `BENCH,event=inference,run=…,cnn_us=…,pred_idx=…` | STM32 (live+offline), MAX78000 (live+offline), Coral (bench) |
| `>>> left (73%) mfcc=… us infer=… us` | Coral (live) |

The dashboard runs one reader thread per MCU, **asserts DTR for Coral** (its CDC
console drops bytes until DTR is raised — plain `screen`/`cat` show nothing), and
**host-timestamps every line with `time.monotonic()`** so the three streams can be
aligned. `pred_idx`→label and `cnn_us`↔`infer_us` are normalized in
[`mcu_stream/protocol.py`](mcu_stream/protocol.py).

> ⚠️ Never open a Coral port at **1200 baud** — that resets it to the bootloader.

## Run

```bash
cd cnn-trad-fpool3_model/platforms/dashboard
./run_dashboard.sh        # first run creates .venv-dashboard and installs deps
```

Then: pick **Individual** (1 MCU) or **Multiple** (up to 3) → per slot choose
MCU, serial port (⟳ to rescan), and firmware mode → **Connect**. Confident /
non-`unknown` detections are highlighted; per-stream stats show total inferences,
inf/s, and lines/s.

## Layout

```
mcu_stream/
  protocol.py   # InferenceRecord, LABELS, BENCH + coral-live parsers, MCU profiles
  reader.py     # SerialReader(QThread): open(+DTR), readline, timestamp, parse, emit
  app.py        # PyQt UI: launcher + McuPanel + MonitorWindow
```

## Roadmap

- Mode/connection orchestration and flashing from the GUI
- Combined synchronized timeline view across MCUs
- Data-analysis tab (latency/energy/accuracy plots), CSV/Parquet export
- Shared parser reused by the existing `kws_measure_*` scripts (single source of truth)
