# GAP9 Deeploy — Vollständiger Workflow (alle Commands manuell)

---

## Schritt 1 — Board prüfen (Host)

```bash
lsusb | grep -i "0403:6011"
```

Muss etwas ausgeben. Falls nicht: Board einstecken, USB-Kabel prüfen.

---

## Schritt 2 — ftdi_sio Treiber ablösen (Host, als sudo)

Der Kernel-Treiber `ftdi_sio` blockiert OpenOCD. Muss jedes Mal gemacht werden wenn das Board eingesteckt wird.

```bash
# Alle Interfaces des GAP9-FTDI-Chips finden und ftdi_sio ablösen
for dev in /sys/bus/usb/devices/*; do
  if [[ -f "$dev/idVendor" && -f "$dev/idProduct" ]]; then
    if [[ "$(cat "$dev/idVendor")" == "0403" && "$(cat "$dev/idProduct")" == "6011" ]]; then
      for iface in "$dev":*; do
        [[ -e "$iface" ]] || continue
        iface_name="$(basename "$iface")"
        if [[ -L "$iface/driver" ]]; then
          driver="$(basename "$(readlink "$iface/driver")")"
          if [[ "$driver" == "ftdi_sio" ]]; then
            echo "Unbinding $iface_name from ftdi_sio"
            echo "$iface_name" | sudo tee /sys/bus/usb/drivers/ftdi_sio/unbind
          fi
        fi
      done
    fi
  fi
done
```

Überprüfen ob es geklappt hat:

```bash
lsusb -t | grep -A4 "0403"
```

---

## Schritt 3 — Docker-Container starten (Host)

```bash
cd /home/pascal/Documents/ml_on_mcu/MLonMCU_Project/cnn-trad-fpool3_model/platforms/gap9/Deeploy

docker run -it --rm --privileged --network host \
  -v "$PWD":/app/Deeploy \
  -v "$(dirname "$PWD")/measurements":/app/gap9_measurements \
  -v /dev/bus/usb:/dev/bus/usb \
  ghcr.io/pulp-platform/deeploy-gap9:latest \
  /bin/bash -lc '
    source /app/install/gap9-sdk/.gap9-venv/bin/activate
    source /app/install/gap9-sdk/configs/gap9_evk_audio.sh
    export GVSOC_INSTALL_DIR=/app/install/gap9-sdk/install/workstation
    export GAP_SDK_VERSION=dev
    sed -i "s/^ftdi_vid_pid.*/ftdi_vid_pid 0x0403 0x6011/" \
      /app/install/gap9-sdk/utils/openocd_tools/tcl/gapuino_ftdi.cfg
    exec /bin/zsh
  '
```

Du landest in einer zsh-Shell im Container. Alle weiteren Commands laufen darin.

---

## Schritt 4 — Deeploy installieren (im Container, einmal pro Container-Start)

```bash
cd /app/Deeploy
pip install -e .
```

> **Muss bei jedem Container-Start wiederholt werden** — der Container ist ephemer (`--rm`),
> die Installation überlebt den `exit` nicht.

---

## Schritt 5 — Tests laufen lassen (im Container)

```bash
cd /app/Deeploy/DeeployTest
```

### Referenznetzwerk: MLPerf KeywordSpotting (INT8)

```bash
# GVSoC (kein Board nötig)
python deeployRunner_gap9.py -t Tests/Models/MLPerf/KeywordSpotting -s gvsoc

# Board
python deeployRunner_gap9.py -t Tests/Models/MLPerf/KeywordSpotting -s board
```

Erwartetes Ergebnis:
```
Errors: 0 out of 12
Runtime: ~2,980,000 cycles
✓ Test KeywordSpotting PASSED
```

---

### Unser DS-CNN-L (Float32, "left"-Sample)

```bash
# GVSoC
python deeployRunner_gap9.py -t Tests/Models/DSCNNL -s gvsoc

# Board
python deeployRunner_gap9.py -t Tests/Models/DSCNNL -s board
```

Erwartetes Ergebnis:
```
Errors: 0 out of 12
Runtime: ~27,620,000 cycles
Predicted class: 2 (confidence 0.9998)   ← "Left"
✓ Test DSCNNL PASSED
```

---

### Live-Inferenz: Mikrofon-Keyword-Spotting auf dem Board

Erkennt Schlüsselwörter in Echtzeit über das PDM-Mikrofon (Vesper, SAI1).
Klassen: `down go left no off on right stop up yes silence unknown`

```bash
cd /app/Deeploy/DeeployTest

# Standard (kein Debug-Output)
python deeployRunner_gap9.py -t Tests/Models/DSCNNL -s board -D LIVE_INFERENCE=ON

# Mit c0-Debug (alle 49 c0-Werte pro Fenster — für Fehlersuche)
python deeployRunner_gap9.py -t Tests/Models/DSCNNL -s board -D LIVE_INFERENCE=ON -D KWS_DEBUG_C0
```

Erwartete Ausgabe nach Start:
```
Initializing network...
Network ready.
Microphone ready.
-- Listening --
-- Listening --
```

Nach einem gesprochenen Keyword:
```
--- Detected (peak=800000000) peak_frame=65 (1300ms in 2s buf) ---
[win 0] peak@frame 14  start= 980ms  best=on       p=0.8500  [27835000 cycles]
[win 1] peak@frame 20  start= 900ms  best=on       p=0.9200  [27835000 cycles]
[win 2] peak@frame 26  start= 840ms  best=on       p=0.8900  [27835000 cycles]
>> on        (class 5, avg score 0.8867)
```

**Hinweise:**
- **Dashboard:** Das Ganze geht auch per Klick — im MCU-Dashboard
  (`platforms/dashboard/run_dashboard.sh`) den Slot auf *GAP9 EVK* stellen und
  **Build + Run** drücken. Voraussetzung bleibt Schritt 2 (ftdi_sio-Unbind)
  oder die einmalige udev-Regel `platforms/dashboard/99-gap9-ftdi.rules`.
- Nach `-- Listening --` hat man **2 Sekunden** Zeit das Keyword zu sprechen.
- Das System nimmt immer 2 s auf, sucht den Amplituden-Peak und probiert 3 verschiedene Fenster-Alignments (±120 ms um Frame 20). Die Softmax-Ausgaben werden gemittelt.
- Erkennung gilt nur bei avg score ≥ 0.60 (sonst `>> no detection`).
- Bei Rebuild nach Code-Änderungen denselben Befehl einfach nochmal ausführen — CMake erkennt Änderungen automatisch.

---

## Schritt 6 — Offline-Messung: 50 Inferences auf dem Board (im Container)

Nach dem ersten erfolgreichen Board-Run alle nötigen Dateien kompiliert — jetzt 50 Inferences laufen lassen und Stats automatisch speichern.

```bash
# Im Container (nach deeployRunner_gap9.py -t Tests/Models/DSCNNL -s board)
cd /app/Deeploy
python tools/kws20_measure_gap9_ds_cnn.py
```

Optionale Argumente:
```bash
python tools/kws20_measure_gap9_ds_cnn.py --runs 10       # nur 10 Runs (schneller)
python tools/kws20_measure_gap9_ds_cnn.py --runs 50       # default: 50
```

Ergebnis wird gespeichert unter:
`/app/gap9_measurements/offline_measurements/kws20_ds_cnn_l_float32_offline_YYYYMMDD_HHMMSS/`
→ Host: `platforms/gap9/measurements/offline_measurements/`

Inhalt:
- `summary.json`   — Zyklen-Stats (count/min/max/avg/median/std/p95), Latenz, MACs/Cycle, ELF-Größe
- `cycles_raw.csv` — Rohdaten: pro Run cycles / latency_us / errors / predicted_class
- `run_logs/`      — Rohausgabe von gapy pro Run (für Debugging)

---

## Schritt 8 — Test-Artefakte neu generieren (nur bei Änderungen, auf dem Host)

```bash
cd /home/pascal/Documents/ml_on_mcu/MLonMCU_Project/cnn-trad-fpool3_model/platforms/gap9

# Float32
python3 create_dscnnl_test.py
# → Deeploy/DeeployTest/Tests/Models/DSCNNL/network.onnx, inputs.npz, outputs.npz

# INT8 (PULP SIMD)
python3 create_dscnnl_int8_test.py
# → Deeploy/DeeployTest/Tests/Models/DSCNNL_INT8/network.onnx, inputs.npz, outputs.npz
```

---

### Unser DS-CNN-L (INT8 PULP SIMD)

Im Container:

```bash
# GVSoC (Simulator)
python deeployRunner_gap9.py -t Tests/Models/DSCNNL_INT8 -s gvsoc

# Board
python deeployRunner_gap9.py -t Tests/Models/DSCNNL_INT8 -s board
```

Erwartetes Ergebnis:
```
Errors: 0 out of 12
Runtime: ~4,120,000 cycles  (INT8 SIMD, ~6.7× schneller als Float32 ~27.6M)
Predicted class: 2 (logit 109)   ← "Left"
✓ Test DSCNNL_INT8 PASSED
```

---

## Schnellreferenz

```bash
# ── HOST ──────────────────────────────────────────────────────────────────────
lsusb | grep -i "0403:6011"          # Board prüfen

# ftdi_sio ablösen (einmalig nach Board einstecken)
for dev in /sys/bus/usb/devices/*; do
  [[ -f "$dev/idVendor" ]] || continue
  [[ "$(cat "$dev/idVendor")" == "0403" && "$(cat "$dev/idProduct")" == "6011" ]] || continue
  for iface in "$dev":*; do
    [[ -L "$iface/driver" ]] || continue
    [[ "$(basename "$(readlink "$iface/driver")")" == "ftdi_sio" ]] || continue
    echo "$(basename "$iface")" | sudo tee /sys/bus/usb/drivers/ftdi_sio/unbind
  done
done

# Docker starten
cd /home/pascal/Documents/ml_on_mcu/MLonMCU_Project/cnn-trad-fpool3_model/platforms/gap9/Deeploy
docker run -it --rm --privileged --network host \
  -v "$PWD":/app/Deeploy \
  -v "$(dirname "$PWD")/measurements":/app/gap9_measurements \
  -v /dev/bus/usb:/dev/bus/usb \
  ghcr.io/pulp-platform/deeploy-gap9:latest \
  /bin/bash -lc '
    source /app/install/gap9-sdk/.gap9-venv/bin/activate
    source /app/install/gap9-sdk/configs/gap9_evk_audio.sh
    export GVSOC_INSTALL_DIR=/app/install/gap9-sdk/install/workstation
    export GAP_SDK_VERSION=dev
    sed -i "s/^ftdi_vid_pid.*/ftdi_vid_pid 0x0403 0x6011/" \
      /app/install/gap9-sdk/utils/openocd_tools/tcl/gapuino_ftdi.cfg
    exec /bin/zsh
  '

# ── CONTAINER ─────────────────────────────────────────────────────────────────
cd /app/Deeploy && pip install -e .          # Deeploy installieren (jedes Mal!)
cd DeeployTest

python deeployRunner_gap9.py -t Tests/Models/MLPerf/KeywordSpotting -s board   # Referenz
python deeployRunner_gap9.py -t Tests/Models/DSCNNL -s board                          # DS-CNN-L Float32 (offline test)
python deeployRunner_gap9.py -t Tests/Models/DSCNNL -s board -D LIVE_INFERENCE=ON    # DS-CNN-L Live-Mikrofon
python deeployRunner_gap9.py -t Tests/Models/DSCNNL_INT8 -s board              # DS-CNN-L INT8

# Offline-Messung Float32 (20 Runs → summary.json + CSV)
cd /app/Deeploy
python tools/kws20_measure_gap9_ds_cnn.py

# Offline-Messung INT8 (20 Runs → summary.json + CSV)
python tools/kws20_measure_gap9_ds_cnn_int8.py
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `gap9-shell.sh` klappt nicht | Alle Commands aus Schritt 2+3 manuell ausführen |
| Board nicht erkannt (`lsusb` leer) | USB-Kabel, anderen Port versuchen |
| `ftdi_sio` noch aktiv nach Unbind | Board kurz aus-/einstecken, Schritt 2 wiederholen |
| `No module named 'Deeploy'` | `cd /app/Deeploy && pip install -e .` |
| `No module named 'coloredlogs'` | falsche Python-Env; `.gap9-venv` muss aktiv sein (Schritt 3) |
| OpenOCD Timeout / keine Verbindung | Board aus-/einstecken, Schritt 2 wiederholen |
| Semihost: nur erster printf | bereits gefixt (`load_and_start_binary` in `gap9_board.cmake`) |

---

## NNTool / NE16 — Docker-Container starten

Separater Container für NNTool/GAPflow (kein Deeploy nötig):

```bash
cd /home/pascal/Documents/ml_on_mcu/MLonMCU_Project/cnn-trad-fpool3_model/platforms/gap9

docker run -it --rm --privileged --network host \
  -v "$PWD/nn_profiling":/app/nn_profiling \
  -v "$PWD/../../models":/app/models \
  -v "$PWD/measurements":/app/gap9_measurements \
  -v /dev/bus/usb:/dev/bus/usb \
  ghcr.io/pulp-platform/deeploy-gap9:latest \
  /bin/bash -lc '
    source /app/install/gap9-sdk/.gap9-venv/bin/activate
    source /app/install/gap9-sdk/configs/gap9_evk_audio.sh
    export GAP_SDK_VERSION=dev
    sed -i "s/^ftdi_vid_pid.*/ftdi_vid_pid 0x0403 0x6011/" \
      /app/install/gap9-sdk/utils/openocd_tools/tcl/gapuino_ftdi.cfg
    pip install ipython --quiet
    exec /bin/zsh
  '
```

### NE16 Profiling (einmaliger Run, Board)

```bash
# Im Container
cd /app/nn_profiling
python nn_profiler.py /app/models/ds_cnn_l.onnx \
  --config configs/quant_ne16.yml \
  --target-config configs/target_config_opt.yml \
  --platform board \
  --exp-label ne16_int8_board
```

Erwartetes Ergebnis:
```
Total: 188,646 cycles  (~0.79 ms @ 240 MHz)
MACs/cycle: 20.32
```

### NE16 Offline-Messung: 20 Inferences + "Left"-Verifikation

```bash
# Im Container (nach erstem board-Run damit Binary gecacht ist)
cd /app/nn_profiling
python ne16_measure.py --config configs/ne16_measure_config.json
```

Optionale Argumente:
```bash
python ne16_measure.py --runs 5     # Quick-Test
python ne16_measure.py --runs 20    # default: 20
```

**Wichtig**: Erster Run kompiliert (~5-6 Min). Folge-Runs nutzen den Build in `/tmp/nn_profiler_ne16/`
(make sieht keine Änderungen → nur gapy läuft → schnell).

Ergebnis wird gespeichert unter:
`/app/gap9_measurements/offline_measurements/ds_cnn_l_ne16_int8_offline_YYYYMMDD_HHMMSS/`
→ Host: `platforms/gap9/measurements/offline_measurements/`

---

## Schritt 6b — Offline-Messung INT8 SIMD: 20 Inferences auf dem Board (im Container)

Nach dem ersten erfolgreichen `board`-Run für INT8 läuft das Binary bereits. Jetzt 20 Inferences messen:

```bash
# Im Container (nach deeployRunner_gap9.py -t Tests/Models/DSCNNL_INT8 -s board)
cd /app/Deeploy
python tools/kws20_measure_gap9_ds_cnn_int8.py
```

Optionale Argumente:
```bash
python tools/kws20_measure_gap9_ds_cnn_int8.py --runs 10       # nur 10 Runs (schneller)
python tools/kws20_measure_gap9_ds_cnn_int8.py --runs 20       # default: 20
```

Ergebnis wird gespeichert unter:
`/app/gap9_measurements/offline_measurements/kws20_ds_cnn_l_int8_offline_YYYYMMDD_HHMMSS/`
→ Host: `platforms/gap9/measurements/offline_measurements/`

---

## Metriken DS-CNN-L auf GAP9 (Float32, Board)

| Metrik | Wert |
|--------|------|
| Cycles | 27,621,337 |
| Latenz @ 240 MHz | ~115 ms |
| MACs | 3,824,768 (3.82 M) |
| L2 Speicher | ~180 KB |
| Fehler | 0 / 12 ✓ |
| Predicted class | 2 = "Left" (confidence 0.9998) ✓ |

---

## Metriken DS-CNN-L auf GAP9 (INT8 SIMD, Board)

| Metrik | Wert |
|--------|------|
| Cycles | 4,120,265 |
| Latenz @ 240 MHz | ~17.2 ms |
| MACs | 3,824,768 (3.82 M) |
| Fehler | 0 / 12 ✓ |
| Predicted class | 2 = "Left" (logit 109) ✓ |
| **Speedup vs Float32** | **~6.7×** |

---

## Metriken DS-CNN-L auf GAP9 (INT8 NE16 Accelerator, Board)

Gemessen mit NNTool/GAPflow, `use_ne16: true`, SQ8-Quantisierung.
Experiment: `nn_profiling/experiments/ds_cnn_l_quant_ne16_ne16_int8_board_20260609_172228/`

| Metrik | Wert |
|--------|------|
| Cycles (Board) | 188,646 |
| Cycles (GVSoC) | 184,055 (2.5% Abweichung) |
| Latenz @ 240 MHz | ~0.79 ms |
| MACs | 3,832,460 |
| MACs/Cycle | 20.32 |
| L1 Speicher (Shared) | 20.0 KB / 119 KB (16.8%) |
| L2 / eMRAM (Gewichte) | 36.4 KB |
| **Speedup vs Float32** | **~146×** |
| **Speedup vs INT8 SIMD** | **~21.8×** |

### Layer-Breakdown (NE16, Board)

| Layer | Cycles | Ops | Ops/Cycle |
|-------|--------|-----|-----------|
| Conv2D (initial) | 12,950 | 320,000 | 24.71 |
| DW Conv × 6 | ~17,700 | 72,000 | ~4.1 |
| PW Conv × 6 | ~10,700 | 512,000 | ~47–48 |
| AvgPool | 2,843 | 7,680 | 2.70 |
| Dense | 1,252 | 768 | 0.61 |
| Softmax | 1,445 | 12 | 0.01 |
| **Total** | **188,646** | **3,832,460** | **20.32** |

> Bottleneck: Depthwise Convolutions (DW) laufen mit nur ~4 Ops/Cycle auf NE16 —
> NE16 ist für pointwise/dense Convolutions optimiert; DW sind weniger effizient.
> Pointwise Convs erreichen ~48 Ops/Cycle (sehr gut).
