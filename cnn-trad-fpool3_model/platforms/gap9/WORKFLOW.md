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
python deeployRunner_gap9.py -t Tests/Models/DSCNNL -s board                   # DS-CNN-L Float32
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

## Schritt 6b — Offline-Messung INT8: 20 Inferences auf dem Board (im Container)

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
