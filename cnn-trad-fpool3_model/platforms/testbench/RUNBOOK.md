# DS-CNN accuracy test bench — runbook

How to run the **device-in-the-loop accuracy eval** for every MCU + model. Each
run streams (or bakes, for Coral) the **same GSC test audio** through the board's
**own frontend + model** and tallies on-device **accuracy + confusion matrix +
latency + flash/SRAM** into one place:

```text
testbench/results/
  RESULTS_LEDGER.md          # one unique row per (board, model); re-running replaces it
  <board>_<model>/  summary.json  confusion_matrix.png
```

All paths below are relative to the repo root. Find the board's serial port with
`ls /dev/tty.usbmodem*` (pass it as `--port`; needed when >1 board is attached).

| Board | model | config | EVAL mechanism |
|---|---|---|---|
| max | v1 | int8-accel (Edge accel) | stream audio over UART |
| max | v0 | fp32-cpu | stream audio over UART |
| u5  | v1 | int8-cpu (CMSIS-NN)     | stream over HAL UART |
| u5  | v0 | fp32-cpu (X-CUBE-AI)    | stream over HAL UART |
| coral | v1 | int8 Edge-TPU         | baked LittleFS audio set |
| coral | v0 | fp32 M7-CPU           | baked LittleFS audio set |

> The MAX/U5 EVAL firmware is gated by `KWS20_CFG_ENABLE_EVAL 1` in that folder's
> `Core/Inc/kws20_mode_config.h` (already set). The build+flash steps below produce
> the EVAL firmware; set it back to `0` to return to the normal live/offline build.

---

## MAX78000

```bash
# v1 — int8 + CNN accelerator
cd cnn-trad-fpool3_model/platforms/max78000/keyword_spotting_max78000_ds_cnn_8-bit_with-acc
./tools/build_flash_offline.sh                 # build + flash the EVAL firmware
python3 ../../testbench/testbench.py --board max --model v1 --per-class 12

# v0 — fp32 float CPU port  (~2048 ms/inference)
cd ../keyword_spotting_max78000_ds_cnn_32-bit
./tools/build_flash_max78000_ds_cnn.sh         # build + flash the EVAL firmware
python3 ../../testbench/testbench.py --board max --model v0 --per-class 12
```

- MAX has its own debug probe, so the testbench usually autodetects the port; add
  `--port /dev/tty.usbmodemXXXX` if needed.
- `--per-class 12` → 144 clips (12/class), matching Coral's set. Use `--per-class 5`
  for a quick check.

## STM32U5

```bash
# v1 — int8 / CMSIS-NN
cd cnn-trad-fpool3_model/platforms/stm32u5/keyword_spotting_u5_ds_cnn_8-bit
./tools/build_flash.sh --mode offline          # CMake build + flash
python3 ../../testbench/testbench.py --board u5 --model v1 --per-class 12 --port /dev/tty.usbmodemXXXX

# v0 — fp32 / X-CUBE-AI
cd ../keyword_spotting_u5_ds_cnn_32-bit
./tools/build_flash.sh --mode offline          # headless STM32CubeIDE build + flash
python3 ../../testbench/testbench.py --board u5 --model v0 --per-class 12 --port /dev/tty.usbmodemXXXX
```

- `--mode` only sets a compile flag; `ENABLE_EVAL` overrides the dispatch, so the
  board runs EVAL regardless. (`--mode offline` is the correct one — it compiles
  cleanly after the v0 `labels` fix.)
- **Always pass `--port`** (the U5 is an ST-Link VCP — autodetect can grab the wrong
  `usbmodem` when other boards are attached).

## Coral Dev Board Micro

Coral's console drops host→device USB bytes, so we **bake** the subset into LittleFS
instead of streaming. **Bake first**, then build+flash (the build copies the current
`eval/audio_set.bin` into LittleFS), then read the board's autonomous pass.

```bash
cd cnn-trad-fpool3_model/platforms/coral
python3 ../testbench/make_eval_audio_set.py --per-class 12   # → eval/audio_set.bin

# v1 — Edge-TPU int8
./scripts/build_and_flash_eval_stream.sh
python3 ../testbench/testbench.py --board coral --model v1 --port /dev/tty.usbmodemXXXX

# v0 — M7 CPU fp32  (~417 ms/inference)
./scripts/build_and_flash_eval_stream_cpu.sh
python3 ../testbench/testbench.py --board coral --model v0 --port /dev/tty.usbmodemXXXX
```

- **Bake before build.** Re-run `make_eval_audio_set.py --per-class N` to change the
  set size; the build bakes whatever `eval/audio_set.bin` currently exists.
- **No `--per-class` on the testbench for Coral** — the baked set sets N. The board
  runs it autonomously and self-reports `true_idx`+`pred_idx`+`cnn_us`; the testbench
  just reads one pass.
- The board **re-runs every ~8 s**. The testbench waits for the next pass start
  (≤ ~70 s for v0). It prints a `(board alive, mid-pass …)` heartbeat while waiting.
- ⚠️ Never open the Coral at 1200 baud (→ bootloader). The testbench uses 115200 and
  asserts DTR automatically.

### Coral optimization variants — honest version ids (ONE network per version)

Every Coral int8 variant runs the **same `kws_eval_stream` firmware** (same Edge-TPU +
MFCC frontend); only the **single baked network** changes, selected by its version id.
Flash and read use the **same `vNN`** — and the board self-reports its identity in the
`eval_ready` line (`version=…,model=…`), which the testbench cross-checks (it aborts on
a mismatch). Audio set is shared; bake once with `--per-class 12` if missing.

| Version | Config | Network |
|---|---|---|
| `v1`  | int8-accel        | 6-block `ds_cnn_l_static_v2` |
| `v21` | int8-prune        | f64b4 (4-blk, 64f) — prune winner |
| `v22` | int8-prune        | f32b6 (6-blk, 32f) |
| `v23` | int8-prune        | f32b4 (4-blk, 32f) |
| `v31` | int8-prune-distill| f64b4+KD (ladder top) |
| `v32` | int8-prune-distill| f32b4+KD (32-filter rescue) |

```bash
cd cnn-trad-fpool3_model/platforms/coral

# one network at a time — flash vNN, then read vNN
./scripts/build_and_flash_eval_stream.sh --version v21
python3 ../testbench/testbench.py --board coral --model v21 --port /dev/tty.usbmodemXXXX
# ...repeat for v22 v23 v31 v32. Return to the int8 baseline with --version v1.
```

- The baked network is re-selected at CMake-configure time on every
  `build_and_flash_eval_stream.sh` run — just change `--version`, no file edits.
- All variants share `v1`'s input scale (0.5847/83), so the device MFCC
  (`KwsMfccCompute`) feeds them correctly — no rescaling (prune validated by the older
  `kws_eval` ⁿ numbers; distill `v31/v32` are the **first on-device measurement**, their
  Edge-TPU models compiled via `scripts/compile_edgetpu_v2.sh`, all ops on-TPU).

---

## Output & flags

- Results: `testbench/results/<board>_<model>/summary.json` + `confusion_matrix.png`;
  one upserted row per model in `testbench/results/RESULTS_LEDGER.md`. The old
  `Experiments/RESULTS_LEDGER.md` is never touched.
- `--no-serial` builds/checks the subset only (no board).
- `--deployed-elf <path>` to report flash/SRAM from the deployed (not EVAL) build.
- `--dataset <path>` if the GSC test set isn't at the default
  `~/MAX78000_Toolchain/ai8x-training/data/KWS/raw_test`.

## Add a new (board, model)

Add an entry to `REGISTRY` in `testbench.py` (name, config_id, firmware dir, ELF
path, `assert_dtr`/`embedded` if applicable, `flash_hint`) and give that firmware an
EVAL mode speaking the protocol:

```text
host  : "EVAL <idx> <nsamples>\n"  then  nsamples * int16 little-endian   (streaming)
board : "BENCH,event=eval,idx=<i>,pred_idx=<p>[,true_idx=<t>],cnn_us=<us>\r\n"
```
