# DS-CNN unified test bench

One command per `(board, model)` → a **normalized** record: on-device
**accuracy + 12-class confusion matrix + inference latency + flash/SRAM**,
all from a single device-in-the-loop run. Everything new lives in one place:

```text
results/
  RESULTS_LEDGER.md          # NEW ledger (separate from Experiments/); 1 unique row per model
  max_v1/  summary.json  confusion_matrix.png
  max_v0/  summary.json  confusion_matrix.png
  ...                        # one <board>_<model>/ per MCU+model, overwritten on re-run
```

Re-running a model **replaces** its folder + its ledger row (unique latest result).
The old `Experiments/RESULTS_LEDGER.md` is never touched.

## How it works
The host streams the **same Google Speech Commands test audio** to every board;
each board runs its **own frontend + model** and returns the prediction (+ `cnn_us`).
Raw audio (not MFCC) is the common input because each model is *scale-locked to its
own frontend's MFCC* — and audio also exercises the frontend, where bugs hide.

## Run
1. **Flash the board's EVAL firmware** (`KWS20_CFG_ENABLE_EVAL=1`, then the board's
   build+flash script — the tool prints the exact `flash_hint`). Hardware step = yours.
2. **Run the bench**:
   ```bash
   python3 testbench.py --board max --model v1 --per-class 12
   python3 testbench.py --board max --model v0 --per-class 12
   ```
   `--no-serial` builds + checks the subset only (no board). `--port` overrides
   autodetect. `--per-class N` sets clips/class (12 ≈ Coral's 144-sample eval).

## Memory note
By default flash/SRAM are read from the **EVAL** build's ELF (which carries the
eval harness) — flagged in the output. For the *deployed* footprint, pass
`--deployed-elf <path-to-offline/live-build.elf>`.

## Adding a board
Add an entry to `REGISTRY` in `testbench.py` (name, config id, firmware dir, ELF
path, label order, flash hint) and give that firmware an EVAL mode that speaks the
protocol (see `keyword_spotting_max78000_*/kws20_eval.c`). Next up: U5, Coral.
