# U5 keyword-spotting analysis tools

## compare_dumps.py

Compares an STM32U5 live AI input dump against the MAX78000 reference dump
(both 16384 int8 samples, the actual tensor fed to the model).  Use this to
diagnose where the live-mic accuracy gap comes from -- since the U5 model
itself is verified correct (TEST 2 in the offline harness PASSes the MAX78000
vector), the only remaining variable is the captured audio.

### Capture procedure

1. Build & flash the firmware (kws20_live.c already calls `dump_live_i8_to_uart()`
   right before `run_inference()`, so every triggered keyword produces an
   `AI_INPUT_DUMP_BEGIN..END` block on the UART).

2. Start a serial logger at 115200 baud and **capture into a file**, e.g.:

       picocom -b 115200 /dev/ttyACM0 | tee logs/my_capture.log
       # or
       screen -L -Logfile logs/my_capture.log /dev/ttyACM0 115200

3. Say a keyword (e.g. "five") into the on-board MEMS mic.

4. **Wait at least 10 seconds** after the trigger fires before stopping the
   capture -- the dump itself is ~7 s at 115200 baud (16384 values × ~5 chars).
   Stopping early truncates the dump.

5. Stop the logger, save the `.log`.

### Inspecting captures

List all dumps in a log with their predictions:

    python3 tools/compare_dumps.py logs/my_capture.log --list

Compare a specific dump against the MAX78000 reference:

    python3 tools/compare_dumps.py logs/my_capture.log --u5-dump-index 0

Pick a specific MAX78000 reference dump and save plots:

    python3 tools/compare_dumps.py logs/my_capture.log \
        --max-dump-index 0 --save logs/my_capture_dump0

### What to look at

Console output:
- **Histogram & summary**: how close is the U5 distribution (zero%, peak,
  RMS) to the MAX reference?  TEST 2 PASSed when fed the MAX reference, so
  if your live capture matches the MAX distribution closely and STILL gets
  predicted wrong, the gap is in the spectrum, not the levels.
- **Band energy in dB**: 6 frequency bands (0-0.25, 0.25-1, 1-2, 2-4, 4-6,
  6-8 kHz).  The `U5-MAX` column is the smoking gun:
  - Negative deltas in 4-8 kHz => SINC4 droop (fricatives lose energy).
  - Positive deltas everywhere => U5 simply louder (gain too high).
  - HF much hotter than LF => U5 has excess high-frequency content
    (mic noise / quantisation noise / aliasing).
- **Interpretation**: a few rule-based hints generated from the deltas.

Plots (saved as `<prefix>_compare.png` if `--save` is given):
- Waveform overlay
- Power spectrum (FFT, dB)
- Smoothed delta spectrum (U5 - MAX, dB)
- Spectrograms side-by-side

### Known reference

`ai85kws20netv3_model/test_vectors/max78000_ai_input_dump.log` contains four
captured "five" utterances on the MAX78000.  Dump #0 is what the offline
test (TEST 2) uses and is the canonical "this is what the model expects"
input.
