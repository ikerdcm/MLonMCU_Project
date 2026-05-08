This folder is a separated MAX78000 demo variant for the width-reduced `w90` KWS20v3 model.

Intended usage:
- keep `kws20_demo/` as the baseline demo
- use `kws20_demo_w90/` for the `w90` network integration and measurements

To finalize this variant, replace these generated network files with the `w90`
artifacts from `ai8x-synthesis/sdk/Examples/MAX78000/CNN/kws20_v3_w90/`:
- `cnn.c`
- `cnn.h`
- `weights.h`
