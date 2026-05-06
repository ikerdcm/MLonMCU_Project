This folder is the isolated quantization workspace for the U5 int8 model flow.

It does not touch:
- `platforms/stm32u5/u5_keyword_spotting`
- `platforms/stm32u5/u5_keyword_spotting_quantized`

Main files:
- `prepare_kws20_u5_calibration_config.json`
- `prepare_kws20_u5_calibration_set.py`
- `prepare_kws20_u5_calibration_set.sh`
- `quantize_kws20_u5_onnx_config.json`
- `quantize_kws20_u5_onnx.py`
- `quantize_kws20_u5_onnx.sh`

Typical flow:
```bash
bash platforms/stm32u5/quantized/prepare_kws20_u5_calibration_set.sh
bash platforms/stm32u5/quantized/quantize_kws20_u5_onnx.sh
```

Output files:
- `calibration_u5_current_model_nhwc.npz`
- `calibration_u5_current_model_nchw.npz`
- `calibration_u5_current_model_metadata.json`
- `kws20_v3_qat_true2d_nchw_allconst_folded_for_cubeai_int8_qdq.onnx`
- `kws20_v3_qat_true2d_nchw_allconst_folded_for_cubeai_int8_qdq_report.json`
