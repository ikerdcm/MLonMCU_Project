# MODEL_PATH=~/Work/gap_sdk/tools/nntool/tests/graph/mobilenet_v1_0_25_128_quant.tflite
MODEL_PATH=../vww_tinymlperf/vww_mlperf_quantized.tflite

python nn_profiler.py $MODEL_PATH --load-tflite-quant --scheme sq8  --target-config configs/target_config_no_l1_promo.yml --exp-label no_l1_promo
python nn_profiler.py $MODEL_PATH --load-tflite-quant --scheme sq8  --target-config configs/target_config_opt.yml         --exp-label l1_promo
python nn_profiler.py $MODEL_PATH --load-tflite-quant --scheme ne16 --target-config configs/target_config_no_l1_promo.yml --exp-label no_l1_promo
python nn_profiler.py $MODEL_PATH --load-tflite-quant --scheme ne16 --target-config configs/target_config_opt.yml         --exp-label l1_promo
