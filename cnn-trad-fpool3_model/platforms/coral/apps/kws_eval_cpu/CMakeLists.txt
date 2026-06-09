# On-device accuracy eval for the FP32 CPU model (config: fp32-cpu). No Edge TPU.
# Loads the test subset from LittleFS (float kernels leave no room to embed it).
add_executable_m7(kws_eval_cpu
    kws_eval_cpu.cc
    DATA
    ${CMAKE_CURRENT_LIST_DIR}/../../../../models/ds_cnn_l_float.tflite>/models/ds_cnn_l_float.tflite
    ${CMAKE_CURRENT_LIST_DIR}/../common/ds_cnn_eval_set.bin>/eval/eval_set.bin
)

target_link_libraries(kws_eval_cpu
    libs_base-m7_freertos
)
