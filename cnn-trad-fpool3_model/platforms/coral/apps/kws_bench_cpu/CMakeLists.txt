# FP32 CPU offline benchmark (config: fp32-cpu). No Edge TPU.
# Model comes from the canonical models/ dir (no edgetpu compile step needed):
#   ../../../../models == cnn-trad-fpool3_model/models
add_executable_m7(kws_bench_cpu
    kws_bench_cpu.cc
    DATA
    ${CMAKE_CURRENT_LIST_DIR}/../../../../models/ds_cnn_l_float.tflite>/models/ds_cnn_l_float.tflite
)

target_include_directories(kws_bench_cpu PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../common
)

target_link_libraries(kws_bench_cpu
    libs_base-m7_freertos
)
