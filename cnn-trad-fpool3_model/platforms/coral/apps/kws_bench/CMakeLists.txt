add_executable_m7(kws_bench
    kws_bench.cc
    DATA
    ${CMAKE_CURRENT_LIST_DIR}/../../model/ds_cnn_l_static_edgetpu.tflite>/models/ds_cnn_l_static_edgetpu.tflite
)

target_link_libraries(kws_bench
    libs_base-m7_freertos
)
