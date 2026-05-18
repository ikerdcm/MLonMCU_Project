add_executable_m7(kws_coral
    kws_coral.cc
    DATA
    ${CMAKE_SOURCE_DIR}/models/ds_cnn_l_static_edgetpu.tflite
)

target_link_libraries(kws_coral
    libs_base-m7_freertos
)
