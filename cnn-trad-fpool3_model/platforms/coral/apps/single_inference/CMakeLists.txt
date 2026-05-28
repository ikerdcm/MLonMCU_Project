add_executable_m7(single_inference
    single_inference.cc
    DATA
    ${CMAKE_SOURCE_DIR}/models/ds_cnn_l_static_edgetpu.tflite
)

target_link_libraries(single_inference
    libs_base-m7_freertos
)
