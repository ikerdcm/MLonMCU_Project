# On-device accuracy eval (device-in-the-loop) for the v2 INT8 Edge-TPU model.
add_executable_m7(kws_eval
    kws_eval.cc
    DATA
    ${CMAKE_CURRENT_LIST_DIR}/../../model/ds_cnn_l_static_v2_edgetpu.tflite>/models/ds_cnn_l_static_v2_edgetpu.tflite
)

target_include_directories(kws_eval PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../common
)

target_link_libraries(kws_eval
    libs_base-m7_freertos
)
