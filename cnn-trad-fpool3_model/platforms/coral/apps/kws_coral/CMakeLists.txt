add_executable_m7(kws_coral
    kws_coral.cc
    DATA
    ${CMAKE_CURRENT_LIST_DIR}/../../model/ds_cnn_l_static_edgetpu.tflite>/models/ds_cnn_l_static_edgetpu.tflite
)

target_include_directories(kws_coral PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../common
)

target_link_libraries(kws_coral
    libs_base-m7_freertos
)
