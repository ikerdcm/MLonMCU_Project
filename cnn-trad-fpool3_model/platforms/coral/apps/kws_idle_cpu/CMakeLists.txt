# Duty-cycled power firmware (config: fp32-cpu, NO Edge TPU): ONE offline
# inference every kPeriodMs with the M7 in WFI sleep between. For power-meter
# measurement of the active-vs-idle duty cycle. Model from the canonical models/.
add_executable_m7(kws_idle_cpu
    kws_idle_cpu.cc
    DATA
    ${CMAKE_CURRENT_LIST_DIR}/../../../../models/ds_cnn_l_float.tflite>/models/ds_cnn_l_float.tflite
)

target_include_directories(kws_idle_cpu PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../common
)

target_link_libraries(kws_idle_cpu
    libs_base-m7_freertos
)
