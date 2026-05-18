cmake_minimum_required(VERSION 3.13)

# Standalone app for coralmicro SDK.
# See README.md for build instructions.
project(kws_coral)

add_executable_m7(kws_coral
    kws_coral.cc
    model_data.cc   # Edge TPU model embedded as C array (xxd -i generated)
)

target_include_directories(kws_coral PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(kws_coral
    libs_base-m7_freertos
    libs_tpu_edgetpu_manager-m7
    libs_tpu_edgetpu_op-m7
    tensorflow-m7
)
