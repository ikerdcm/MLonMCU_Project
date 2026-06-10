# FP32 CPU live KWS (config: fp32-cpu, online). No Edge TPU.
# Reuses the MFCC front-end + tables from the kws_live app (no duplication).
# Model from the canonical models/ dir: ../../../../models == cnn-trad-fpool3_model/models
add_executable_m7(kws_live_cpu
    kws_live_cpu.cc
    ${CMAKE_CURRENT_LIST_DIR}/../kws_live/kws_mfcc.cc
    ${CMAKE_CURRENT_LIST_DIR}/../kws_live/kws_mfcc_tables_ocram.cc
    ${CMAKE_CURRENT_LIST_DIR}/../kws_live/kws_mfcc_tables_sdram.cc
    DATA
    ${CMAKE_CURRENT_LIST_DIR}/../../../../models/ds_cnn_l_float.tflite>/models/ds_cnn_l_float.tflite
)

target_include_directories(kws_live_cpu PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../kws_live
    ${CMAKE_CURRENT_LIST_DIR}/../common
)

target_link_libraries(kws_live_cpu
    libs_base-m7_freertos
    libs_audio_freertos
)
