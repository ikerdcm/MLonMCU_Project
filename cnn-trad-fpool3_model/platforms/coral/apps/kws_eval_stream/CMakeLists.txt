# Streaming device-in-the-loop accuracy eval (Edge-TPU, v2 INT8) — receives raw
# audio from the host (testbench.py), runs Coral's MFCC frontend + Edge TPU.
add_executable_m7(kws_eval_stream
    kws_eval_stream.cc
    ${CMAKE_CURRENT_LIST_DIR}/../kws_live/kws_mfcc.cc
    ${CMAKE_CURRENT_LIST_DIR}/../kws_live/kws_mfcc_tables_ocram.cc
    ${CMAKE_CURRENT_LIST_DIR}/../kws_live/kws_mfcc_tables_sdram.cc
    DATA
    ${CMAKE_CURRENT_LIST_DIR}/../../model/ds_cnn_l_static_v2_edgetpu.tflite>/models/ds_cnn_l_static_v2_edgetpu.tflite
    ${CMAKE_CURRENT_LIST_DIR}/../../eval/audio_set.bin>/eval/audio_set.bin
)

target_include_directories(kws_eval_stream PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../common
    ${CMAKE_CURRENT_LIST_DIR}/../kws_live
)

target_link_libraries(kws_eval_stream
    libs_base-m7_freertos
)
