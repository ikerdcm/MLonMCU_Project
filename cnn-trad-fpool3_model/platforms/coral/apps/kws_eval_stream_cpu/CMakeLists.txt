# Streaming device-in-the-loop accuracy eval, fp32 CPU (M7, NO Edge TPU).
# Receives raw audio from the host (testbench.py), runs Coral's MFCC frontend +
# the float DS-CNN-L on the Cortex-M7. Model from cnn-trad-fpool3_model/models.
add_executable_m7(kws_eval_stream_cpu
    kws_eval_stream_cpu.cc
    ${CMAKE_CURRENT_LIST_DIR}/../kws_live/kws_mfcc.cc
    ${CMAKE_CURRENT_LIST_DIR}/../kws_live/kws_mfcc_tables_ocram.cc
    ${CMAKE_CURRENT_LIST_DIR}/../kws_live/kws_mfcc_tables_sdram.cc
    DATA
    ${CMAKE_CURRENT_LIST_DIR}/../../../../models/ds_cnn_l_float.tflite>/models/ds_cnn_l_float.tflite
    ${CMAKE_CURRENT_LIST_DIR}/../../eval/audio_set.bin>/eval/audio_set.bin
)

target_include_directories(kws_eval_stream_cpu PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../kws_live
    ${CMAKE_CURRENT_LIST_DIR}/../common
)

target_link_libraries(kws_eval_stream_cpu
    libs_base-m7_freertos
)
