add_executable_m7(kws_live
    kws_live.cc
    kws_mfcc.cc
    kws_mfcc_tables_ocram.cc
    kws_mfcc_tables_sdram.cc
    DATA
    ${CMAKE_SOURCE_DIR}/models/ds_cnn_l_static_edgetpu.tflite
)

target_link_libraries(kws_live
    libs_base-m7_freertos
    libs_audio_freertos
)
