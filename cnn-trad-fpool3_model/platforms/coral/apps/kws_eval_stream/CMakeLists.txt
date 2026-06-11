# Streaming device-in-the-loop accuracy eval (Edge-TPU, INT8) — receives raw
# audio from the host (testbench.py), runs Coral's MFCC frontend + Edge TPU.
#
# Which Edge-TPU model gets baked is selectable at configure time:
#   cmake ... -DKWS_EVAL_STREAM_MODEL=ds_cnn_l_pruned_f64b4_int8_edgetpu
# It is always baked under the fixed LittleFS name /models/eval_model_edgetpu.tflite
# (kModelPath in the .cc), so only this variable changes between variants.
# Default = the v2 6-block model (the int8-accel / coral v1 row).
set(KWS_EVAL_STREAM_MODEL "ds_cnn_l_static_v2_edgetpu"
    CACHE STRING "edgetpu model basename in ../../model to bake for kws_eval_stream")
# Honest version id of the baked network (v1/v21/v22/v23/v31/v32) — reported by
# the firmware so the board self-identifies. Set by build_and_flash_eval_stream.sh.
set(KWS_EVAL_STREAM_VERSION "v1"
    CACHE STRING "version id of the baked network (one network per version)")

add_executable_m7(kws_eval_stream
    kws_eval_stream.cc
    ${CMAKE_CURRENT_LIST_DIR}/../kws_live/kws_mfcc.cc
    ${CMAKE_CURRENT_LIST_DIR}/../kws_live/kws_mfcc_tables_ocram.cc
    ${CMAKE_CURRENT_LIST_DIR}/../kws_live/kws_mfcc_tables_sdram.cc
    DATA
    ${CMAKE_CURRENT_LIST_DIR}/../../model/${KWS_EVAL_STREAM_MODEL}.tflite>/models/eval_model_edgetpu.tflite
    ${CMAKE_CURRENT_LIST_DIR}/../../eval/audio_set.bin>/eval/audio_set.bin
)

# Bake the chosen model's basename into the firmware so it self-reports which
# model is running (the LittleFS file is always the generic eval_model name).
target_compile_definitions(kws_eval_stream PRIVATE
    KWS_EVAL_STREAM_MODEL_NAME="${KWS_EVAL_STREAM_MODEL}"
    KWS_EVAL_STREAM_VERSION_NAME="${KWS_EVAL_STREAM_VERSION}")

target_include_directories(kws_eval_stream PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../common
    ${CMAKE_CURRENT_LIST_DIR}/../kws_live
)

target_link_libraries(kws_eval_stream
    libs_base-m7_freertos
)
