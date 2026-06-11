# Duty-cycled power firmware (Edge TPU): ONE offline inference every kPeriodMs
# with the M7 in WFI sleep between. For power-meter measurement of the
# active-vs-idle duty cycle. TPU twin of kws_idle_cpu (which is fixed to v0).
#
# Which TPU network gets baked is selectable (any of v1/v21/v22/v23/v31/v32),
# mirroring kws_eval_stream — set by build_and_flash_idle.sh --version vNN:
#   cmake ... -DKWS_IDLE_MODEL=ds_cnn_l_pruned_f64b4_int8_edgetpu -DKWS_IDLE_VERSION=v21
# Always baked under the fixed LittleFS name /models/idle_model_edgetpu.tflite.
# Default = v1 (6-block ds_cnn_l_static_v2).
set(KWS_IDLE_MODEL "ds_cnn_l_static_v2_edgetpu"
    CACHE STRING "edgetpu model basename in ../../model to bake for kws_idle")
set(KWS_IDLE_VERSION "v1"
    CACHE STRING "version id of the baked network (one network per version)")

add_executable_m7(kws_idle
    kws_idle.cc
    DATA
    ${CMAKE_CURRENT_LIST_DIR}/../../model/${KWS_IDLE_MODEL}.tflite>/models/idle_model_edgetpu.tflite
)

# Bake the chosen version/model names in so the firmware self-identifies on boot.
target_compile_definitions(kws_idle PRIVATE
    KWS_IDLE_MODEL_NAME="${KWS_IDLE_MODEL}"
    KWS_IDLE_VERSION_NAME="${KWS_IDLE_VERSION}")

target_include_directories(kws_idle PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../common
)

target_link_libraries(kws_idle
    libs_base-m7_freertos
)
