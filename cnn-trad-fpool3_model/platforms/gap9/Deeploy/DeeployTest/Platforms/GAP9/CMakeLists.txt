# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

set(ProjectId ${TESTNAME})

set(${SDKCONFIG_FILE} ${CMAKE_CURRENT_LIST_DIR}/sdk.config)

# Path to the MFCC library (inside Deeploy/, accessible in Docker).
# From:  /app/Deeploy/DeeployTest/Platforms/GAP9/
# To:    /app/Deeploy/mfcc/
set(MFCC_DIR ${CMAKE_CURRENT_LIST_DIR}/../../../mfcc)

# ── Source selection ──────────────────────────────────────────────────────────
if(LIVE_INFERENCE)
    # Live microphone inference: MFCC + SFU PDM mic + live deeploytest
    set(SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/src/CycleCounter.c
        ${CMAKE_CURRENT_LIST_DIR}/src/deeploytest_live.c
        ${MFCC_DIR}/src/mfcc.c
        ${MFCC_DIR}/src/mfcc_tables.c
        ${MFCC_DIR}/src/mic.c
    )
else()
    # Offline / power-measurement mode (default)
    file(GLOB_RECURSE SOURCES
        src/CycleCounter.c
        src/deeploytest.c
    )
endif()

add_deeploy_executable(${ProjectId} EXCLUDE_FROM_ALL ${SOURCES})
# add_executable(${ProjectId} ${SOURCES})

# ── Include directories ───────────────────────────────────────────────────────
target_include_directories(${ProjectId} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/inc)

if(LIVE_INFERENCE)
    target_include_directories(${ProjectId} PRIVATE ${MFCC_DIR}/include)
    target_compile_definitions(${ProjectId} PRIVATE LIVE_INFERENCE)

    # Verbose frontend diagnostics (per-window results, probability table,
    # c0 trace).  Enable with:  -D KWS_DEBUG=ON
    if(KWS_DEBUG)
        target_compile_definitions(${ProjectId} PRIVATE KWS_DEBUG KWS_DEBUG_C0)
    endif()

    # Compile the SFU graph (KWSMic.src → KWSMic_L2_Descr.c + KWSMic_L2_Descr.h).
    # SFU tool must be in PATH (source gap9-sdk/sourceme.sh before cmake).
    if(NOT SFU)
        find_program(SFU NAMES SFU
            HINTS $ENV{GAP_SDK_HOME}/install/workstation/bin
            REQUIRED)
    endif()
    sfu_add_graphs(${ProjectId} SOURCE ${MFCC_DIR}/KWSMic.src)

    # sfu_add_graphs declares OUTPUTs at ../../../ (build root) but the SFU tool
    # writes files into CMAKE_CURRENT_BINARY_DIR.  Append copy steps so the files
    # exist at the paths cmake passes to the compiler.
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/../../../KWSMic.c
        APPEND
        COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_BINARY_DIR}/KWSMic.c
            ${CMAKE_CURRENT_BINARY_DIR}/../../../KWSMic.c
    )
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/../../../Graph_L2_Descr.c
        APPEND
        COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_BINARY_DIR}/KWSMic_L2_Descr.c
            ${CMAKE_CURRENT_BINARY_DIR}/../../../Graph_L2_Descr.c
    )

    # KWSMic_L2_Descr.h is generated into CMAKE_CURRENT_BINARY_DIR.
    target_include_directories(${ProjectId} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

    # Link the SFU runtime library (CONFIG_DRIVER_SFU_RUNTIME_V2 in sdk_board.config)
    setup_sfu_libs(${ProjectId})
endif()

target_link_libraries(${ProjectId} PRIVATE network deeploylib)
target_compile_options(${ProjectId} INTERFACE network)
add_gvsoc_emulation(${ProjectId} "gap9.evk")
add_board_deployment(${ProjectId} "gap9.evk")

if(POWER_MEASUREMENT)
  target_compile_definitions(${ProjectId} PRIVATE POWER_MEASUREMENT)
endif()

# RW: Waive sign comparison warnings from pulp_nn_utils.h
target_compile_options(network PRIVATE
    -Wno-sign-compare
    -Wno-pointer-sign
    -Wno-unknown-pragmas
    -Wno-error
  )

target_link_options(${ProjectId} PRIVATE
    -Wl,--print-memory-usage
)
link_compile_dump(${TESTNAME})
