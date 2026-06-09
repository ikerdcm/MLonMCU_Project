# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0


macro(add_board_deployment name target)

    if(NOT DEFINED GVSOC_INSTALL_DIR)
        message(FATAL_ERROR "Environment variable GVSOC_INSTALL_DIR not set")
    endif()

    message(STATUS "[Deeploy GAP9] Running on Board")

    set(BOARD_WORKDIR ${CMAKE_BINARY_DIR}/board_workdir)
    set(DEEPLOY_BINARY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${name}")
    set(GAP9_SDK_HOME $ENV{GAP_SDK_HOME})
    set(GAPY "${GAP9_SDK_HOME}/utils/gapy_v2/bin/gapy")

    make_directory(${BOARD_WORKDIR})

    if(NOT DEFINED GAP9_SDK_HOME)
        message(FATAL_ERROR "Environment variable GAP_SDK_HOME not set")
    endif()

    # Use load_and_start_binary via gapy (same as helloworld run target).
    # Omitting --flash-property for fsbl/ssbl makes gapy generate
    # "load_and_start_binary" instead of a true MRAM flash boot, which
    # is required for semihost to work correctly on the GAP9 EVK.
    set(GAPY_CMD
        ${GAPY}
        --target=gap9.evk
        --platform=board
        --target-property=boot.flash_device=mram
        --target-property=boot.mode=flash
        --target-dir=${GAP9_SDK_HOME}/utils/gapy_v2/targets
        --openocd-cable=${GAP9_SDK_HOME}/utils/openocd_tools/tcl/gapuino_ftdi.cfg
        --openocd-script=${GAP9_SDK_HOME}/utils/openocd_tools/tcl/gap9revb.tcl
        --openocd-tools=${GAP9_SDK_HOME}/utils/openocd_tools
        --binary=${DEEPLOY_BINARY}
        --work-dir=${BOARD_WORKDIR}
        run
        --py-stack
    )

    # Add readfs files if provided
    if(GAPY_RUNNER_ARGS)
        list(LENGTH GAPY_RUNNER_ARGS num_readfs_files)
        message(STATUS "[Deeploy GAP9] Adding ${num_readfs_files} readfs file(s)")
        list(APPEND GAPY_CMD ${GAPY_RUNNER_ARGS})
    endif()

    # Convert list to string for printing
    string(REPLACE ";" " " GAPY_CMD_STR "${GAPY_CMD}")

    add_custom_target(board_${name}
        DEPENDS ${name}
        WORKING_DIRECTORY ${BOARD_WORKDIR}
        COMMAND ${CMAKE_COMMAND} -E echo "=========================================="
        COMMAND ${CMAKE_COMMAND} -E echo "[Deeploy GAP9] Executing gapy command to run on board:"
        COMMAND ${CMAKE_COMMAND} -E echo "${GAPY_CMD_STR}"
        COMMAND ${CMAKE_COMMAND} -E echo "=========================================="
        COMMAND ${GAPY_CMD}
        COMMENT "Running ${name} with gapy on GAP9 board"
        POST_BUILD
        USES_TERMINAL
        VERBATIM
    )
endmacro()