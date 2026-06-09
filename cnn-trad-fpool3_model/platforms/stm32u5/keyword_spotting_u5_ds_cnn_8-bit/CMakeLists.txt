cmake_minimum_required(VERSION 3.22)

#
# This file is generated only once,
# and is not re-generated if converter is called multiple times.
#
# User is free to modify the file as much as necessary
#

# Setup compiler settings
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)


# Define the build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug")
endif()

# Set the project name
set(CMAKE_PROJECT_NAME keyword_spotting_u5_ds_cnn_v1)

# Enable compile command to ease indexing with e.g. clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

# Core project settings
project(${CMAKE_PROJECT_NAME})
message("Build type: " ${CMAKE_BUILD_TYPE})

# Enable CMake support for ASM and C languages
enable_language(C ASM)

# Create an executable object type
add_executable(${CMAKE_PROJECT_NAME})

# Add STM32CubeMX generated sources
add_subdirectory(cmake/stm32cubemx)

# Link directories setup
target_link_directories(${CMAKE_PROJECT_NAME} PRIVATE
    # Add user defined library search paths
)

# Add sources to executable
target_sources(${CMAKE_PROJECT_NAME} PRIVATE
    # KWS application sources (not managed by CubeMX)
    Core/Src/kws20_live.c
    Core/Src/kws20_measure.c
    Core/Src/kws20_test.c
    Core/Src/ds_cnn_frontend.c
    Core/Src/audio_hw_probe.c
    Core/Src/audio_st_minimal.c
    Core/Src/mic2_poll.c
    Core/Src/mic_debug.c
    # X-CUBE-AI generated model sources
    X-CUBE-AI/App/network.c
    X-CUBE-AI/App/network_data.c
    X-CUBE-AI/App/network_data_params.c
    # MDF HAL module + B-U585 BSP audio (hand-added mic subsystem, not in the .ioc)
    Drivers/STM32U5xx_HAL_Driver/Src/stm32u5xx_hal_mdf.c
    Drivers/BSP/B-U585I-IOT02A/b_u585i_iot02a.c
    Drivers/BSP/B-U585I-IOT02A/b_u585i_iot02a_audio.c
)

# Add include paths
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
    # X-CUBE-AI runtime headers (copied from the X-CUBE-AI 10.2 pack)
    Middlewares/ST/AI/Inc
    # B-U585 BSP headers (mic/audio)
    Drivers/BSP/B-U585I-IOT02A
)

# Add project symbols (macros)
target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE
    # Add user defined symbols
)

# Remove wrong libob.a library dependency when using cpp files
list(REMOVE_ITEM CMAKE_C_IMPLICIT_LINK_LIBRARIES ob)

# Add linked libraries
target_link_libraries(${CMAKE_PROJECT_NAME}
    stm32cubemx

    # X-CUBE-AI 10.2 neural-network runtime (Cortex-M33, GCC)
    ${CMAKE_CURRENT_SOURCE_DIR}/Middlewares/ST/AI/Lib/GCC/ARMCortexM33/NetworkRuntime1020_CM33_GCC.a
)
