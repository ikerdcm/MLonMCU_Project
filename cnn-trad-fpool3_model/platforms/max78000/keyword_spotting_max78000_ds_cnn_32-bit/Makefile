###############################################################################
#
# DS-CNN-L Keyword Spotting — MAX78000 FTHR, Phase 1 (CPU-only)
#
# Copyright (C) 2024 Analog Devices, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
###############################################################################

# ** Readme! **
# Don't edit this file! This is the core Makefile for a MaximSDK
# project. The available configuration options can be overridden
# in "project.mk", on the command-line, or with system environment
# variables.
#
# See https://analogdevicesinc.github.io/msdk/USERGUIDE/#build-system

# ---------------------------------------------------------------------------
# Target micro
# ---------------------------------------------------------------------------
ifeq "$(TARGET)" ""
TARGET    := MAX78000
TARGET_UC := MAX78000
TARGET_LC := max78000
else
TARGET_UC := $(subst m,M,$(subst a,A,$(subst x,X,$(TARGET))))
TARGET_LC := $(subst M,m,$(subst A,a,$(subst X,x,$(TARGET))))
endif

# Default board (can be overridden on command line)
BOARD ?= FTHR_RevA

# ---------------------------------------------------------------------------
# MaximSDK location
# ---------------------------------------------------------------------------
ifneq "$(MAXIM_PATH)" ""
MAXIM_PATH := $(subst \,/,$(MAXIM_PATH))
LIBS_DIR   := $(abspath $(MAXIM_PATH)/Libraries)
CMSIS_ROOT := $(LIBS_DIR)/CMSIS
endif

# ---------------------------------------------------------------------------
# Include project.mk early so it can set BOARD, MFLOAT_ABI, etc.
# ---------------------------------------------------------------------------
PROJECTMK ?= $(abspath ./project.mk)
include $(PROJECTMK)
$(info Loaded project.mk)

# ---------------------------------------------------------------------------
# Final path resolution
# ---------------------------------------------------------------------------
ifeq "$(MAXIM_PATH)" ""
DEPTH      := ../../../../
MAXIM_PATH := $(abspath $(DEPTH))
$(warning Warning: MAXIM_PATH is not set! Attempting to use $(MAXIM_PATH) from relative path.)
else
MAXIM_PATH := $(subst \,/,$(MAXIM_PATH))
endif

LIBS_DIR   := $(abspath $(MAXIM_PATH)/Libraries)
CMSIS_ROOT := $(LIBS_DIR)/CMSIS

TARGET_UC  := $(subst m,M,$(subst a,A,$(subst x,X,$(TARGET))))
TARGET_LC  := $(subst M,m,$(subst A,a,$(subst X,x,$(TARGET))))

export TARGET
export TARGET_UC
export TARGET_LC
export CMSIS_ROOT

# ---------------------------------------------------------------------------
# Source search paths
# ---------------------------------------------------------------------------
VPATH += .
IPATH += .

# Auto-detect all .c files in the project directory
AUTOSEARCH ?= 1
ifeq ($(AUTOSEARCH), 1)
SRCS += $(wildcard $(addsuffix /*.c, $(VPATH)))
endif

SRCS := $(SRCS)

# ---------------------------------------------------------------------------
# Output name
# ---------------------------------------------------------------------------
PROJECT ?= $(TARGET_LC)

# ---------------------------------------------------------------------------
# Compiler flags
# ---------------------------------------------------------------------------
PROJ_CFLAGS += -Wall
PROJ_CFLAGS += -DMXC_ASSERT_ENABLE

# Hard FPU (Cortex-M4F)
MFLOAT_ABI ?= hard
export MFLOAT_ABI

# Add CMSIS startup to search path
VPATH += $(CMSIS_ROOT)/Device/Maxim/$(TARGET_UC)/Source

# ---------------------------------------------------------------------------
# No extra libraries needed (no CNN, no TFT, no audio BSP, no SDHC)
# ---------------------------------------------------------------------------
# LIB_BOARD, LIB_PERIPHDRIVERS are enabled by default in libs.mk

# ---------------------------------------------------------------------------
# Include SDK build system
# ---------------------------------------------------------------------------
include $(LIBS_DIR)/libs.mk
include $(CMSIS_ROOT)/Device/Maxim/$(TARGET_UC)/Source/GCC/$(TARGET_LC).mk

# Keep .DEFAULT_GOAL working
ifeq "$(.DEFAULT_GOAL)" ""
override .DEFAULT_GOAL := all
endif
ifeq "$(MAKECMDGOALS)" ""
MAKECMDGOALS := $(.DEFAULT_GOAL)
endif

all:
	$(PREFIX)-size --format=berkeley $(BUILD_DIR)/$(PROJECT).elf

libclean:
	$(MAKE) -f ${PERIPH_DRIVER_DIR}/periphdriver.mk clean.periph

clean:
	# extend if needed

distclean: clean libclean
