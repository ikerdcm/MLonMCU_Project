###############################################################################
# kws20_int8_cpu — INT8 quantized weights, software inference (no CNN accelerator)
#
# Build:
#   make -C kws20_int8_cpu BOARD=FTHR_RevA
###############################################################################

SRCS += main.c
SRCS += kws20_int8_cpu_inference.c

MXC_OPTIMIZE_CFLAGS = -O3

PROJ_CFLAGS += -mfpu=fpv4-sp-d16 -mfloat-abi=softfp

ifeq "$(BOARD)" "FTHR_RevA"
IPATH += TFT/fthr
VPATH += TFT/fthr
FONTS = LiberationSans16x16
LIB_SDHC = 1
endif

ifeq "$(BOARD)" "EvKit_V1"
PROJ_CFLAGS += -DTFT_ENABLE
IPATH += TFT/evkit/
VPATH += TFT/evkit/
endif

ifeq ($(BOARD),CAM01_RevA)
$(error ERR_NOTSUPPORTED: This project is not supported for the CAM01 board)
endif

ifeq ($(BOARD),CAM02_RevA)
$(error ERR_NOTSUPPORTED: This project is not supported for the CAM02 board)
endif
