###############################################################################
# kws20_32bit — Float32 CPU-only inference (no CNN hardware accelerator)
#
# Build:
#   make -C kws20_32bit BOARD=FTHR_RevA
#
# IMPORTANT: Generate kws20_32bit_weights.h first:
#   cd ~/max78000/ai8x-training
#   python3 <project>/extract_weights.py \
#       --checkpoint ../ai8x-synthesis/trained/ai85-kws20_v3-qat8-q.pth.tar \
#       --out        <project>/kws20_32bit_weights.h
###############################################################################

# Source files (no cnn.c / weights.h / softmax.c)
SRCS += main.c
SRCS += kws20_32bit_inference.c

# Optimization: maximum speed (float32 is compute-heavy on M4F)
MXC_OPTIMIZE_CFLAGS = -O3

# Enable FPU for Cortex-M4F hardware float
PROJ_CFLAGS += -ffast-math -mfpu=fpv4-sp-d16 -mfloat-abi=hard

# Board-specific settings (same as kws20_demo)
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
