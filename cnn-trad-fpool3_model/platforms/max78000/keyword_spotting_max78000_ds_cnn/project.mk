###############################################################################
# DS-CNN-L KWS — MAX78000 FTHR, Phase 1 (CPU-only)
###############################################################################

# Target board
BOARD = FTHR_RevA

# Optimization: -O2 for best CPU performance measurement
MXC_OPTIMIZE_CFLAGS = -O2

# Enable hard FPU
MFLOAT_ABI = hard

# Project-specific defines
PROJ_CFLAGS += -DBOARD_FTHR_RevA

# No CNN accelerator, no audio BSP, no TFT, no SDHC
# (all are disabled by default — nothing to add here)
