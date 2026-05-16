###############################################################################
# DS-CNN-L KWS — MAX78000 FTHR, v1 (CNN accelerator, INT8)
###############################################################################

BOARD = FTHR_RevA

MXC_OPTIMIZE_CFLAGS = -O2

MFLOAT_ABI = hard

PROJ_CFLAGS += -DBOARD_FTHR_RevA

# Link math library for roundf() in cnn_inference.c
PROJ_LIBS += m
