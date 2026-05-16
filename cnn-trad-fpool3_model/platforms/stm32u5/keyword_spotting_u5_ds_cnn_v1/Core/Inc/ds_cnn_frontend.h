#ifndef DS_CNN_FRONTEND_H
#define DS_CNN_FRONTEND_H

#include <stdint.h>

#include "kws20_model_io.h"

int ds_cnn_frontend_compute(const int16_t *audio,
                            uint32_t sample_count,
                            kws20_input_elem_t *out,
                            uint32_t out_count);

#endif
