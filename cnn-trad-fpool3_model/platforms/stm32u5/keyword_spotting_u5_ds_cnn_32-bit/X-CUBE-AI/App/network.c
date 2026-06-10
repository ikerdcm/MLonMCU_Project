/**
  ******************************************************************************
  * @file    network.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-05-15T12:12:12+0200
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */


#include "network.h"
#include "network_data.h"

#include "ai_platform.h"
#include "ai_platform_interface.h"
#include "ai_math_helpers.h"

#include "core_common.h"
#include "core_convert.h"

#include "layers.h"



#undef AI_NET_OBJ_INSTANCE
#define AI_NET_OBJ_INSTANCE g_network
 
#undef AI_NETWORK_MODEL_SIGNATURE
#define AI_NETWORK_MODEL_SIGNATURE     "0x53300598dc7ef266a61f5b141f1d3873"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     ""
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "2026-05-15T12:12:12+0200"

#undef AI_TOOLS_COMPILE_TIME
#define AI_TOOLS_COMPILE_TIME    __DATE__ " " __TIME__

#undef AI_NETWORK_N_BATCHES
#define AI_NETWORK_N_BATCHES         (1)

static ai_ptr g_network_activations_map[1] = AI_C_ARRAY_INIT;
static ai_ptr g_network_weights_map[1] = AI_C_ARRAY_INIT;



/**  Array declarations section  **********************************************/
/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  input_layer0_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 490, AI_STATIC)

/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  functional_1_conv2d_1_BiasAdd__60_to_chfirst_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 490, AI_STATIC)

/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  functional_1_conv2d_1_BiasAdd0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8000, AI_STATIC)

/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  functional_1_activation_1_Relu0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8000, AI_STATIC)

/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  Conv__1000_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8000, AI_STATIC)

/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  functional_1_activation_1_2_Relu0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8000, AI_STATIC)

/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  functional_1_conv2d_1_2_BiasAdd0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8000, AI_STATIC)

/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  functional_1_activation_2_1_Relu0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8000, AI_STATIC)

/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  Conv__1180_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8000, AI_STATIC)

/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  functional_1_activation_3_1_Relu0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8000, AI_STATIC)

/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  functional_1_conv2d_2_1_BiasAdd0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8000, AI_STATIC)

/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  functional_1_activation_4_1_Relu0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8000, AI_STATIC)

/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  Conv__1360_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8000, AI_STATIC)

/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  functional_1_activation_5_1_Relu0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8000, AI_STATIC)

/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  functional_1_conv2d_3_1_BiasAdd0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8000, AI_STATIC)

/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  functional_1_activation_6_1_Relu0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8000, AI_STATIC)

/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  Conv__1540_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8000, AI_STATIC)

/* Array#17 */
AI_ARRAY_OBJ_DECLARE(
  functional_1_activation_7_1_Relu0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8000, AI_STATIC)

/* Array#18 */
AI_ARRAY_OBJ_DECLARE(
  functional_1_conv2d_4_1_BiasAdd0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8000, AI_STATIC)

/* Array#19 */
AI_ARRAY_OBJ_DECLARE(
  functional_1_activation_8_1_Relu0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8000, AI_STATIC)

/* Array#20 */
AI_ARRAY_OBJ_DECLARE(
  Conv__1720_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8000, AI_STATIC)

/* Array#21 */
AI_ARRAY_OBJ_DECLARE(
  functional_1_activation_9_1_Relu0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8000, AI_STATIC)

/* Array#22 */
AI_ARRAY_OBJ_DECLARE(
  functional_1_conv2d_5_1_BiasAdd0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8000, AI_STATIC)

/* Array#23 */
AI_ARRAY_OBJ_DECLARE(
  functional_1_activation_10_1_Relu0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8000, AI_STATIC)

/* Array#24 */
AI_ARRAY_OBJ_DECLARE(
  Conv__1900_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8000, AI_STATIC)

/* Array#25 */
AI_ARRAY_OBJ_DECLARE(
  functional_1_activation_11_1_Relu0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8000, AI_STATIC)

/* Array#26 */
AI_ARRAY_OBJ_DECLARE(
  functional_1_conv2d_6_1_BiasAdd0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8000, AI_STATIC)

/* Array#27 */
AI_ARRAY_OBJ_DECLARE(
  functional_1_activation_12_1_Relu0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8000, AI_STATIC)

/* Array#28 */
AI_ARRAY_OBJ_DECLARE(
  functional_1_average_pooling2d_1_AvgPool0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#29 */
AI_ARRAY_OBJ_DECLARE(
  functional_1_dense_1_MatMul0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12, AI_STATIC)

/* Array#30 */
AI_ARRAY_OBJ_DECLARE(
  Identity0_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 12, AI_STATIC)

/* Array#31 */
AI_ARRAY_OBJ_DECLARE(
  functional_1_conv2d_1_BiasAdd0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2560, AI_STATIC)

/* Array#32 */
AI_ARRAY_OBJ_DECLARE(
  functional_1_conv2d_1_BiasAdd0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#33 */
AI_ARRAY_OBJ_DECLARE(
  Conv__1000_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 576, AI_STATIC)

/* Array#34 */
AI_ARRAY_OBJ_DECLARE(
  Conv__1000_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#35 */
AI_ARRAY_OBJ_DECLARE(
  functional_1_conv2d_1_2_BiasAdd0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4096, AI_STATIC)

/* Array#36 */
AI_ARRAY_OBJ_DECLARE(
  functional_1_conv2d_1_2_BiasAdd0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#37 */
AI_ARRAY_OBJ_DECLARE(
  Conv__1180_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 576, AI_STATIC)

/* Array#38 */
AI_ARRAY_OBJ_DECLARE(
  Conv__1180_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#39 */
AI_ARRAY_OBJ_DECLARE(
  functional_1_conv2d_2_1_BiasAdd0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4096, AI_STATIC)

/* Array#40 */
AI_ARRAY_OBJ_DECLARE(
  functional_1_conv2d_2_1_BiasAdd0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#41 */
AI_ARRAY_OBJ_DECLARE(
  Conv__1360_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 576, AI_STATIC)

/* Array#42 */
AI_ARRAY_OBJ_DECLARE(
  Conv__1360_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#43 */
AI_ARRAY_OBJ_DECLARE(
  functional_1_conv2d_3_1_BiasAdd0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4096, AI_STATIC)

/* Array#44 */
AI_ARRAY_OBJ_DECLARE(
  functional_1_conv2d_3_1_BiasAdd0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#45 */
AI_ARRAY_OBJ_DECLARE(
  Conv__1540_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 576, AI_STATIC)

/* Array#46 */
AI_ARRAY_OBJ_DECLARE(
  Conv__1540_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#47 */
AI_ARRAY_OBJ_DECLARE(
  functional_1_conv2d_4_1_BiasAdd0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4096, AI_STATIC)

/* Array#48 */
AI_ARRAY_OBJ_DECLARE(
  functional_1_conv2d_4_1_BiasAdd0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#49 */
AI_ARRAY_OBJ_DECLARE(
  Conv__1720_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 576, AI_STATIC)

/* Array#50 */
AI_ARRAY_OBJ_DECLARE(
  Conv__1720_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#51 */
AI_ARRAY_OBJ_DECLARE(
  functional_1_conv2d_5_1_BiasAdd0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4096, AI_STATIC)

/* Array#52 */
AI_ARRAY_OBJ_DECLARE(
  functional_1_conv2d_5_1_BiasAdd0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#53 */
AI_ARRAY_OBJ_DECLARE(
  Conv__1900_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 576, AI_STATIC)

/* Array#54 */
AI_ARRAY_OBJ_DECLARE(
  Conv__1900_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#55 */
AI_ARRAY_OBJ_DECLARE(
  functional_1_conv2d_6_1_BiasAdd0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4096, AI_STATIC)

/* Array#56 */
AI_ARRAY_OBJ_DECLARE(
  functional_1_conv2d_6_1_BiasAdd0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#57 */
AI_ARRAY_OBJ_DECLARE(
  functional_1_dense_1_MatMul0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 768, AI_STATIC)

/* Array#58 */
AI_ARRAY_OBJ_DECLARE(
  functional_1_dense_1_MatMul0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12, AI_STATIC)

/* Array#59 */
AI_ARRAY_OBJ_DECLARE(
  functional_1_conv2d_1_BiasAdd0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 40, AI_STATIC)

/* Array#60 */
AI_ARRAY_OBJ_DECLARE(
  functional_1_conv2d_1_2_BiasAdd0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#61 */
AI_ARRAY_OBJ_DECLARE(
  functional_1_conv2d_2_1_BiasAdd0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#62 */
AI_ARRAY_OBJ_DECLARE(
  functional_1_conv2d_3_1_BiasAdd0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#63 */
AI_ARRAY_OBJ_DECLARE(
  functional_1_conv2d_4_1_BiasAdd0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#64 */
AI_ARRAY_OBJ_DECLARE(
  functional_1_conv2d_5_1_BiasAdd0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#65 */
AI_ARRAY_OBJ_DECLARE(
  functional_1_conv2d_6_1_BiasAdd0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/**  Tensor declarations section  *********************************************/
/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  Conv__1000_bias, AI_STATIC,
  0, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &Conv__1000_bias_array, NULL)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  Conv__1000_output, AI_STATIC,
  1, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 5, 25), AI_STRIDE_INIT(4, 4, 4, 256, 1280),
  1, &Conv__1000_output_array, NULL)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  Conv__1000_weights, AI_STATIC,
  2, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 3, 64), AI_STRIDE_INIT(4, 1, 64, 64, 64),
  1, &Conv__1000_weights_array, NULL)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  Conv__1180_bias, AI_STATIC,
  3, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &Conv__1180_bias_array, NULL)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  Conv__1180_output, AI_STATIC,
  4, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 5, 25), AI_STRIDE_INIT(4, 4, 4, 256, 1280),
  1, &Conv__1180_output_array, NULL)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  Conv__1180_weights, AI_STATIC,
  5, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 3, 64), AI_STRIDE_INIT(4, 1, 64, 64, 64),
  1, &Conv__1180_weights_array, NULL)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  Conv__1360_bias, AI_STATIC,
  6, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &Conv__1360_bias_array, NULL)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  Conv__1360_output, AI_STATIC,
  7, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 5, 25), AI_STRIDE_INIT(4, 4, 4, 256, 1280),
  1, &Conv__1360_output_array, NULL)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  Conv__1360_weights, AI_STATIC,
  8, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 3, 64), AI_STRIDE_INIT(4, 1, 64, 64, 64),
  1, &Conv__1360_weights_array, NULL)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  Conv__1540_bias, AI_STATIC,
  9, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &Conv__1540_bias_array, NULL)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  Conv__1540_output, AI_STATIC,
  10, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 5, 25), AI_STRIDE_INIT(4, 4, 4, 256, 1280),
  1, &Conv__1540_output_array, NULL)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  Conv__1540_weights, AI_STATIC,
  11, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 3, 64), AI_STRIDE_INIT(4, 1, 64, 64, 64),
  1, &Conv__1540_weights_array, NULL)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  Conv__1720_bias, AI_STATIC,
  12, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &Conv__1720_bias_array, NULL)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  Conv__1720_output, AI_STATIC,
  13, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 5, 25), AI_STRIDE_INIT(4, 4, 4, 256, 1280),
  1, &Conv__1720_output_array, NULL)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  Conv__1720_weights, AI_STATIC,
  14, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 3, 64), AI_STRIDE_INIT(4, 1, 64, 64, 64),
  1, &Conv__1720_weights_array, NULL)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  Conv__1900_bias, AI_STATIC,
  15, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &Conv__1900_bias_array, NULL)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  Conv__1900_output, AI_STATIC,
  16, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 5, 25), AI_STRIDE_INIT(4, 4, 4, 256, 1280),
  1, &Conv__1900_output_array, NULL)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  Conv__1900_weights, AI_STATIC,
  17, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 3, 64), AI_STRIDE_INIT(4, 1, 64, 64, 64),
  1, &Conv__1900_weights_array, NULL)

/* Tensor #18 */
AI_TENSOR_OBJ_DECLARE(
  Identity0_output, AI_STATIC,
  18, 0x0,
  AI_SHAPE_INIT(4, 1, 12, 1, 1), AI_STRIDE_INIT(4, 4, 4, 48, 48),
  1, &Identity0_output_array, NULL)

/* Tensor #19 */
AI_TENSOR_OBJ_DECLARE(
  functional_1_activation_10_1_Relu0_output, AI_STATIC,
  19, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 5, 25), AI_STRIDE_INIT(4, 4, 4, 256, 1280),
  1, &functional_1_activation_10_1_Relu0_output_array, NULL)

/* Tensor #20 */
AI_TENSOR_OBJ_DECLARE(
  functional_1_activation_11_1_Relu0_output, AI_STATIC,
  20, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 5, 25), AI_STRIDE_INIT(4, 4, 4, 256, 1280),
  1, &functional_1_activation_11_1_Relu0_output_array, NULL)

/* Tensor #21 */
AI_TENSOR_OBJ_DECLARE(
  functional_1_activation_12_1_Relu0_output, AI_STATIC,
  21, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 5, 25), AI_STRIDE_INIT(4, 4, 4, 256, 1280),
  1, &functional_1_activation_12_1_Relu0_output_array, NULL)

/* Tensor #22 */
AI_TENSOR_OBJ_DECLARE(
  functional_1_activation_1_2_Relu0_output, AI_STATIC,
  22, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 5, 25), AI_STRIDE_INIT(4, 4, 4, 256, 1280),
  1, &functional_1_activation_1_2_Relu0_output_array, NULL)

/* Tensor #23 */
AI_TENSOR_OBJ_DECLARE(
  functional_1_activation_1_Relu0_output, AI_STATIC,
  23, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 5, 25), AI_STRIDE_INIT(4, 4, 4, 256, 1280),
  1, &functional_1_activation_1_Relu0_output_array, NULL)

/* Tensor #24 */
AI_TENSOR_OBJ_DECLARE(
  functional_1_activation_2_1_Relu0_output, AI_STATIC,
  24, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 5, 25), AI_STRIDE_INIT(4, 4, 4, 256, 1280),
  1, &functional_1_activation_2_1_Relu0_output_array, NULL)

/* Tensor #25 */
AI_TENSOR_OBJ_DECLARE(
  functional_1_activation_3_1_Relu0_output, AI_STATIC,
  25, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 5, 25), AI_STRIDE_INIT(4, 4, 4, 256, 1280),
  1, &functional_1_activation_3_1_Relu0_output_array, NULL)

/* Tensor #26 */
AI_TENSOR_OBJ_DECLARE(
  functional_1_activation_4_1_Relu0_output, AI_STATIC,
  26, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 5, 25), AI_STRIDE_INIT(4, 4, 4, 256, 1280),
  1, &functional_1_activation_4_1_Relu0_output_array, NULL)

/* Tensor #27 */
AI_TENSOR_OBJ_DECLARE(
  functional_1_activation_5_1_Relu0_output, AI_STATIC,
  27, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 5, 25), AI_STRIDE_INIT(4, 4, 4, 256, 1280),
  1, &functional_1_activation_5_1_Relu0_output_array, NULL)

/* Tensor #28 */
AI_TENSOR_OBJ_DECLARE(
  functional_1_activation_6_1_Relu0_output, AI_STATIC,
  28, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 5, 25), AI_STRIDE_INIT(4, 4, 4, 256, 1280),
  1, &functional_1_activation_6_1_Relu0_output_array, NULL)

/* Tensor #29 */
AI_TENSOR_OBJ_DECLARE(
  functional_1_activation_7_1_Relu0_output, AI_STATIC,
  29, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 5, 25), AI_STRIDE_INIT(4, 4, 4, 256, 1280),
  1, &functional_1_activation_7_1_Relu0_output_array, NULL)

/* Tensor #30 */
AI_TENSOR_OBJ_DECLARE(
  functional_1_activation_8_1_Relu0_output, AI_STATIC,
  30, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 5, 25), AI_STRIDE_INIT(4, 4, 4, 256, 1280),
  1, &functional_1_activation_8_1_Relu0_output_array, NULL)

/* Tensor #31 */
AI_TENSOR_OBJ_DECLARE(
  functional_1_activation_9_1_Relu0_output, AI_STATIC,
  31, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 5, 25), AI_STRIDE_INIT(4, 4, 4, 256, 1280),
  1, &functional_1_activation_9_1_Relu0_output_array, NULL)

/* Tensor #32 */
AI_TENSOR_OBJ_DECLARE(
  functional_1_average_pooling2d_1_AvgPool0_output, AI_STATIC,
  32, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &functional_1_average_pooling2d_1_AvgPool0_output_array, NULL)

/* Tensor #33 */
AI_TENSOR_OBJ_DECLARE(
  functional_1_conv2d_1_2_BiasAdd0_bias, AI_STATIC,
  33, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &functional_1_conv2d_1_2_BiasAdd0_bias_array, NULL)

/* Tensor #34 */
AI_TENSOR_OBJ_DECLARE(
  functional_1_conv2d_1_2_BiasAdd0_output, AI_STATIC,
  34, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 5, 25), AI_STRIDE_INIT(4, 4, 4, 256, 1280),
  1, &functional_1_conv2d_1_2_BiasAdd0_output_array, NULL)

/* Tensor #35 */
AI_TENSOR_OBJ_DECLARE(
  functional_1_conv2d_1_2_BiasAdd0_scratch0, AI_STATIC,
  35, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &functional_1_conv2d_1_2_BiasAdd0_scratch0_array, NULL)

/* Tensor #36 */
AI_TENSOR_OBJ_DECLARE(
  functional_1_conv2d_1_2_BiasAdd0_weights, AI_STATIC,
  36, 0x0,
  AI_SHAPE_INIT(4, 64, 1, 1, 64), AI_STRIDE_INIT(4, 4, 256, 16384, 16384),
  1, &functional_1_conv2d_1_2_BiasAdd0_weights_array, NULL)

/* Tensor #37 */
AI_TENSOR_OBJ_DECLARE(
  functional_1_conv2d_1_BiasAdd0_bias, AI_STATIC,
  37, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &functional_1_conv2d_1_BiasAdd0_bias_array, NULL)

/* Tensor #38 */
AI_TENSOR_OBJ_DECLARE(
  functional_1_conv2d_1_BiasAdd0_output, AI_STATIC,
  38, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 5, 25), AI_STRIDE_INIT(4, 4, 4, 256, 1280),
  1, &functional_1_conv2d_1_BiasAdd0_output_array, NULL)

/* Tensor #39 */
AI_TENSOR_OBJ_DECLARE(
  functional_1_conv2d_1_BiasAdd0_scratch0, AI_STATIC,
  39, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 4, 10), AI_STRIDE_INIT(4, 4, 4, 4, 16),
  1, &functional_1_conv2d_1_BiasAdd0_scratch0_array, NULL)

/* Tensor #40 */
AI_TENSOR_OBJ_DECLARE(
  functional_1_conv2d_1_BiasAdd0_weights, AI_STATIC,
  40, 0x0,
  AI_SHAPE_INIT(4, 1, 4, 10, 64), AI_STRIDE_INIT(4, 4, 4, 256, 1024),
  1, &functional_1_conv2d_1_BiasAdd0_weights_array, NULL)

/* Tensor #41 */
AI_TENSOR_OBJ_DECLARE(
  functional_1_conv2d_1_BiasAdd__60_to_chfirst_output, AI_STATIC,
  41, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 10, 49), AI_STRIDE_INIT(4, 4, 4, 4, 40),
  1, &functional_1_conv2d_1_BiasAdd__60_to_chfirst_output_array, NULL)

/* Tensor #42 */
AI_TENSOR_OBJ_DECLARE(
  functional_1_conv2d_2_1_BiasAdd0_bias, AI_STATIC,
  42, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &functional_1_conv2d_2_1_BiasAdd0_bias_array, NULL)

/* Tensor #43 */
AI_TENSOR_OBJ_DECLARE(
  functional_1_conv2d_2_1_BiasAdd0_output, AI_STATIC,
  43, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 5, 25), AI_STRIDE_INIT(4, 4, 4, 256, 1280),
  1, &functional_1_conv2d_2_1_BiasAdd0_output_array, NULL)

/* Tensor #44 */
AI_TENSOR_OBJ_DECLARE(
  functional_1_conv2d_2_1_BiasAdd0_scratch0, AI_STATIC,
  44, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &functional_1_conv2d_2_1_BiasAdd0_scratch0_array, NULL)

/* Tensor #45 */
AI_TENSOR_OBJ_DECLARE(
  functional_1_conv2d_2_1_BiasAdd0_weights, AI_STATIC,
  45, 0x0,
  AI_SHAPE_INIT(4, 64, 1, 1, 64), AI_STRIDE_INIT(4, 4, 256, 16384, 16384),
  1, &functional_1_conv2d_2_1_BiasAdd0_weights_array, NULL)

/* Tensor #46 */
AI_TENSOR_OBJ_DECLARE(
  functional_1_conv2d_3_1_BiasAdd0_bias, AI_STATIC,
  46, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &functional_1_conv2d_3_1_BiasAdd0_bias_array, NULL)

/* Tensor #47 */
AI_TENSOR_OBJ_DECLARE(
  functional_1_conv2d_3_1_BiasAdd0_output, AI_STATIC,
  47, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 5, 25), AI_STRIDE_INIT(4, 4, 4, 256, 1280),
  1, &functional_1_conv2d_3_1_BiasAdd0_output_array, NULL)

/* Tensor #48 */
AI_TENSOR_OBJ_DECLARE(
  functional_1_conv2d_3_1_BiasAdd0_scratch0, AI_STATIC,
  48, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &functional_1_conv2d_3_1_BiasAdd0_scratch0_array, NULL)

/* Tensor #49 */
AI_TENSOR_OBJ_DECLARE(
  functional_1_conv2d_3_1_BiasAdd0_weights, AI_STATIC,
  49, 0x0,
  AI_SHAPE_INIT(4, 64, 1, 1, 64), AI_STRIDE_INIT(4, 4, 256, 16384, 16384),
  1, &functional_1_conv2d_3_1_BiasAdd0_weights_array, NULL)

/* Tensor #50 */
AI_TENSOR_OBJ_DECLARE(
  functional_1_conv2d_4_1_BiasAdd0_bias, AI_STATIC,
  50, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &functional_1_conv2d_4_1_BiasAdd0_bias_array, NULL)

/* Tensor #51 */
AI_TENSOR_OBJ_DECLARE(
  functional_1_conv2d_4_1_BiasAdd0_output, AI_STATIC,
  51, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 5, 25), AI_STRIDE_INIT(4, 4, 4, 256, 1280),
  1, &functional_1_conv2d_4_1_BiasAdd0_output_array, NULL)

/* Tensor #52 */
AI_TENSOR_OBJ_DECLARE(
  functional_1_conv2d_4_1_BiasAdd0_scratch0, AI_STATIC,
  52, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &functional_1_conv2d_4_1_BiasAdd0_scratch0_array, NULL)

/* Tensor #53 */
AI_TENSOR_OBJ_DECLARE(
  functional_1_conv2d_4_1_BiasAdd0_weights, AI_STATIC,
  53, 0x0,
  AI_SHAPE_INIT(4, 64, 1, 1, 64), AI_STRIDE_INIT(4, 4, 256, 16384, 16384),
  1, &functional_1_conv2d_4_1_BiasAdd0_weights_array, NULL)

/* Tensor #54 */
AI_TENSOR_OBJ_DECLARE(
  functional_1_conv2d_5_1_BiasAdd0_bias, AI_STATIC,
  54, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &functional_1_conv2d_5_1_BiasAdd0_bias_array, NULL)

/* Tensor #55 */
AI_TENSOR_OBJ_DECLARE(
  functional_1_conv2d_5_1_BiasAdd0_output, AI_STATIC,
  55, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 5, 25), AI_STRIDE_INIT(4, 4, 4, 256, 1280),
  1, &functional_1_conv2d_5_1_BiasAdd0_output_array, NULL)

/* Tensor #56 */
AI_TENSOR_OBJ_DECLARE(
  functional_1_conv2d_5_1_BiasAdd0_scratch0, AI_STATIC,
  56, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &functional_1_conv2d_5_1_BiasAdd0_scratch0_array, NULL)

/* Tensor #57 */
AI_TENSOR_OBJ_DECLARE(
  functional_1_conv2d_5_1_BiasAdd0_weights, AI_STATIC,
  57, 0x0,
  AI_SHAPE_INIT(4, 64, 1, 1, 64), AI_STRIDE_INIT(4, 4, 256, 16384, 16384),
  1, &functional_1_conv2d_5_1_BiasAdd0_weights_array, NULL)

/* Tensor #58 */
AI_TENSOR_OBJ_DECLARE(
  functional_1_conv2d_6_1_BiasAdd0_bias, AI_STATIC,
  58, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &functional_1_conv2d_6_1_BiasAdd0_bias_array, NULL)

/* Tensor #59 */
AI_TENSOR_OBJ_DECLARE(
  functional_1_conv2d_6_1_BiasAdd0_output, AI_STATIC,
  59, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 5, 25), AI_STRIDE_INIT(4, 4, 4, 256, 1280),
  1, &functional_1_conv2d_6_1_BiasAdd0_output_array, NULL)

/* Tensor #60 */
AI_TENSOR_OBJ_DECLARE(
  functional_1_conv2d_6_1_BiasAdd0_scratch0, AI_STATIC,
  60, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &functional_1_conv2d_6_1_BiasAdd0_scratch0_array, NULL)

/* Tensor #61 */
AI_TENSOR_OBJ_DECLARE(
  functional_1_conv2d_6_1_BiasAdd0_weights, AI_STATIC,
  61, 0x0,
  AI_SHAPE_INIT(4, 64, 1, 1, 64), AI_STRIDE_INIT(4, 4, 256, 16384, 16384),
  1, &functional_1_conv2d_6_1_BiasAdd0_weights_array, NULL)

/* Tensor #62 */
AI_TENSOR_OBJ_DECLARE(
  functional_1_dense_1_MatMul0_bias, AI_STATIC,
  62, 0x0,
  AI_SHAPE_INIT(4, 1, 12, 1, 1), AI_STRIDE_INIT(4, 4, 4, 48, 48),
  1, &functional_1_dense_1_MatMul0_bias_array, NULL)

/* Tensor #63 */
AI_TENSOR_OBJ_DECLARE(
  functional_1_dense_1_MatMul0_output, AI_STATIC,
  63, 0x0,
  AI_SHAPE_INIT(4, 1, 12, 1, 1), AI_STRIDE_INIT(4, 4, 4, 48, 48),
  1, &functional_1_dense_1_MatMul0_output_array, NULL)

/* Tensor #64 */
AI_TENSOR_OBJ_DECLARE(
  functional_1_dense_1_MatMul0_weights, AI_STATIC,
  64, 0x0,
  AI_SHAPE_INIT(4, 64, 12, 1, 1), AI_STRIDE_INIT(4, 4, 256, 3072, 3072),
  1, &functional_1_dense_1_MatMul0_weights_array, NULL)

/* Tensor #65 */
AI_TENSOR_OBJ_DECLARE(
  input_layer0_output, AI_STATIC,
  65, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 10, 49), AI_STRIDE_INIT(4, 4, 4, 4, 40),
  1, &input_layer0_output_array, NULL)

/* Tensor #66 */
AI_TENSOR_OBJ_DECLARE(
  input_layer0_output0, AI_STATIC,
  66, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 49, 1), AI_STRIDE_INIT(4, 4, 4, 40, 1960),
  1, &input_layer0_output_array, NULL)



/**  Layer declarations section  **********************************************/


AI_TENSOR_CHAIN_OBJ_DECLARE(
  Identity0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &functional_1_dense_1_MatMul0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Identity0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  Identity0_layer, 58,
  SM_TYPE, 0x0, NULL,
  sm, forward_sm,
  &Identity0_chain,
  NULL, &Identity0_layer, AI_STATIC, 
  .nl_params = NULL, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  functional_1_dense_1_MatMul0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &functional_1_average_pooling2d_1_AvgPool0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &functional_1_dense_1_MatMul0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &functional_1_dense_1_MatMul0_weights, &functional_1_dense_1_MatMul0_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  functional_1_dense_1_MatMul0_layer, 57,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &functional_1_dense_1_MatMul0_chain,
  NULL, &Identity0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  functional_1_average_pooling2d_1_AvgPool0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &functional_1_activation_12_1_Relu0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &functional_1_average_pooling2d_1_AvgPool0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  functional_1_average_pooling2d_1_AvgPool0_layer, 54,
  POOL_TYPE, 0x0, NULL,
  pool, forward_ap,
  &functional_1_average_pooling2d_1_AvgPool0_chain,
  NULL, &functional_1_dense_1_MatMul0_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(5, 24), 
  .pool_stride = AI_SHAPE_2D_INIT(5, 24), 
  .count_include_pad = 0, 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  functional_1_activation_12_1_Relu0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &functional_1_conv2d_6_1_BiasAdd0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &functional_1_activation_12_1_Relu0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  functional_1_activation_12_1_Relu0_layer, 53,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &functional_1_activation_12_1_Relu0_chain,
  NULL, &functional_1_average_pooling2d_1_AvgPool0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  functional_1_conv2d_6_1_BiasAdd0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &functional_1_activation_11_1_Relu0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &functional_1_conv2d_6_1_BiasAdd0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &functional_1_conv2d_6_1_BiasAdd0_weights, &functional_1_conv2d_6_1_BiasAdd0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &functional_1_conv2d_6_1_BiasAdd0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  functional_1_conv2d_6_1_BiasAdd0_layer, 52,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &functional_1_conv2d_6_1_BiasAdd0_chain,
  NULL, &functional_1_activation_12_1_Relu0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  functional_1_activation_11_1_Relu0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Conv__1900_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &functional_1_activation_11_1_Relu0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  functional_1_activation_11_1_Relu0_layer, 49,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &functional_1_activation_11_1_Relu0_chain,
  NULL, &functional_1_conv2d_6_1_BiasAdd0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  Conv__1900_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &functional_1_activation_10_1_Relu0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Conv__1900_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &Conv__1900_weights, &Conv__1900_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  Conv__1900_layer, 48,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_if32of32wf32,
  &Conv__1900_chain,
  NULL, &functional_1_activation_11_1_Relu0_layer, AI_STATIC, 
  .groups = 64, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  functional_1_activation_10_1_Relu0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &functional_1_conv2d_5_1_BiasAdd0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &functional_1_activation_10_1_Relu0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  functional_1_activation_10_1_Relu0_layer, 45,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &functional_1_activation_10_1_Relu0_chain,
  NULL, &Conv__1900_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  functional_1_conv2d_5_1_BiasAdd0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &functional_1_activation_9_1_Relu0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &functional_1_conv2d_5_1_BiasAdd0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &functional_1_conv2d_5_1_BiasAdd0_weights, &functional_1_conv2d_5_1_BiasAdd0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &functional_1_conv2d_5_1_BiasAdd0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  functional_1_conv2d_5_1_BiasAdd0_layer, 44,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &functional_1_conv2d_5_1_BiasAdd0_chain,
  NULL, &functional_1_activation_10_1_Relu0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  functional_1_activation_9_1_Relu0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Conv__1720_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &functional_1_activation_9_1_Relu0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  functional_1_activation_9_1_Relu0_layer, 41,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &functional_1_activation_9_1_Relu0_chain,
  NULL, &functional_1_conv2d_5_1_BiasAdd0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  Conv__1720_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &functional_1_activation_8_1_Relu0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Conv__1720_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &Conv__1720_weights, &Conv__1720_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  Conv__1720_layer, 40,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_if32of32wf32,
  &Conv__1720_chain,
  NULL, &functional_1_activation_9_1_Relu0_layer, AI_STATIC, 
  .groups = 64, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  functional_1_activation_8_1_Relu0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &functional_1_conv2d_4_1_BiasAdd0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &functional_1_activation_8_1_Relu0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  functional_1_activation_8_1_Relu0_layer, 37,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &functional_1_activation_8_1_Relu0_chain,
  NULL, &Conv__1720_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  functional_1_conv2d_4_1_BiasAdd0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &functional_1_activation_7_1_Relu0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &functional_1_conv2d_4_1_BiasAdd0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &functional_1_conv2d_4_1_BiasAdd0_weights, &functional_1_conv2d_4_1_BiasAdd0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &functional_1_conv2d_4_1_BiasAdd0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  functional_1_conv2d_4_1_BiasAdd0_layer, 36,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &functional_1_conv2d_4_1_BiasAdd0_chain,
  NULL, &functional_1_activation_8_1_Relu0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  functional_1_activation_7_1_Relu0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Conv__1540_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &functional_1_activation_7_1_Relu0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  functional_1_activation_7_1_Relu0_layer, 33,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &functional_1_activation_7_1_Relu0_chain,
  NULL, &functional_1_conv2d_4_1_BiasAdd0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  Conv__1540_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &functional_1_activation_6_1_Relu0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Conv__1540_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &Conv__1540_weights, &Conv__1540_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  Conv__1540_layer, 32,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_if32of32wf32,
  &Conv__1540_chain,
  NULL, &functional_1_activation_7_1_Relu0_layer, AI_STATIC, 
  .groups = 64, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  functional_1_activation_6_1_Relu0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &functional_1_conv2d_3_1_BiasAdd0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &functional_1_activation_6_1_Relu0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  functional_1_activation_6_1_Relu0_layer, 29,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &functional_1_activation_6_1_Relu0_chain,
  NULL, &Conv__1540_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  functional_1_conv2d_3_1_BiasAdd0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &functional_1_activation_5_1_Relu0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &functional_1_conv2d_3_1_BiasAdd0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &functional_1_conv2d_3_1_BiasAdd0_weights, &functional_1_conv2d_3_1_BiasAdd0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &functional_1_conv2d_3_1_BiasAdd0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  functional_1_conv2d_3_1_BiasAdd0_layer, 28,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &functional_1_conv2d_3_1_BiasAdd0_chain,
  NULL, &functional_1_activation_6_1_Relu0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  functional_1_activation_5_1_Relu0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Conv__1360_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &functional_1_activation_5_1_Relu0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  functional_1_activation_5_1_Relu0_layer, 25,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &functional_1_activation_5_1_Relu0_chain,
  NULL, &functional_1_conv2d_3_1_BiasAdd0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  Conv__1360_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &functional_1_activation_4_1_Relu0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Conv__1360_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &Conv__1360_weights, &Conv__1360_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  Conv__1360_layer, 24,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_if32of32wf32,
  &Conv__1360_chain,
  NULL, &functional_1_activation_5_1_Relu0_layer, AI_STATIC, 
  .groups = 64, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  functional_1_activation_4_1_Relu0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &functional_1_conv2d_2_1_BiasAdd0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &functional_1_activation_4_1_Relu0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  functional_1_activation_4_1_Relu0_layer, 21,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &functional_1_activation_4_1_Relu0_chain,
  NULL, &Conv__1360_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  functional_1_conv2d_2_1_BiasAdd0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &functional_1_activation_3_1_Relu0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &functional_1_conv2d_2_1_BiasAdd0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &functional_1_conv2d_2_1_BiasAdd0_weights, &functional_1_conv2d_2_1_BiasAdd0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &functional_1_conv2d_2_1_BiasAdd0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  functional_1_conv2d_2_1_BiasAdd0_layer, 20,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &functional_1_conv2d_2_1_BiasAdd0_chain,
  NULL, &functional_1_activation_4_1_Relu0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  functional_1_activation_3_1_Relu0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Conv__1180_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &functional_1_activation_3_1_Relu0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  functional_1_activation_3_1_Relu0_layer, 17,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &functional_1_activation_3_1_Relu0_chain,
  NULL, &functional_1_conv2d_2_1_BiasAdd0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  Conv__1180_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &functional_1_activation_2_1_Relu0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Conv__1180_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &Conv__1180_weights, &Conv__1180_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  Conv__1180_layer, 16,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_if32of32wf32,
  &Conv__1180_chain,
  NULL, &functional_1_activation_3_1_Relu0_layer, AI_STATIC, 
  .groups = 64, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  functional_1_activation_2_1_Relu0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &functional_1_conv2d_1_2_BiasAdd0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &functional_1_activation_2_1_Relu0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  functional_1_activation_2_1_Relu0_layer, 13,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &functional_1_activation_2_1_Relu0_chain,
  NULL, &Conv__1180_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  functional_1_conv2d_1_2_BiasAdd0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &functional_1_activation_1_2_Relu0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &functional_1_conv2d_1_2_BiasAdd0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &functional_1_conv2d_1_2_BiasAdd0_weights, &functional_1_conv2d_1_2_BiasAdd0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &functional_1_conv2d_1_2_BiasAdd0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  functional_1_conv2d_1_2_BiasAdd0_layer, 12,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &functional_1_conv2d_1_2_BiasAdd0_chain,
  NULL, &functional_1_activation_2_1_Relu0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  functional_1_activation_1_2_Relu0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Conv__1000_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &functional_1_activation_1_2_Relu0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  functional_1_activation_1_2_Relu0_layer, 9,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &functional_1_activation_1_2_Relu0_chain,
  NULL, &functional_1_conv2d_1_2_BiasAdd0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  Conv__1000_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &functional_1_activation_1_Relu0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Conv__1000_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &Conv__1000_weights, &Conv__1000_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  Conv__1000_layer, 8,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_if32of32wf32,
  &Conv__1000_chain,
  NULL, &functional_1_activation_1_2_Relu0_layer, AI_STATIC, 
  .groups = 64, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  functional_1_activation_1_Relu0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &functional_1_conv2d_1_BiasAdd0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &functional_1_activation_1_Relu0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  functional_1_activation_1_Relu0_layer, 5,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &functional_1_activation_1_Relu0_chain,
  NULL, &Conv__1000_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  functional_1_conv2d_1_BiasAdd0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &functional_1_conv2d_1_BiasAdd__60_to_chfirst_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &functional_1_conv2d_1_BiasAdd0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &functional_1_conv2d_1_BiasAdd0_weights, &functional_1_conv2d_1_BiasAdd0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &functional_1_conv2d_1_BiasAdd0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  functional_1_conv2d_1_BiasAdd0_layer, 4,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &functional_1_conv2d_1_BiasAdd0_chain,
  NULL, &functional_1_activation_1_Relu0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(2, 2), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 4, 1, 5, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  functional_1_conv2d_1_BiasAdd__60_to_chfirst_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_layer0_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &functional_1_conv2d_1_BiasAdd__60_to_chfirst_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  functional_1_conv2d_1_BiasAdd__60_to_chfirst_layer, 1,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &functional_1_conv2d_1_BiasAdd__60_to_chfirst_chain,
  NULL, &functional_1_conv2d_1_BiasAdd0_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_CHANNEL, AI_SHAPE_WIDTH, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)


#if (AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 128816, 1, 1),
    128816, NULL, NULL),
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 59648, 1, 1),
    59648, NULL, NULL),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_IN_NUM, &input_layer0_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_OUT_NUM, &Identity0_output),
  &functional_1_conv2d_1_BiasAdd__60_to_chfirst_layer, 0xa74433d4, NULL)

#else

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 128816, 1, 1),
      128816, NULL, NULL)
  ),
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 59648, 1, 1),
      59648, NULL, NULL)
  ),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_IN_NUM, &input_layer0_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_OUT_NUM, &Identity0_output),
  &functional_1_conv2d_1_BiasAdd__60_to_chfirst_layer, 0xa74433d4, NULL)

#endif	/*(AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)*/



/******************************************************************************/
AI_DECLARE_STATIC
ai_bool network_configure_activations(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_activations_map(g_network_activations_map, 1, params)) {
    /* Updating activations (byte) offsets */
    
    input_layer0_output_array.data = AI_PTR(g_network_activations_map[0] + 23568);
    input_layer0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 23568);
    functional_1_conv2d_1_BiasAdd__60_to_chfirst_output_array.data = AI_PTR(g_network_activations_map[0] + 25528);
    functional_1_conv2d_1_BiasAdd__60_to_chfirst_output_array.data_start = AI_PTR(g_network_activations_map[0] + 25528);
    functional_1_conv2d_1_BiasAdd0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 27488);
    functional_1_conv2d_1_BiasAdd0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 27488);
    functional_1_conv2d_1_BiasAdd0_output_array.data = AI_PTR(g_network_activations_map[0] + 27648);
    functional_1_conv2d_1_BiasAdd0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 27648);
    functional_1_activation_1_Relu0_output_array.data = AI_PTR(g_network_activations_map[0] + 27648);
    functional_1_activation_1_Relu0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 27648);
    Conv__1000_output_array.data = AI_PTR(g_network_activations_map[0] + 24320);
    Conv__1000_output_array.data_start = AI_PTR(g_network_activations_map[0] + 24320);
    functional_1_activation_1_2_Relu0_output_array.data = AI_PTR(g_network_activations_map[0] + 24320);
    functional_1_activation_1_2_Relu0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 24320);
    functional_1_conv2d_1_2_BiasAdd0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 59392);
    functional_1_conv2d_1_2_BiasAdd0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 59392);
    functional_1_conv2d_1_2_BiasAdd0_output_array.data = AI_PTR(g_network_activations_map[0] + 23040);
    functional_1_conv2d_1_2_BiasAdd0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 23040);
    functional_1_activation_2_1_Relu0_output_array.data = AI_PTR(g_network_activations_map[0] + 23040);
    functional_1_activation_2_1_Relu0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 23040);
    Conv__1180_output_array.data = AI_PTR(g_network_activations_map[0] + 19712);
    Conv__1180_output_array.data_start = AI_PTR(g_network_activations_map[0] + 19712);
    functional_1_activation_3_1_Relu0_output_array.data = AI_PTR(g_network_activations_map[0] + 19712);
    functional_1_activation_3_1_Relu0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 19712);
    functional_1_conv2d_2_1_BiasAdd0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 59392);
    functional_1_conv2d_2_1_BiasAdd0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 59392);
    functional_1_conv2d_2_1_BiasAdd0_output_array.data = AI_PTR(g_network_activations_map[0] + 18432);
    functional_1_conv2d_2_1_BiasAdd0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 18432);
    functional_1_activation_4_1_Relu0_output_array.data = AI_PTR(g_network_activations_map[0] + 18432);
    functional_1_activation_4_1_Relu0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 18432);
    Conv__1360_output_array.data = AI_PTR(g_network_activations_map[0] + 15104);
    Conv__1360_output_array.data_start = AI_PTR(g_network_activations_map[0] + 15104);
    functional_1_activation_5_1_Relu0_output_array.data = AI_PTR(g_network_activations_map[0] + 15104);
    functional_1_activation_5_1_Relu0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 15104);
    functional_1_conv2d_3_1_BiasAdd0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 59392);
    functional_1_conv2d_3_1_BiasAdd0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 59392);
    functional_1_conv2d_3_1_BiasAdd0_output_array.data = AI_PTR(g_network_activations_map[0] + 13824);
    functional_1_conv2d_3_1_BiasAdd0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 13824);
    functional_1_activation_6_1_Relu0_output_array.data = AI_PTR(g_network_activations_map[0] + 13824);
    functional_1_activation_6_1_Relu0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 13824);
    Conv__1540_output_array.data = AI_PTR(g_network_activations_map[0] + 10496);
    Conv__1540_output_array.data_start = AI_PTR(g_network_activations_map[0] + 10496);
    functional_1_activation_7_1_Relu0_output_array.data = AI_PTR(g_network_activations_map[0] + 10496);
    functional_1_activation_7_1_Relu0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 10496);
    functional_1_conv2d_4_1_BiasAdd0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 59392);
    functional_1_conv2d_4_1_BiasAdd0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 59392);
    functional_1_conv2d_4_1_BiasAdd0_output_array.data = AI_PTR(g_network_activations_map[0] + 9216);
    functional_1_conv2d_4_1_BiasAdd0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 9216);
    functional_1_activation_8_1_Relu0_output_array.data = AI_PTR(g_network_activations_map[0] + 9216);
    functional_1_activation_8_1_Relu0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 9216);
    Conv__1720_output_array.data = AI_PTR(g_network_activations_map[0] + 5888);
    Conv__1720_output_array.data_start = AI_PTR(g_network_activations_map[0] + 5888);
    functional_1_activation_9_1_Relu0_output_array.data = AI_PTR(g_network_activations_map[0] + 5888);
    functional_1_activation_9_1_Relu0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 5888);
    functional_1_conv2d_5_1_BiasAdd0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 59392);
    functional_1_conv2d_5_1_BiasAdd0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 59392);
    functional_1_conv2d_5_1_BiasAdd0_output_array.data = AI_PTR(g_network_activations_map[0] + 4608);
    functional_1_conv2d_5_1_BiasAdd0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 4608);
    functional_1_activation_10_1_Relu0_output_array.data = AI_PTR(g_network_activations_map[0] + 4608);
    functional_1_activation_10_1_Relu0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 4608);
    Conv__1900_output_array.data = AI_PTR(g_network_activations_map[0] + 1280);
    Conv__1900_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1280);
    functional_1_activation_11_1_Relu0_output_array.data = AI_PTR(g_network_activations_map[0] + 1280);
    functional_1_activation_11_1_Relu0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1280);
    functional_1_conv2d_6_1_BiasAdd0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 59392);
    functional_1_conv2d_6_1_BiasAdd0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 59392);
    functional_1_conv2d_6_1_BiasAdd0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    functional_1_conv2d_6_1_BiasAdd0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    functional_1_activation_12_1_Relu0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    functional_1_activation_12_1_Relu0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    functional_1_average_pooling2d_1_AvgPool0_output_array.data = AI_PTR(g_network_activations_map[0] + 32000);
    functional_1_average_pooling2d_1_AvgPool0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 32000);
    functional_1_dense_1_MatMul0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    functional_1_dense_1_MatMul0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    Identity0_output_array.data = AI_PTR(g_network_activations_map[0] + 48);
    Identity0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 48);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_ACTIVATIONS);
  return false;
}




/******************************************************************************/
AI_DECLARE_STATIC
ai_bool network_configure_weights(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_weights_map(g_network_weights_map, 1, params)) {
    /* Updating weights (byte) offsets */
    
    functional_1_conv2d_1_BiasAdd0_weights_array.format |= AI_FMT_FLAG_CONST;
    functional_1_conv2d_1_BiasAdd0_weights_array.data = AI_PTR(g_network_weights_map[0] + 0);
    functional_1_conv2d_1_BiasAdd0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 0);
    functional_1_conv2d_1_BiasAdd0_bias_array.format |= AI_FMT_FLAG_CONST;
    functional_1_conv2d_1_BiasAdd0_bias_array.data = AI_PTR(g_network_weights_map[0] + 10240);
    functional_1_conv2d_1_BiasAdd0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 10240);
    Conv__1000_weights_array.format |= AI_FMT_FLAG_CONST;
    Conv__1000_weights_array.data = AI_PTR(g_network_weights_map[0] + 10496);
    Conv__1000_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 10496);
    Conv__1000_bias_array.format |= AI_FMT_FLAG_CONST;
    Conv__1000_bias_array.data = AI_PTR(g_network_weights_map[0] + 12800);
    Conv__1000_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 12800);
    functional_1_conv2d_1_2_BiasAdd0_weights_array.format |= AI_FMT_FLAG_CONST;
    functional_1_conv2d_1_2_BiasAdd0_weights_array.data = AI_PTR(g_network_weights_map[0] + 13056);
    functional_1_conv2d_1_2_BiasAdd0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 13056);
    functional_1_conv2d_1_2_BiasAdd0_bias_array.format |= AI_FMT_FLAG_CONST;
    functional_1_conv2d_1_2_BiasAdd0_bias_array.data = AI_PTR(g_network_weights_map[0] + 29440);
    functional_1_conv2d_1_2_BiasAdd0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 29440);
    Conv__1180_weights_array.format |= AI_FMT_FLAG_CONST;
    Conv__1180_weights_array.data = AI_PTR(g_network_weights_map[0] + 29696);
    Conv__1180_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 29696);
    Conv__1180_bias_array.format |= AI_FMT_FLAG_CONST;
    Conv__1180_bias_array.data = AI_PTR(g_network_weights_map[0] + 32000);
    Conv__1180_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 32000);
    functional_1_conv2d_2_1_BiasAdd0_weights_array.format |= AI_FMT_FLAG_CONST;
    functional_1_conv2d_2_1_BiasAdd0_weights_array.data = AI_PTR(g_network_weights_map[0] + 32256);
    functional_1_conv2d_2_1_BiasAdd0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 32256);
    functional_1_conv2d_2_1_BiasAdd0_bias_array.format |= AI_FMT_FLAG_CONST;
    functional_1_conv2d_2_1_BiasAdd0_bias_array.data = AI_PTR(g_network_weights_map[0] + 48640);
    functional_1_conv2d_2_1_BiasAdd0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 48640);
    Conv__1360_weights_array.format |= AI_FMT_FLAG_CONST;
    Conv__1360_weights_array.data = AI_PTR(g_network_weights_map[0] + 48896);
    Conv__1360_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 48896);
    Conv__1360_bias_array.format |= AI_FMT_FLAG_CONST;
    Conv__1360_bias_array.data = AI_PTR(g_network_weights_map[0] + 51200);
    Conv__1360_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 51200);
    functional_1_conv2d_3_1_BiasAdd0_weights_array.format |= AI_FMT_FLAG_CONST;
    functional_1_conv2d_3_1_BiasAdd0_weights_array.data = AI_PTR(g_network_weights_map[0] + 51456);
    functional_1_conv2d_3_1_BiasAdd0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 51456);
    functional_1_conv2d_3_1_BiasAdd0_bias_array.format |= AI_FMT_FLAG_CONST;
    functional_1_conv2d_3_1_BiasAdd0_bias_array.data = AI_PTR(g_network_weights_map[0] + 67840);
    functional_1_conv2d_3_1_BiasAdd0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 67840);
    Conv__1540_weights_array.format |= AI_FMT_FLAG_CONST;
    Conv__1540_weights_array.data = AI_PTR(g_network_weights_map[0] + 68096);
    Conv__1540_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 68096);
    Conv__1540_bias_array.format |= AI_FMT_FLAG_CONST;
    Conv__1540_bias_array.data = AI_PTR(g_network_weights_map[0] + 70400);
    Conv__1540_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 70400);
    functional_1_conv2d_4_1_BiasAdd0_weights_array.format |= AI_FMT_FLAG_CONST;
    functional_1_conv2d_4_1_BiasAdd0_weights_array.data = AI_PTR(g_network_weights_map[0] + 70656);
    functional_1_conv2d_4_1_BiasAdd0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 70656);
    functional_1_conv2d_4_1_BiasAdd0_bias_array.format |= AI_FMT_FLAG_CONST;
    functional_1_conv2d_4_1_BiasAdd0_bias_array.data = AI_PTR(g_network_weights_map[0] + 87040);
    functional_1_conv2d_4_1_BiasAdd0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 87040);
    Conv__1720_weights_array.format |= AI_FMT_FLAG_CONST;
    Conv__1720_weights_array.data = AI_PTR(g_network_weights_map[0] + 87296);
    Conv__1720_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 87296);
    Conv__1720_bias_array.format |= AI_FMT_FLAG_CONST;
    Conv__1720_bias_array.data = AI_PTR(g_network_weights_map[0] + 89600);
    Conv__1720_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 89600);
    functional_1_conv2d_5_1_BiasAdd0_weights_array.format |= AI_FMT_FLAG_CONST;
    functional_1_conv2d_5_1_BiasAdd0_weights_array.data = AI_PTR(g_network_weights_map[0] + 89856);
    functional_1_conv2d_5_1_BiasAdd0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 89856);
    functional_1_conv2d_5_1_BiasAdd0_bias_array.format |= AI_FMT_FLAG_CONST;
    functional_1_conv2d_5_1_BiasAdd0_bias_array.data = AI_PTR(g_network_weights_map[0] + 106240);
    functional_1_conv2d_5_1_BiasAdd0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 106240);
    Conv__1900_weights_array.format |= AI_FMT_FLAG_CONST;
    Conv__1900_weights_array.data = AI_PTR(g_network_weights_map[0] + 106496);
    Conv__1900_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 106496);
    Conv__1900_bias_array.format |= AI_FMT_FLAG_CONST;
    Conv__1900_bias_array.data = AI_PTR(g_network_weights_map[0] + 108800);
    Conv__1900_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 108800);
    functional_1_conv2d_6_1_BiasAdd0_weights_array.format |= AI_FMT_FLAG_CONST;
    functional_1_conv2d_6_1_BiasAdd0_weights_array.data = AI_PTR(g_network_weights_map[0] + 109056);
    functional_1_conv2d_6_1_BiasAdd0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 109056);
    functional_1_conv2d_6_1_BiasAdd0_bias_array.format |= AI_FMT_FLAG_CONST;
    functional_1_conv2d_6_1_BiasAdd0_bias_array.data = AI_PTR(g_network_weights_map[0] + 125440);
    functional_1_conv2d_6_1_BiasAdd0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 125440);
    functional_1_dense_1_MatMul0_weights_array.format |= AI_FMT_FLAG_CONST;
    functional_1_dense_1_MatMul0_weights_array.data = AI_PTR(g_network_weights_map[0] + 125696);
    functional_1_dense_1_MatMul0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 125696);
    functional_1_dense_1_MatMul0_bias_array.format |= AI_FMT_FLAG_CONST;
    functional_1_dense_1_MatMul0_bias_array.data = AI_PTR(g_network_weights_map[0] + 128768);
    functional_1_dense_1_MatMul0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 128768);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_WEIGHTS);
  return false;
}


/**  PUBLIC APIs SECTION  *****************************************************/



AI_DEPRECATED
AI_API_ENTRY
ai_bool ai_network_get_info(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_NETWORK_MODEL_NAME,
      .model_signature   = AI_NETWORK_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 3937717,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .params            = AI_STRUCT_INIT,
      .activations       = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0xa74433d4,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}



AI_API_ENTRY
ai_bool ai_network_get_report(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_NETWORK_MODEL_NAME,
      .model_signature   = AI_NETWORK_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 3937717,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .map_signature     = AI_MAGIC_SIGNATURE,
      .map_weights       = AI_STRUCT_INIT,
      .map_activations   = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0xa74433d4,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}


AI_API_ENTRY
ai_error ai_network_get_error(ai_handle network)
{
  return ai_platform_network_get_error(network);
}


AI_API_ENTRY
ai_error ai_network_create(
  ai_handle* network, const ai_buffer* network_config)
{
  return ai_platform_network_create(
    network, network_config, 
    AI_CONTEXT_OBJ(&AI_NET_OBJ_INSTANCE),
    AI_TOOLS_API_VERSION_MAJOR, AI_TOOLS_API_VERSION_MINOR, AI_TOOLS_API_VERSION_MICRO);
}


AI_API_ENTRY
ai_error ai_network_create_and_init(
  ai_handle* network, const ai_handle activations[], const ai_handle weights[])
{
  ai_error err;
  ai_network_params params;

  err = ai_network_create(network, AI_NETWORK_DATA_CONFIG);
  if (err.type != AI_ERROR_NONE) {
    return err;
  }
  
  if (ai_network_data_params_get(&params) != true) {
    err = ai_network_get_error(*network);
    return err;
  }
#if defined(AI_NETWORK_DATA_ACTIVATIONS_COUNT)
  /* set the addresses of the activations buffers */
  for (ai_u16 idx=0; activations && idx<params.map_activations.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_activations, idx, activations[idx]);
  }
#endif
#if defined(AI_NETWORK_DATA_WEIGHTS_COUNT)
  /* set the addresses of the weight buffers */
  for (ai_u16 idx=0; weights && idx<params.map_weights.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_weights, idx, weights[idx]);
  }
#endif
  if (ai_network_init(*network, &params) != true) {
    err = ai_network_get_error(*network);
  }
  return err;
}


AI_API_ENTRY
ai_buffer* ai_network_inputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_inputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_buffer* ai_network_outputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_outputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_handle ai_network_destroy(ai_handle network)
{
  return ai_platform_network_destroy(network);
}


AI_API_ENTRY
ai_bool ai_network_init(
  ai_handle network, const ai_network_params* params)
{
  ai_network* net_ctx = AI_NETWORK_OBJ(ai_platform_network_init(network, params));
  ai_bool ok = true;

  if (!net_ctx) return false;
  ok &= network_configure_weights(net_ctx, params);
  ok &= network_configure_activations(net_ctx, params);

  ok &= ai_platform_network_post_init(network);

  return ok;
}


AI_API_ENTRY
ai_i32 ai_network_run(
  ai_handle network, const ai_buffer* input, ai_buffer* output)
{
  return ai_platform_network_process(network, input, output);
}


AI_API_ENTRY
ai_i32 ai_network_forward(ai_handle network, const ai_buffer* input)
{
  return ai_platform_network_process(network, input, NULL);
}



#undef AI_NETWORK_MODEL_SIGNATURE
#undef AI_NET_OBJ_INSTANCE
#undef AI_TOOLS_DATE_TIME
#undef AI_TOOLS_COMPILE_TIME

