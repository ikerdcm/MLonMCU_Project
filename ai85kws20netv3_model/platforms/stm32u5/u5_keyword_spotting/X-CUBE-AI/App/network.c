/**
  ******************************************************************************
  * @file    network.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-04-30T14:00:44+0200
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
#define AI_NETWORK_MODEL_SIGNATURE     "0xdb86a77d95172b296f49d176fd27a40c"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     ""
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "2026-04-30T14:00:44+0200"

#undef AI_TOOLS_COMPILE_TIME
#define AI_TOOLS_COMPILE_TIME    __DATE__ " " __TIME__

#undef AI_NETWORK_N_BATCHES
#define AI_NETWORK_N_BATCHES         (1)

static ai_ptr g_network_activations_map[1] = AI_C_ARRAY_INIT;
static ai_ptr g_network_weights_map[1] = AI_C_ARRAY_INIT;



/**  Array declarations section  **********************************************/
/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  kws_input_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 16384, AI_STATIC)

/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv1_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12800, AI_STATIC)

/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv1_activate_Relu_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12800, AI_STATIC)

/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv1_quantize_Mul_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12800, AI_STATIC)

/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv1_quantize_Round_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12800, AI_STATIC)

/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv1_quantize_Div_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12800, AI_STATIC)

/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv2_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12096, AI_STATIC)

/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv2_activate_Relu_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12096, AI_STATIC)

/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv2_quantize_Mul_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12096, AI_STATIC)

/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv2_quantize_Round_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12096, AI_STATIC)

/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv2_quantize_Div_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12096, AI_STATIC)

/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv2_Mul_2_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12096, AI_STATIC)

/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv3_pool_MaxPool_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6048, AI_STATIC)

/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv3_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4032, AI_STATIC)

/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv3_activate_Relu_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4032, AI_STATIC)

/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv3_quantize_Mul_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4032, AI_STATIC)

/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv3_quantize_Round_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4032, AI_STATIC)

/* Array#17 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv3_quantize_Div_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4032, AI_STATIC)

/* Array#18 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv4_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2928, AI_STATIC)

/* Array#19 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv4_activate_Relu_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2928, AI_STATIC)

/* Array#20 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv4_quantize_Mul_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2928, AI_STATIC)

/* Array#21 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv4_quantize_Round_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2928, AI_STATIC)

/* Array#22 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv4_quantize_Div_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2928, AI_STATIC)

/* Array#23 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv4_Mul_2_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2928, AI_STATIC)

/* Array#24 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv1_pool_MaxPool_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1440, AI_STATIC)

/* Array#25 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv1_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1920, AI_STATIC)

/* Array#26 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv1_activate_Relu_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1920, AI_STATIC)

/* Array#27 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv1_quantize_Mul_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1920, AI_STATIC)

/* Array#28 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv1_quantize_Round_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1920, AI_STATIC)

/* Array#29 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv1_quantize_Div_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1920, AI_STATIC)

/* Array#30 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv2_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2688, AI_STATIC)

/* Array#31 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv2_activate_Relu_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2688, AI_STATIC)

/* Array#32 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv2_quantize_Mul_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2688, AI_STATIC)

/* Array#33 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv2_quantize_Round_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2688, AI_STATIC)

/* Array#34 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv2_quantize_Div_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2688, AI_STATIC)

/* Array#35 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv2_Mul_2_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2688, AI_STATIC)

/* Array#36 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv3_pool_AveragePool_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1344, AI_STATIC)

/* Array#37 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv3_quantize_pool_Mul_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1344, AI_STATIC)

/* Array#38 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv3_quantize_pool_Floor_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1344, AI_STATIC)

/* Array#39 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv3_quantize_pool_Div_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1344, AI_STATIC)

/* Array#40 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv3_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1400, AI_STATIC)

/* Array#41 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv3_activate_Relu_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1400, AI_STATIC)

/* Array#42 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv3_quantize_Mul_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1400, AI_STATIC)

/* Array#43 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv3_quantize_Round_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1400, AI_STATIC)

/* Array#44 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv3_quantize_Div_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1400, AI_STATIC)

/* Array#45 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv3_Mul_2_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1400, AI_STATIC)

/* Array#46 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv4_pool_MaxPool_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 700, AI_STATIC)

/* Array#47 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv4_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 256, AI_STATIC)

/* Array#48 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv4_activate_Relu_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 256, AI_STATIC)

/* Array#49 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv4_quantize_Mul_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 256, AI_STATIC)

/* Array#50 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv4_quantize_Round_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 256, AI_STATIC)

/* Array#51 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv4_quantize_Div_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 256, AI_STATIC)

/* Array#52 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv4_Mul_2_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 256, AI_STATIC)

/* Array#53 */
AI_ARRAY_OBJ_DECLARE(
  _Reshape_output_0_to_chlast_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 256, AI_STATIC)

/* Array#54 */
AI_ARRAY_OBJ_DECLARE(
  _fc_Gemm_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 21, AI_STATIC)

/* Array#55 */
AI_ARRAY_OBJ_DECLARE(
  _fc_quantize_Round_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 21, AI_STATIC)

/* Array#56 */
AI_ARRAY_OBJ_DECLARE(
  _fc_quantize_Div_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 21, AI_STATIC)

/* Array#57 */
AI_ARRAY_OBJ_DECLARE(
  kws_logits_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 21, AI_STATIC)

/* Array#58 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv1_quantize_Constant_1_output_0_4D_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#59 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv2_quantize_Constant_1_output_0_4D_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#60 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv3_quantize_Constant_1_output_0_4D_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#61 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv4_quantize_Constant_1_output_0_4D_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#62 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv1_quantize_Constant_1_output_0_4D_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#63 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv2_quantize_Constant_1_output_0_4D_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#64 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv3_quantize_pool_Constant_1_output_0_4D_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#65 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv3_quantize_Constant_1_output_0_4D_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#66 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv4_quantize_Constant_1_output_0_4D_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#67 */
AI_ARRAY_OBJ_DECLARE(
  _fc_quantize_Constant_1_output_0_2D_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#68 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv1_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12800, AI_STATIC)

/* Array#69 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv1_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#70 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv1_quantize_Mul_output_0_scale_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#71 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv1_quantize_Mul_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#72 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv2_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 28800, AI_STATIC)

/* Array#73 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv2_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 96, AI_STATIC)

/* Array#74 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv2_quantize_Mul_output_0_scale_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 96, AI_STATIC)

/* Array#75 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv2_quantize_Mul_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 96, AI_STATIC)

/* Array#76 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv2_Mul_2_output_0_scale_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 96, AI_STATIC)

/* Array#77 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv3_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 18432, AI_STATIC)

/* Array#78 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv3_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#79 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv3_quantize_Mul_output_0_scale_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#80 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv3_quantize_Mul_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#81 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv4_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 9216, AI_STATIC)

/* Array#82 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv4_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 48, AI_STATIC)

/* Array#83 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv4_quantize_Mul_output_0_scale_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 48, AI_STATIC)

/* Array#84 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv4_quantize_Mul_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 48, AI_STATIC)

/* Array#85 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv4_Mul_2_output_0_scale_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 48, AI_STATIC)

/* Array#86 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv1_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 9216, AI_STATIC)

/* Array#87 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv1_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#88 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv2_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 18432, AI_STATIC)

/* Array#89 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv2_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 96, AI_STATIC)

/* Array#90 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv3_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 28800, AI_STATIC)

/* Array#91 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv3_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#92 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv3_Mul_2_output_0_scale_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 100, AI_STATIC)

/* Array#93 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv4_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 38400, AI_STATIC)

/* Array#94 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv4_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#95 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv4_Mul_2_output_0_scale_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#96 */
AI_ARRAY_OBJ_DECLARE(
  _fc_Gemm_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 5376, AI_STATIC)

/* Array#97 */
AI_ARRAY_OBJ_DECLARE(
  _fc_Gemm_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 21, AI_STATIC)

/* Array#98 */
AI_ARRAY_OBJ_DECLARE(
  kws_logits_scale_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 21, AI_STATIC)

/* Array#99 */
AI_ARRAY_OBJ_DECLARE(
  kws_logits_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 21, AI_STATIC)

/* Array#100 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv1_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#101 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv2_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 300, AI_STATIC)

/* Array#102 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv3_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 288, AI_STATIC)

/* Array#103 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv4_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 192, AI_STATIC)

/* Array#104 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv1_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 144, AI_STATIC)

/* Array#105 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv2_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 192, AI_STATIC)

/* Array#106 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv3_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 288, AI_STATIC)

/* Array#107 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv4_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 600, AI_STATIC)

/**  Tensor declarations section  *********************************************/
/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  _Reshape_output_0_to_chlast_output, AI_STATIC,
  0, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 4, 64), AI_STRIDE_INIT(4, 4, 4, 4, 16),
  1, &_Reshape_output_0_to_chlast_output_array, NULL)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  _Reshape_output_0_to_chlast_output0, AI_STATIC,
  1, 0x0,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &_Reshape_output_0_to_chlast_output_array, NULL)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  _fc_Gemm_output_0_bias, AI_STATIC,
  2, 0x0,
  AI_SHAPE_INIT(4, 1, 21, 1, 1), AI_STRIDE_INIT(4, 4, 4, 84, 84),
  1, &_fc_Gemm_output_0_bias_array, NULL)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  _fc_Gemm_output_0_output, AI_STATIC,
  3, 0x0,
  AI_SHAPE_INIT(4, 1, 21, 1, 1), AI_STRIDE_INIT(4, 4, 4, 84, 84),
  1, &_fc_Gemm_output_0_output_array, NULL)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  _fc_Gemm_output_0_weights, AI_STATIC,
  4, 0x0,
  AI_SHAPE_INIT(4, 256, 21, 1, 1), AI_STRIDE_INIT(4, 4, 1024, 21504, 21504),
  1, &_fc_Gemm_output_0_weights_array, NULL)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  _fc_quantize_Constant_1_output_0_2D, AI_STATIC,
  5, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &_fc_quantize_Constant_1_output_0_2D_array, NULL)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  _fc_quantize_Div_output_0_output, AI_STATIC,
  6, 0x0,
  AI_SHAPE_INIT(4, 1, 21, 1, 1), AI_STRIDE_INIT(4, 4, 4, 84, 84),
  1, &_fc_quantize_Div_output_0_output_array, NULL)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  _fc_quantize_Round_output_0_output, AI_STATIC,
  7, 0x0,
  AI_SHAPE_INIT(4, 1, 21, 1, 1), AI_STRIDE_INIT(4, 4, 4, 84, 84),
  1, &_fc_quantize_Round_output_0_output_array, NULL)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv1_Conv_output_0_bias, AI_STATIC,
  8, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_kws_conv1_Conv_output_0_bias_array, NULL)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv1_Conv_output_0_output, AI_STATIC,
  9, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 30), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_kws_conv1_Conv_output_0_output_array, NULL)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv1_Conv_output_0_scratch0, AI_STATIC,
  10, 0x0,
  AI_SHAPE_INIT(4, 1, 48, 1, 3), AI_STRIDE_INIT(4, 4, 4, 192, 192),
  1, &_kws_conv1_Conv_output_0_scratch0_array, NULL)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv1_Conv_output_0_weights, AI_STATIC,
  11, 0x0,
  AI_SHAPE_INIT(4, 48, 1, 3, 64), AI_STRIDE_INIT(4, 4, 192, 12288, 12288),
  1, &_kws_conv1_Conv_output_0_weights_array, NULL)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv1_activate_Relu_output_0_output, AI_STATIC,
  12, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 30), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_kws_conv1_activate_Relu_output_0_output_array, NULL)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv1_pool_MaxPool_output_0_output, AI_STATIC,
  13, 0x0,
  AI_SHAPE_INIT(4, 1, 48, 1, 30), AI_STRIDE_INIT(4, 4, 4, 192, 192),
  1, &_kws_conv1_pool_MaxPool_output_0_output_array, NULL)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv1_quantize_Constant_1_output_0_4D, AI_STATIC,
  14, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &_kws_conv1_quantize_Constant_1_output_0_4D_array, NULL)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv1_quantize_Div_output_0_output, AI_STATIC,
  15, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 30), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_kws_conv1_quantize_Div_output_0_output_array, NULL)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv1_quantize_Mul_output_0_output, AI_STATIC,
  16, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 30), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_kws_conv1_quantize_Mul_output_0_output_array, NULL)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv1_quantize_Round_output_0_output, AI_STATIC,
  17, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 30), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_kws_conv1_quantize_Round_output_0_output_array, NULL)

/* Tensor #18 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv2_Conv_output_0_bias, AI_STATIC,
  18, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 4, 4, 384, 384),
  1, &_kws_conv2_Conv_output_0_bias_array, NULL)

/* Tensor #19 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv2_Conv_output_0_output, AI_STATIC,
  19, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 28), AI_STRIDE_INIT(4, 4, 4, 384, 384),
  1, &_kws_conv2_Conv_output_0_output_array, NULL)

/* Tensor #20 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv2_Conv_output_0_scratch0, AI_STATIC,
  20, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 3), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_kws_conv2_Conv_output_0_scratch0_array, NULL)

/* Tensor #21 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv2_Conv_output_0_weights, AI_STATIC,
  21, 0x0,
  AI_SHAPE_INIT(4, 64, 1, 3, 96), AI_STRIDE_INIT(4, 4, 256, 24576, 24576),
  1, &_kws_conv2_Conv_output_0_weights_array, NULL)

/* Tensor #22 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv2_Mul_2_output_0_output, AI_STATIC,
  22, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 28), AI_STRIDE_INIT(4, 4, 4, 384, 384),
  1, &_kws_conv2_Mul_2_output_0_output_array, NULL)

/* Tensor #23 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv2_activate_Relu_output_0_output, AI_STATIC,
  23, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 28), AI_STRIDE_INIT(4, 4, 4, 384, 384),
  1, &_kws_conv2_activate_Relu_output_0_output_array, NULL)

/* Tensor #24 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv2_quantize_Constant_1_output_0_4D, AI_STATIC,
  24, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &_kws_conv2_quantize_Constant_1_output_0_4D_array, NULL)

/* Tensor #25 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv2_quantize_Div_output_0_output, AI_STATIC,
  25, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 28), AI_STRIDE_INIT(4, 4, 4, 384, 384),
  1, &_kws_conv2_quantize_Div_output_0_output_array, NULL)

/* Tensor #26 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv2_quantize_Mul_output_0_output, AI_STATIC,
  26, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 28), AI_STRIDE_INIT(4, 4, 4, 384, 384),
  1, &_kws_conv2_quantize_Mul_output_0_output_array, NULL)

/* Tensor #27 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv2_quantize_Round_output_0_output, AI_STATIC,
  27, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 28), AI_STRIDE_INIT(4, 4, 4, 384, 384),
  1, &_kws_conv2_quantize_Round_output_0_output_array, NULL)

/* Tensor #28 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv3_Conv_output_0_bias, AI_STATIC,
  28, 0x0,
  AI_SHAPE_INIT(4, 1, 100, 1, 1), AI_STRIDE_INIT(4, 4, 4, 400, 400),
  1, &_kws_conv3_Conv_output_0_bias_array, NULL)

/* Tensor #29 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv3_Conv_output_0_output, AI_STATIC,
  29, 0x0,
  AI_SHAPE_INIT(4, 1, 100, 1, 14), AI_STRIDE_INIT(4, 4, 4, 400, 400),
  1, &_kws_conv3_Conv_output_0_output_array, NULL)

/* Tensor #30 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv3_Conv_output_0_scratch0, AI_STATIC,
  30, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 3), AI_STRIDE_INIT(4, 4, 4, 384, 384),
  1, &_kws_conv3_Conv_output_0_scratch0_array, NULL)

/* Tensor #31 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv3_Conv_output_0_weights, AI_STATIC,
  31, 0x0,
  AI_SHAPE_INIT(4, 96, 1, 3, 100), AI_STRIDE_INIT(4, 4, 384, 38400, 38400),
  1, &_kws_conv3_Conv_output_0_weights_array, NULL)

/* Tensor #32 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv3_Mul_2_output_0_output, AI_STATIC,
  32, 0x0,
  AI_SHAPE_INIT(4, 1, 100, 1, 14), AI_STRIDE_INIT(4, 4, 4, 400, 400),
  1, &_kws_conv3_Mul_2_output_0_output_array, NULL)

/* Tensor #33 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv3_Mul_2_output_0_scale, AI_STATIC,
  33, 0x0,
  AI_SHAPE_INIT(4, 1, 100, 1, 1), AI_STRIDE_INIT(4, 4, 4, 400, 400),
  1, &_kws_conv3_Mul_2_output_0_scale_array, NULL)

/* Tensor #34 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv3_activate_Relu_output_0_output, AI_STATIC,
  34, 0x0,
  AI_SHAPE_INIT(4, 1, 100, 1, 14), AI_STRIDE_INIT(4, 4, 4, 400, 400),
  1, &_kws_conv3_activate_Relu_output_0_output_array, NULL)

/* Tensor #35 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv3_pool_AveragePool_output_0_output, AI_STATIC,
  35, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 14), AI_STRIDE_INIT(4, 4, 4, 384, 384),
  1, &_kws_conv3_pool_AveragePool_output_0_output_array, NULL)

/* Tensor #36 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv3_quantize_Constant_1_output_0_4D, AI_STATIC,
  36, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &_kws_conv3_quantize_Constant_1_output_0_4D_array, NULL)

/* Tensor #37 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv3_quantize_Div_output_0_output, AI_STATIC,
  37, 0x0,
  AI_SHAPE_INIT(4, 1, 100, 1, 14), AI_STRIDE_INIT(4, 4, 4, 400, 400),
  1, &_kws_conv3_quantize_Div_output_0_output_array, NULL)

/* Tensor #38 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv3_quantize_Mul_output_0_output, AI_STATIC,
  38, 0x0,
  AI_SHAPE_INIT(4, 1, 100, 1, 14), AI_STRIDE_INIT(4, 4, 4, 400, 400),
  1, &_kws_conv3_quantize_Mul_output_0_output_array, NULL)

/* Tensor #39 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv3_quantize_Round_output_0_output, AI_STATIC,
  39, 0x0,
  AI_SHAPE_INIT(4, 1, 100, 1, 14), AI_STRIDE_INIT(4, 4, 4, 400, 400),
  1, &_kws_conv3_quantize_Round_output_0_output_array, NULL)

/* Tensor #40 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv3_quantize_pool_Constant_1_output_0_4D, AI_STATIC,
  40, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &_kws_conv3_quantize_pool_Constant_1_output_0_4D_array, NULL)

/* Tensor #41 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv3_quantize_pool_Div_output_0_output, AI_STATIC,
  41, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 14), AI_STRIDE_INIT(4, 4, 4, 384, 384),
  1, &_kws_conv3_quantize_pool_Div_output_0_output_array, NULL)

/* Tensor #42 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv3_quantize_pool_Floor_output_0_output, AI_STATIC,
  42, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 14), AI_STRIDE_INIT(4, 4, 4, 384, 384),
  1, &_kws_conv3_quantize_pool_Floor_output_0_output_array, NULL)

/* Tensor #43 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv3_quantize_pool_Mul_output_0_output, AI_STATIC,
  43, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 14), AI_STRIDE_INIT(4, 4, 4, 384, 384),
  1, &_kws_conv3_quantize_pool_Mul_output_0_output_array, NULL)

/* Tensor #44 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv4_Conv_output_0_bias, AI_STATIC,
  44, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_kws_conv4_Conv_output_0_bias_array, NULL)

/* Tensor #45 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv4_Conv_output_0_output, AI_STATIC,
  45, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 4), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_kws_conv4_Conv_output_0_output_array, NULL)

/* Tensor #46 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv4_Conv_output_0_scratch0, AI_STATIC,
  46, 0x0,
  AI_SHAPE_INIT(4, 1, 100, 1, 6), AI_STRIDE_INIT(4, 4, 4, 400, 400),
  1, &_kws_conv4_Conv_output_0_scratch0_array, NULL)

/* Tensor #47 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv4_Conv_output_0_weights, AI_STATIC,
  47, 0x0,
  AI_SHAPE_INIT(4, 100, 1, 6, 64), AI_STRIDE_INIT(4, 4, 400, 25600, 25600),
  1, &_kws_conv4_Conv_output_0_weights_array, NULL)

/* Tensor #48 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv4_Mul_2_output_0_output, AI_STATIC,
  48, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 4), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_kws_conv4_Mul_2_output_0_output_array, NULL)

/* Tensor #49 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv4_Mul_2_output_0_scale, AI_STATIC,
  49, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_kws_conv4_Mul_2_output_0_scale_array, NULL)

/* Tensor #50 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv4_activate_Relu_output_0_output, AI_STATIC,
  50, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 4), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_kws_conv4_activate_Relu_output_0_output_array, NULL)

/* Tensor #51 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv4_pool_MaxPool_output_0_output, AI_STATIC,
  51, 0x0,
  AI_SHAPE_INIT(4, 1, 100, 1, 7), AI_STRIDE_INIT(4, 4, 4, 400, 400),
  1, &_kws_conv4_pool_MaxPool_output_0_output_array, NULL)

/* Tensor #52 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv4_quantize_Constant_1_output_0_4D, AI_STATIC,
  52, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &_kws_conv4_quantize_Constant_1_output_0_4D_array, NULL)

/* Tensor #53 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv4_quantize_Div_output_0_output, AI_STATIC,
  53, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 4), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_kws_conv4_quantize_Div_output_0_output_array, NULL)

/* Tensor #54 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv4_quantize_Mul_output_0_output, AI_STATIC,
  54, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 4), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_kws_conv4_quantize_Mul_output_0_output_array, NULL)

/* Tensor #55 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv4_quantize_Round_output_0_output, AI_STATIC,
  55, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 4), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_kws_conv4_quantize_Round_output_0_output_array, NULL)

/* Tensor #56 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv1_Conv_output_0_bias, AI_STATIC,
  56, 0x0,
  AI_SHAPE_INIT(4, 1, 100, 1, 1), AI_STRIDE_INIT(4, 4, 4, 400, 400),
  1, &_voice_conv1_Conv_output_0_bias_array, NULL)

/* Tensor #57 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv1_Conv_output_0_output, AI_STATIC,
  57, 0x0,
  AI_SHAPE_INIT(4, 1, 100, 1, 128), AI_STRIDE_INIT(4, 4, 4, 400, 400),
  1, &_voice_conv1_Conv_output_0_output_array, NULL)

/* Tensor #58 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv1_Conv_output_0_scratch0, AI_STATIC,
  58, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 128, 1), AI_STRIDE_INIT(4, 4, 4, 4, 512),
  1, &_voice_conv1_Conv_output_0_scratch0_array, NULL)

/* Tensor #59 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv1_Conv_output_0_weights, AI_STATIC,
  59, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 100), AI_STRIDE_INIT(4, 4, 4, 400, 51200),
  1, &_voice_conv1_Conv_output_0_weights_array, NULL)

/* Tensor #60 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv1_activate_Relu_output_0_output, AI_STATIC,
  60, 0x0,
  AI_SHAPE_INIT(4, 1, 100, 1, 128), AI_STRIDE_INIT(4, 4, 4, 400, 400),
  1, &_voice_conv1_activate_Relu_output_0_output_array, NULL)

/* Tensor #61 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv1_quantize_Constant_1_output_0_4D, AI_STATIC,
  61, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &_voice_conv1_quantize_Constant_1_output_0_4D_array, NULL)

/* Tensor #62 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv1_quantize_Div_output_0_output, AI_STATIC,
  62, 0x0,
  AI_SHAPE_INIT(4, 1, 100, 1, 128), AI_STRIDE_INIT(4, 4, 4, 400, 400),
  1, &_voice_conv1_quantize_Div_output_0_output_array, NULL)

/* Tensor #63 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv1_quantize_Mul_output_0_bias, AI_STATIC,
  63, 0x0,
  AI_SHAPE_INIT(4, 1, 100, 1, 1), AI_STRIDE_INIT(4, 4, 4, 400, 400),
  1, &_voice_conv1_quantize_Mul_output_0_bias_array, NULL)

/* Tensor #64 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv1_quantize_Mul_output_0_output, AI_STATIC,
  64, 0x0,
  AI_SHAPE_INIT(4, 1, 100, 1, 128), AI_STRIDE_INIT(4, 4, 4, 400, 400),
  1, &_voice_conv1_quantize_Mul_output_0_output_array, NULL)

/* Tensor #65 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv1_quantize_Mul_output_0_scale, AI_STATIC,
  65, 0x0,
  AI_SHAPE_INIT(4, 1, 100, 1, 1), AI_STRIDE_INIT(4, 4, 4, 400, 400),
  1, &_voice_conv1_quantize_Mul_output_0_scale_array, NULL)

/* Tensor #66 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv1_quantize_Round_output_0_output, AI_STATIC,
  66, 0x0,
  AI_SHAPE_INIT(4, 1, 100, 1, 128), AI_STRIDE_INIT(4, 4, 4, 400, 400),
  1, &_voice_conv1_quantize_Round_output_0_output_array, NULL)

/* Tensor #67 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv2_Conv_output_0_bias, AI_STATIC,
  67, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 4, 4, 384, 384),
  1, &_voice_conv2_Conv_output_0_bias_array, NULL)

/* Tensor #68 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv2_Conv_output_0_output, AI_STATIC,
  68, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 126), AI_STRIDE_INIT(4, 4, 4, 384, 384),
  1, &_voice_conv2_Conv_output_0_output_array, NULL)

/* Tensor #69 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv2_Conv_output_0_scratch0, AI_STATIC,
  69, 0x0,
  AI_SHAPE_INIT(4, 1, 100, 1, 3), AI_STRIDE_INIT(4, 4, 4, 400, 400),
  1, &_voice_conv2_Conv_output_0_scratch0_array, NULL)

/* Tensor #70 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv2_Conv_output_0_weights, AI_STATIC,
  70, 0x0,
  AI_SHAPE_INIT(4, 100, 1, 3, 96), AI_STRIDE_INIT(4, 4, 400, 38400, 38400),
  1, &_voice_conv2_Conv_output_0_weights_array, NULL)

/* Tensor #71 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv2_Mul_2_output_0_output, AI_STATIC,
  71, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 126), AI_STRIDE_INIT(4, 4, 4, 384, 384),
  1, &_voice_conv2_Mul_2_output_0_output_array, NULL)

/* Tensor #72 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv2_Mul_2_output_0_scale, AI_STATIC,
  72, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 4, 4, 384, 384),
  1, &_voice_conv2_Mul_2_output_0_scale_array, NULL)

/* Tensor #73 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv2_activate_Relu_output_0_output, AI_STATIC,
  73, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 126), AI_STRIDE_INIT(4, 4, 4, 384, 384),
  1, &_voice_conv2_activate_Relu_output_0_output_array, NULL)

/* Tensor #74 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv2_quantize_Constant_1_output_0_4D, AI_STATIC,
  74, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &_voice_conv2_quantize_Constant_1_output_0_4D_array, NULL)

/* Tensor #75 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv2_quantize_Div_output_0_output, AI_STATIC,
  75, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 126), AI_STRIDE_INIT(4, 4, 4, 384, 384),
  1, &_voice_conv2_quantize_Div_output_0_output_array, NULL)

/* Tensor #76 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv2_quantize_Mul_output_0_bias, AI_STATIC,
  76, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 4, 4, 384, 384),
  1, &_voice_conv2_quantize_Mul_output_0_bias_array, NULL)

/* Tensor #77 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv2_quantize_Mul_output_0_output, AI_STATIC,
  77, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 126), AI_STRIDE_INIT(4, 4, 4, 384, 384),
  1, &_voice_conv2_quantize_Mul_output_0_output_array, NULL)

/* Tensor #78 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv2_quantize_Mul_output_0_scale, AI_STATIC,
  78, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 4, 4, 384, 384),
  1, &_voice_conv2_quantize_Mul_output_0_scale_array, NULL)

/* Tensor #79 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv2_quantize_Round_output_0_output, AI_STATIC,
  79, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 126), AI_STRIDE_INIT(4, 4, 4, 384, 384),
  1, &_voice_conv2_quantize_Round_output_0_output_array, NULL)

/* Tensor #80 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv3_Conv_output_0_bias, AI_STATIC,
  80, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_voice_conv3_Conv_output_0_bias_array, NULL)

/* Tensor #81 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv3_Conv_output_0_output, AI_STATIC,
  81, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 63), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_voice_conv3_Conv_output_0_output_array, NULL)

/* Tensor #82 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv3_Conv_output_0_scratch0, AI_STATIC,
  82, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 3), AI_STRIDE_INIT(4, 4, 4, 384, 384),
  1, &_voice_conv3_Conv_output_0_scratch0_array, NULL)

/* Tensor #83 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv3_Conv_output_0_weights, AI_STATIC,
  83, 0x0,
  AI_SHAPE_INIT(4, 96, 1, 3, 64), AI_STRIDE_INIT(4, 4, 384, 24576, 24576),
  1, &_voice_conv3_Conv_output_0_weights_array, NULL)

/* Tensor #84 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv3_activate_Relu_output_0_output, AI_STATIC,
  84, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 63), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_voice_conv3_activate_Relu_output_0_output_array, NULL)

/* Tensor #85 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv3_pool_MaxPool_output_0_output, AI_STATIC,
  85, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 63), AI_STRIDE_INIT(4, 4, 4, 384, 384),
  1, &_voice_conv3_pool_MaxPool_output_0_output_array, NULL)

/* Tensor #86 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv3_quantize_Constant_1_output_0_4D, AI_STATIC,
  86, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &_voice_conv3_quantize_Constant_1_output_0_4D_array, NULL)

/* Tensor #87 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv3_quantize_Div_output_0_output, AI_STATIC,
  87, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 63), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_voice_conv3_quantize_Div_output_0_output_array, NULL)

/* Tensor #88 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv3_quantize_Mul_output_0_bias, AI_STATIC,
  88, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_voice_conv3_quantize_Mul_output_0_bias_array, NULL)

/* Tensor #89 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv3_quantize_Mul_output_0_output, AI_STATIC,
  89, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 63), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_voice_conv3_quantize_Mul_output_0_output_array, NULL)

/* Tensor #90 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv3_quantize_Mul_output_0_scale, AI_STATIC,
  90, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_voice_conv3_quantize_Mul_output_0_scale_array, NULL)

/* Tensor #91 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv3_quantize_Round_output_0_output, AI_STATIC,
  91, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 63), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_voice_conv3_quantize_Round_output_0_output_array, NULL)

/* Tensor #92 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv4_Conv_output_0_bias, AI_STATIC,
  92, 0x0,
  AI_SHAPE_INIT(4, 1, 48, 1, 1), AI_STRIDE_INIT(4, 4, 4, 192, 192),
  1, &_voice_conv4_Conv_output_0_bias_array, NULL)

/* Tensor #93 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv4_Conv_output_0_output, AI_STATIC,
  93, 0x0,
  AI_SHAPE_INIT(4, 1, 48, 1, 61), AI_STRIDE_INIT(4, 4, 4, 192, 192),
  1, &_voice_conv4_Conv_output_0_output_array, NULL)

/* Tensor #94 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv4_Conv_output_0_scratch0, AI_STATIC,
  94, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 3), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_voice_conv4_Conv_output_0_scratch0_array, NULL)

/* Tensor #95 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv4_Conv_output_0_weights, AI_STATIC,
  95, 0x0,
  AI_SHAPE_INIT(4, 64, 1, 3, 48), AI_STRIDE_INIT(4, 4, 256, 12288, 12288),
  1, &_voice_conv4_Conv_output_0_weights_array, NULL)

/* Tensor #96 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv4_Mul_2_output_0_output, AI_STATIC,
  96, 0x0,
  AI_SHAPE_INIT(4, 1, 48, 1, 61), AI_STRIDE_INIT(4, 4, 4, 192, 192),
  1, &_voice_conv4_Mul_2_output_0_output_array, NULL)

/* Tensor #97 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv4_Mul_2_output_0_scale, AI_STATIC,
  97, 0x0,
  AI_SHAPE_INIT(4, 1, 48, 1, 1), AI_STRIDE_INIT(4, 4, 4, 192, 192),
  1, &_voice_conv4_Mul_2_output_0_scale_array, NULL)

/* Tensor #98 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv4_activate_Relu_output_0_output, AI_STATIC,
  98, 0x0,
  AI_SHAPE_INIT(4, 1, 48, 1, 61), AI_STRIDE_INIT(4, 4, 4, 192, 192),
  1, &_voice_conv4_activate_Relu_output_0_output_array, NULL)

/* Tensor #99 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv4_quantize_Constant_1_output_0_4D, AI_STATIC,
  99, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &_voice_conv4_quantize_Constant_1_output_0_4D_array, NULL)

/* Tensor #100 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv4_quantize_Div_output_0_output, AI_STATIC,
  100, 0x0,
  AI_SHAPE_INIT(4, 1, 48, 1, 61), AI_STRIDE_INIT(4, 4, 4, 192, 192),
  1, &_voice_conv4_quantize_Div_output_0_output_array, NULL)

/* Tensor #101 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv4_quantize_Mul_output_0_bias, AI_STATIC,
  101, 0x0,
  AI_SHAPE_INIT(4, 1, 48, 1, 1), AI_STRIDE_INIT(4, 4, 4, 192, 192),
  1, &_voice_conv4_quantize_Mul_output_0_bias_array, NULL)

/* Tensor #102 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv4_quantize_Mul_output_0_output, AI_STATIC,
  102, 0x0,
  AI_SHAPE_INIT(4, 1, 48, 1, 61), AI_STRIDE_INIT(4, 4, 4, 192, 192),
  1, &_voice_conv4_quantize_Mul_output_0_output_array, NULL)

/* Tensor #103 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv4_quantize_Mul_output_0_scale, AI_STATIC,
  103, 0x0,
  AI_SHAPE_INIT(4, 1, 48, 1, 1), AI_STRIDE_INIT(4, 4, 4, 192, 192),
  1, &_voice_conv4_quantize_Mul_output_0_scale_array, NULL)

/* Tensor #104 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv4_quantize_Round_output_0_output, AI_STATIC,
  104, 0x0,
  AI_SHAPE_INIT(4, 1, 48, 1, 61), AI_STRIDE_INIT(4, 4, 4, 192, 192),
  1, &_voice_conv4_quantize_Round_output_0_output_array, NULL)

/* Tensor #105 */
AI_TENSOR_OBJ_DECLARE(
  kws_input_output, AI_STATIC,
  105, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 128, 128), AI_STRIDE_INIT(4, 4, 4, 4, 512),
  1, &kws_input_output_array, NULL)

/* Tensor #106 */
AI_TENSOR_OBJ_DECLARE(
  kws_logits_bias, AI_STATIC,
  106, 0x0,
  AI_SHAPE_INIT(4, 1, 21, 1, 1), AI_STRIDE_INIT(4, 4, 4, 84, 84),
  1, &kws_logits_bias_array, NULL)

/* Tensor #107 */
AI_TENSOR_OBJ_DECLARE(
  kws_logits_output, AI_STATIC,
  107, 0x0,
  AI_SHAPE_INIT(4, 1, 21, 1, 1), AI_STRIDE_INIT(4, 4, 4, 84, 84),
  1, &kws_logits_output_array, NULL)

/* Tensor #108 */
AI_TENSOR_OBJ_DECLARE(
  kws_logits_scale, AI_STATIC,
  108, 0x0,
  AI_SHAPE_INIT(4, 1, 21, 1, 1), AI_STRIDE_INIT(4, 4, 4, 84, 84),
  1, &kws_logits_scale_array, NULL)



/**  Layer declarations section  **********************************************/


AI_TENSOR_CHAIN_OBJ_DECLARE(
  kws_logits_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_fc_quantize_Div_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &kws_logits_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &kws_logits_scale, &kws_logits_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  kws_logits_layer, 69,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &kws_logits_chain,
  NULL, &kws_logits_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _fc_quantize_Div_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_fc_quantize_Round_output_0_output, &_fc_quantize_Constant_1_output_0_2D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_fc_quantize_Div_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _fc_quantize_Div_output_0_layer, 68,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_fc_quantize_Div_output_0_chain,
  NULL, &kws_logits_layer, AI_STATIC, 
  .operation = ai_div_f32, 
  .buffer_operation = ai_div_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _fc_quantize_Round_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_fc_Gemm_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_fc_quantize_Round_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _fc_quantize_Round_output_0_layer, 67,
  NL_TYPE, 0x0, NULL,
  nl, forward_round,
  &_fc_quantize_Round_output_0_chain,
  NULL, &_fc_quantize_Div_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _fc_Gemm_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Reshape_output_0_to_chlast_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_fc_Gemm_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_fc_Gemm_output_0_weights, &_fc_Gemm_output_0_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _fc_Gemm_output_0_layer, 66,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &_fc_Gemm_output_0_chain,
  NULL, &_fc_quantize_Round_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Reshape_output_0_to_chlast_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv4_Mul_2_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Reshape_output_0_to_chlast_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _Reshape_output_0_to_chlast_layer, 64,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &_Reshape_output_0_to_chlast_chain,
  NULL, &_fc_Gemm_output_0_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_WIDTH, AI_SHAPE_HEIGHT, AI_SHAPE_CHANNEL, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _kws_conv4_Mul_2_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv4_quantize_Div_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv4_Mul_2_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_kws_conv4_Mul_2_output_0_scale, &_voice_conv3_quantize_Mul_output_0_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _kws_conv4_Mul_2_output_0_layer, 63,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &_kws_conv4_Mul_2_output_0_chain,
  NULL, &_Reshape_output_0_to_chlast_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _kws_conv4_quantize_Div_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_kws_conv4_quantize_Round_output_0_output, &_kws_conv4_quantize_Constant_1_output_0_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv4_quantize_Div_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _kws_conv4_quantize_Div_output_0_layer, 62,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_kws_conv4_quantize_Div_output_0_chain,
  NULL, &_kws_conv4_Mul_2_output_0_layer, AI_STATIC, 
  .operation = ai_div_f32, 
  .buffer_operation = ai_div_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _kws_conv4_quantize_Round_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv4_quantize_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv4_quantize_Round_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _kws_conv4_quantize_Round_output_0_layer, 61,
  NL_TYPE, 0x0, NULL,
  nl, forward_round,
  &_kws_conv4_quantize_Round_output_0_chain,
  NULL, &_kws_conv4_quantize_Div_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _kws_conv4_quantize_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv4_activate_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv4_quantize_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_voice_conv3_quantize_Mul_output_0_scale, &_voice_conv3_quantize_Mul_output_0_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _kws_conv4_quantize_Mul_output_0_layer, 60,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &_kws_conv4_quantize_Mul_output_0_chain,
  NULL, &_kws_conv4_quantize_Round_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _kws_conv4_activate_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv4_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv4_activate_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _kws_conv4_activate_Relu_output_0_layer, 59,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &_kws_conv4_activate_Relu_output_0_chain,
  NULL, &_kws_conv4_quantize_Mul_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _kws_conv4_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv4_pool_MaxPool_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv4_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_kws_conv4_Conv_output_0_weights, &_kws_conv4_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_kws_conv4_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _kws_conv4_Conv_output_0_layer, 58,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_kws_conv4_Conv_output_0_chain,
  NULL, &_kws_conv4_activate_Relu_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 0, 1, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _kws_conv4_pool_MaxPool_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv3_Mul_2_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv4_pool_MaxPool_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _kws_conv4_pool_MaxPool_output_0_layer, 56,
  POOL_TYPE, 0x0, NULL,
  pool, forward_mp,
  &_kws_conv4_pool_MaxPool_output_0_chain,
  NULL, &_kws_conv4_Conv_output_0_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(1, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(1, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _kws_conv3_Mul_2_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv3_quantize_Div_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv3_Mul_2_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_kws_conv3_Mul_2_output_0_scale, &_voice_conv1_quantize_Mul_output_0_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _kws_conv3_Mul_2_output_0_layer, 55,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &_kws_conv3_Mul_2_output_0_chain,
  NULL, &_kws_conv4_pool_MaxPool_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _kws_conv3_quantize_Div_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_kws_conv3_quantize_Round_output_0_output, &_kws_conv3_quantize_Constant_1_output_0_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv3_quantize_Div_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _kws_conv3_quantize_Div_output_0_layer, 54,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_kws_conv3_quantize_Div_output_0_chain,
  NULL, &_kws_conv3_Mul_2_output_0_layer, AI_STATIC, 
  .operation = ai_div_f32, 
  .buffer_operation = ai_div_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _kws_conv3_quantize_Round_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv3_quantize_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv3_quantize_Round_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _kws_conv3_quantize_Round_output_0_layer, 53,
  NL_TYPE, 0x0, NULL,
  nl, forward_round,
  &_kws_conv3_quantize_Round_output_0_chain,
  NULL, &_kws_conv3_quantize_Div_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _kws_conv3_quantize_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv3_activate_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv3_quantize_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_voice_conv1_quantize_Mul_output_0_scale, &_voice_conv1_quantize_Mul_output_0_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _kws_conv3_quantize_Mul_output_0_layer, 52,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &_kws_conv3_quantize_Mul_output_0_chain,
  NULL, &_kws_conv3_quantize_Round_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _kws_conv3_activate_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv3_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv3_activate_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _kws_conv3_activate_Relu_output_0_layer, 51,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &_kws_conv3_activate_Relu_output_0_chain,
  NULL, &_kws_conv3_quantize_Mul_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _kws_conv3_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv3_quantize_pool_Div_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv3_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_kws_conv3_Conv_output_0_weights, &_kws_conv3_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_kws_conv3_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _kws_conv3_Conv_output_0_layer, 50,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_kws_conv3_Conv_output_0_chain,
  NULL, &_kws_conv3_activate_Relu_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 0, 1, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _kws_conv3_quantize_pool_Div_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_kws_conv3_quantize_pool_Floor_output_0_output, &_kws_conv3_quantize_pool_Constant_1_output_0_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv3_quantize_pool_Div_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _kws_conv3_quantize_pool_Div_output_0_layer, 48,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_kws_conv3_quantize_pool_Div_output_0_chain,
  NULL, &_kws_conv3_Conv_output_0_layer, AI_STATIC, 
  .operation = ai_div_f32, 
  .buffer_operation = ai_div_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _kws_conv3_quantize_pool_Floor_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv3_quantize_pool_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv3_quantize_pool_Floor_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _kws_conv3_quantize_pool_Floor_output_0_layer, 47,
  NL_TYPE, 0x0, NULL,
  nl, forward_floor,
  &_kws_conv3_quantize_pool_Floor_output_0_chain,
  NULL, &_kws_conv3_quantize_pool_Div_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _kws_conv3_quantize_pool_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv3_pool_AveragePool_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv3_quantize_pool_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_voice_conv2_quantize_Mul_output_0_scale, &_voice_conv2_quantize_Mul_output_0_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _kws_conv3_quantize_pool_Mul_output_0_layer, 46,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &_kws_conv3_quantize_pool_Mul_output_0_chain,
  NULL, &_kws_conv3_quantize_pool_Floor_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _kws_conv3_pool_AveragePool_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv2_Mul_2_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv3_pool_AveragePool_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _kws_conv3_pool_AveragePool_output_0_layer, 45,
  POOL_TYPE, 0x0, NULL,
  pool, forward_ap,
  &_kws_conv3_pool_AveragePool_output_0_chain,
  NULL, &_kws_conv3_quantize_pool_Mul_output_0_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(1, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(1, 2), 
  .count_include_pad = 1, 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _kws_conv2_Mul_2_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv2_quantize_Div_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv2_Mul_2_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_voice_conv2_Mul_2_output_0_scale, &_voice_conv2_quantize_Mul_output_0_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _kws_conv2_Mul_2_output_0_layer, 44,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &_kws_conv2_Mul_2_output_0_chain,
  NULL, &_kws_conv3_pool_AveragePool_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _kws_conv2_quantize_Div_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_kws_conv2_quantize_Round_output_0_output, &_kws_conv2_quantize_Constant_1_output_0_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv2_quantize_Div_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _kws_conv2_quantize_Div_output_0_layer, 43,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_kws_conv2_quantize_Div_output_0_chain,
  NULL, &_kws_conv2_Mul_2_output_0_layer, AI_STATIC, 
  .operation = ai_div_f32, 
  .buffer_operation = ai_div_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _kws_conv2_quantize_Round_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv2_quantize_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv2_quantize_Round_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _kws_conv2_quantize_Round_output_0_layer, 42,
  NL_TYPE, 0x0, NULL,
  nl, forward_round,
  &_kws_conv2_quantize_Round_output_0_chain,
  NULL, &_kws_conv2_quantize_Div_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _kws_conv2_quantize_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv2_activate_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv2_quantize_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_voice_conv2_quantize_Mul_output_0_scale, &_voice_conv2_quantize_Mul_output_0_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _kws_conv2_quantize_Mul_output_0_layer, 41,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &_kws_conv2_quantize_Mul_output_0_chain,
  NULL, &_kws_conv2_quantize_Round_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _kws_conv2_activate_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv2_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv2_activate_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _kws_conv2_activate_Relu_output_0_layer, 40,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &_kws_conv2_activate_Relu_output_0_chain,
  NULL, &_kws_conv2_quantize_Mul_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _kws_conv2_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv1_quantize_Div_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv2_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_kws_conv2_Conv_output_0_weights, &_kws_conv2_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_kws_conv2_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _kws_conv2_Conv_output_0_layer, 39,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_kws_conv2_Conv_output_0_chain,
  NULL, &_kws_conv2_activate_Relu_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _kws_conv1_quantize_Div_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_kws_conv1_quantize_Round_output_0_output, &_kws_conv1_quantize_Constant_1_output_0_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv1_quantize_Div_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _kws_conv1_quantize_Div_output_0_layer, 36,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_kws_conv1_quantize_Div_output_0_chain,
  NULL, &_kws_conv2_Conv_output_0_layer, AI_STATIC, 
  .operation = ai_div_f32, 
  .buffer_operation = ai_div_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _kws_conv1_quantize_Round_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv1_quantize_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv1_quantize_Round_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _kws_conv1_quantize_Round_output_0_layer, 35,
  NL_TYPE, 0x0, NULL,
  nl, forward_round,
  &_kws_conv1_quantize_Round_output_0_chain,
  NULL, &_kws_conv1_quantize_Div_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _kws_conv1_quantize_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv1_activate_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv1_quantize_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_voice_conv3_quantize_Mul_output_0_scale, &_voice_conv3_quantize_Mul_output_0_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _kws_conv1_quantize_Mul_output_0_layer, 34,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &_kws_conv1_quantize_Mul_output_0_chain,
  NULL, &_kws_conv1_quantize_Round_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _kws_conv1_activate_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv1_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv1_activate_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _kws_conv1_activate_Relu_output_0_layer, 33,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &_kws_conv1_activate_Relu_output_0_chain,
  NULL, &_kws_conv1_quantize_Mul_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _kws_conv1_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv1_pool_MaxPool_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv1_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_kws_conv1_Conv_output_0_weights, &_kws_conv1_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_kws_conv1_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _kws_conv1_Conv_output_0_layer, 32,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_kws_conv1_Conv_output_0_chain,
  NULL, &_kws_conv1_activate_Relu_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 0, 1, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _kws_conv1_pool_MaxPool_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv4_Mul_2_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv1_pool_MaxPool_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _kws_conv1_pool_MaxPool_output_0_layer, 30,
  POOL_TYPE, 0x0, NULL,
  pool, forward_mp,
  &_kws_conv1_pool_MaxPool_output_0_chain,
  NULL, &_kws_conv1_Conv_output_0_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(1, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(1, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _voice_conv4_Mul_2_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv4_quantize_Div_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv4_Mul_2_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_voice_conv4_Mul_2_output_0_scale, &_voice_conv4_quantize_Mul_output_0_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _voice_conv4_Mul_2_output_0_layer, 29,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &_voice_conv4_Mul_2_output_0_chain,
  NULL, &_kws_conv1_pool_MaxPool_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _voice_conv4_quantize_Div_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_voice_conv4_quantize_Round_output_0_output, &_voice_conv4_quantize_Constant_1_output_0_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv4_quantize_Div_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _voice_conv4_quantize_Div_output_0_layer, 28,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_voice_conv4_quantize_Div_output_0_chain,
  NULL, &_voice_conv4_Mul_2_output_0_layer, AI_STATIC, 
  .operation = ai_div_f32, 
  .buffer_operation = ai_div_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _voice_conv4_quantize_Round_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv4_quantize_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv4_quantize_Round_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _voice_conv4_quantize_Round_output_0_layer, 27,
  NL_TYPE, 0x0, NULL,
  nl, forward_round,
  &_voice_conv4_quantize_Round_output_0_chain,
  NULL, &_voice_conv4_quantize_Div_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _voice_conv4_quantize_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv4_activate_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv4_quantize_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_voice_conv4_quantize_Mul_output_0_scale, &_voice_conv4_quantize_Mul_output_0_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _voice_conv4_quantize_Mul_output_0_layer, 26,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &_voice_conv4_quantize_Mul_output_0_chain,
  NULL, &_voice_conv4_quantize_Round_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _voice_conv4_activate_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv4_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv4_activate_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _voice_conv4_activate_Relu_output_0_layer, 25,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &_voice_conv4_activate_Relu_output_0_chain,
  NULL, &_voice_conv4_quantize_Mul_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _voice_conv4_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv3_quantize_Div_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv4_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_voice_conv4_Conv_output_0_weights, &_voice_conv4_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_voice_conv4_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _voice_conv4_Conv_output_0_layer, 24,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_voice_conv4_Conv_output_0_chain,
  NULL, &_voice_conv4_activate_Relu_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _voice_conv3_quantize_Div_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_voice_conv3_quantize_Round_output_0_output, &_voice_conv3_quantize_Constant_1_output_0_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv3_quantize_Div_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _voice_conv3_quantize_Div_output_0_layer, 21,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_voice_conv3_quantize_Div_output_0_chain,
  NULL, &_voice_conv4_Conv_output_0_layer, AI_STATIC, 
  .operation = ai_div_f32, 
  .buffer_operation = ai_div_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _voice_conv3_quantize_Round_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv3_quantize_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv3_quantize_Round_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _voice_conv3_quantize_Round_output_0_layer, 20,
  NL_TYPE, 0x0, NULL,
  nl, forward_round,
  &_voice_conv3_quantize_Round_output_0_chain,
  NULL, &_voice_conv3_quantize_Div_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _voice_conv3_quantize_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv3_activate_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv3_quantize_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_voice_conv3_quantize_Mul_output_0_scale, &_voice_conv3_quantize_Mul_output_0_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _voice_conv3_quantize_Mul_output_0_layer, 19,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &_voice_conv3_quantize_Mul_output_0_chain,
  NULL, &_voice_conv3_quantize_Round_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _voice_conv3_activate_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv3_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv3_activate_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _voice_conv3_activate_Relu_output_0_layer, 18,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &_voice_conv3_activate_Relu_output_0_chain,
  NULL, &_voice_conv3_quantize_Mul_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _voice_conv3_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv3_pool_MaxPool_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv3_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_voice_conv3_Conv_output_0_weights, &_voice_conv3_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_voice_conv3_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _voice_conv3_Conv_output_0_layer, 17,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_voice_conv3_Conv_output_0_chain,
  NULL, &_voice_conv3_activate_Relu_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 0, 1, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _voice_conv3_pool_MaxPool_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv2_Mul_2_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv3_pool_MaxPool_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _voice_conv3_pool_MaxPool_output_0_layer, 15,
  POOL_TYPE, 0x0, NULL,
  pool, forward_mp,
  &_voice_conv3_pool_MaxPool_output_0_chain,
  NULL, &_voice_conv3_Conv_output_0_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(1, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(1, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _voice_conv2_Mul_2_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv2_quantize_Div_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv2_Mul_2_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_voice_conv2_Mul_2_output_0_scale, &_voice_conv2_quantize_Mul_output_0_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _voice_conv2_Mul_2_output_0_layer, 14,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &_voice_conv2_Mul_2_output_0_chain,
  NULL, &_voice_conv3_pool_MaxPool_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _voice_conv2_quantize_Div_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_voice_conv2_quantize_Round_output_0_output, &_voice_conv2_quantize_Constant_1_output_0_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv2_quantize_Div_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _voice_conv2_quantize_Div_output_0_layer, 13,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_voice_conv2_quantize_Div_output_0_chain,
  NULL, &_voice_conv2_Mul_2_output_0_layer, AI_STATIC, 
  .operation = ai_div_f32, 
  .buffer_operation = ai_div_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _voice_conv2_quantize_Round_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv2_quantize_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv2_quantize_Round_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _voice_conv2_quantize_Round_output_0_layer, 12,
  NL_TYPE, 0x0, NULL,
  nl, forward_round,
  &_voice_conv2_quantize_Round_output_0_chain,
  NULL, &_voice_conv2_quantize_Div_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _voice_conv2_quantize_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv2_activate_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv2_quantize_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_voice_conv2_quantize_Mul_output_0_scale, &_voice_conv2_quantize_Mul_output_0_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _voice_conv2_quantize_Mul_output_0_layer, 11,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &_voice_conv2_quantize_Mul_output_0_chain,
  NULL, &_voice_conv2_quantize_Round_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _voice_conv2_activate_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv2_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv2_activate_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _voice_conv2_activate_Relu_output_0_layer, 10,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &_voice_conv2_activate_Relu_output_0_chain,
  NULL, &_voice_conv2_quantize_Mul_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _voice_conv2_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv1_quantize_Div_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv2_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_voice_conv2_Conv_output_0_weights, &_voice_conv2_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_voice_conv2_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _voice_conv2_Conv_output_0_layer, 9,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_voice_conv2_Conv_output_0_chain,
  NULL, &_voice_conv2_activate_Relu_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _voice_conv1_quantize_Div_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_voice_conv1_quantize_Round_output_0_output, &_voice_conv1_quantize_Constant_1_output_0_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv1_quantize_Div_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _voice_conv1_quantize_Div_output_0_layer, 6,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_voice_conv1_quantize_Div_output_0_chain,
  NULL, &_voice_conv2_Conv_output_0_layer, AI_STATIC, 
  .operation = ai_div_f32, 
  .buffer_operation = ai_div_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _voice_conv1_quantize_Round_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv1_quantize_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv1_quantize_Round_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _voice_conv1_quantize_Round_output_0_layer, 5,
  NL_TYPE, 0x0, NULL,
  nl, forward_round,
  &_voice_conv1_quantize_Round_output_0_chain,
  NULL, &_voice_conv1_quantize_Div_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _voice_conv1_quantize_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv1_activate_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv1_quantize_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_voice_conv1_quantize_Mul_output_0_scale, &_voice_conv1_quantize_Mul_output_0_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _voice_conv1_quantize_Mul_output_0_layer, 4,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &_voice_conv1_quantize_Mul_output_0_chain,
  NULL, &_voice_conv1_quantize_Round_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _voice_conv1_activate_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv1_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv1_activate_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _voice_conv1_activate_Relu_output_0_layer, 3,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &_voice_conv1_activate_Relu_output_0_chain,
  NULL, &_voice_conv1_quantize_Mul_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _voice_conv1_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &kws_input_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv1_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_voice_conv1_Conv_output_0_weights, &_voice_conv1_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_voice_conv1_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _voice_conv1_Conv_output_0_layer, 2,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_voice_conv1_Conv_output_0_chain,
  NULL, &_voice_conv1_activate_Relu_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


#if (AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 684404, 1, 1),
    684404, NULL, NULL),
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 66832, 1, 1),
    66832, NULL, NULL),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_IN_NUM, &kws_input_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_OUT_NUM, &kws_logits_output),
  &_voice_conv1_Conv_output_0_layer, 0xd0f1b3ac, NULL)

#else

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 684404, 1, 1),
      684404, NULL, NULL)
  ),
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 66832, 1, 1),
      66832, NULL, NULL)
  ),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_IN_NUM, &kws_input_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_OUT_NUM, &kws_logits_output),
  &_voice_conv1_Conv_output_0_layer, 0xd0f1b3ac, NULL)

#endif	/*(AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)*/



/******************************************************************************/
AI_DECLARE_STATIC
ai_bool network_configure_activations(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_activations_map(g_network_activations_map, 1, params)) {
    /* Updating activations (byte) offsets */
    
    kws_input_output_array.data = AI_PTR(g_network_activations_map[0] + 784);
    kws_input_output_array.data_start = AI_PTR(g_network_activations_map[0] + 784);
    _voice_conv1_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 66320);
    _voice_conv1_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 66320);
    _voice_conv1_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 384);
    _voice_conv1_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 384);
    _voice_conv1_activate_Relu_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 384);
    _voice_conv1_activate_Relu_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 384);
    _voice_conv1_quantize_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 384);
    _voice_conv1_quantize_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 384);
    _voice_conv1_quantize_Round_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 384);
    _voice_conv1_quantize_Round_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 384);
    _voice_conv1_quantize_Div_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 384);
    _voice_conv1_quantize_Div_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 384);
    _voice_conv2_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 51584);
    _voice_conv2_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 51584);
    _voice_conv2_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _voice_conv2_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _voice_conv2_activate_Relu_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _voice_conv2_activate_Relu_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _voice_conv2_quantize_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _voice_conv2_quantize_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _voice_conv2_quantize_Round_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _voice_conv2_quantize_Round_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _voice_conv2_quantize_Div_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _voice_conv2_quantize_Div_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _voice_conv2_Mul_2_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _voice_conv2_Mul_2_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _voice_conv3_pool_MaxPool_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _voice_conv3_pool_MaxPool_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _voice_conv3_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 24192);
    _voice_conv3_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 24192);
    _voice_conv3_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 25344);
    _voice_conv3_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 25344);
    _voice_conv3_activate_Relu_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _voice_conv3_activate_Relu_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _voice_conv3_quantize_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 16128);
    _voice_conv3_quantize_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 16128);
    _voice_conv3_quantize_Round_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _voice_conv3_quantize_Round_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _voice_conv3_quantize_Div_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 16128);
    _voice_conv3_quantize_Div_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 16128);
    _voice_conv4_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _voice_conv4_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _voice_conv4_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 768);
    _voice_conv4_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 768);
    _voice_conv4_activate_Relu_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 12480);
    _voice_conv4_activate_Relu_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 12480);
    _voice_conv4_quantize_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _voice_conv4_quantize_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _voice_conv4_quantize_Round_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 11712);
    _voice_conv4_quantize_Round_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 11712);
    _voice_conv4_quantize_Div_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _voice_conv4_quantize_Div_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _voice_conv4_Mul_2_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 11712);
    _voice_conv4_Mul_2_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 11712);
    _kws_conv1_pool_MaxPool_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _kws_conv1_pool_MaxPool_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _kws_conv1_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 5760);
    _kws_conv1_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 5760);
    _kws_conv1_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 6336);
    _kws_conv1_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 6336);
    _kws_conv1_activate_Relu_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 14016);
    _kws_conv1_activate_Relu_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 14016);
    _kws_conv1_quantize_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _kws_conv1_quantize_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _kws_conv1_quantize_Round_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 7680);
    _kws_conv1_quantize_Round_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 7680);
    _kws_conv1_quantize_Div_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _kws_conv1_quantize_Div_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _kws_conv2_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 7680);
    _kws_conv2_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 7680);
    _kws_conv2_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 8448);
    _kws_conv2_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 8448);
    _kws_conv2_activate_Relu_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 19200);
    _kws_conv2_activate_Relu_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 19200);
    _kws_conv2_quantize_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _kws_conv2_quantize_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _kws_conv2_quantize_Round_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 10752);
    _kws_conv2_quantize_Round_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 10752);
    _kws_conv2_quantize_Div_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _kws_conv2_quantize_Div_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _kws_conv2_Mul_2_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 10752);
    _kws_conv2_Mul_2_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 10752);
    _kws_conv3_pool_AveragePool_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _kws_conv3_pool_AveragePool_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _kws_conv3_quantize_pool_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 5376);
    _kws_conv3_quantize_pool_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 5376);
    _kws_conv3_quantize_pool_Floor_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _kws_conv3_quantize_pool_Floor_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _kws_conv3_quantize_pool_Div_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 5376);
    _kws_conv3_quantize_pool_Div_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 5376);
    _kws_conv3_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _kws_conv3_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _kws_conv3_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 10752);
    _kws_conv3_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 10752);
    _kws_conv3_activate_Relu_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _kws_conv3_activate_Relu_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _kws_conv3_quantize_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 5600);
    _kws_conv3_quantize_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 5600);
    _kws_conv3_quantize_Round_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _kws_conv3_quantize_Round_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _kws_conv3_quantize_Div_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 5600);
    _kws_conv3_quantize_Div_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 5600);
    _kws_conv3_Mul_2_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _kws_conv3_Mul_2_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _kws_conv4_pool_MaxPool_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 5600);
    _kws_conv4_pool_MaxPool_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 5600);
    _kws_conv4_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _kws_conv4_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _kws_conv4_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 2400);
    _kws_conv4_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2400);
    _kws_conv4_activate_Relu_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _kws_conv4_activate_Relu_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _kws_conv4_quantize_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 1024);
    _kws_conv4_quantize_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1024);
    _kws_conv4_quantize_Round_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _kws_conv4_quantize_Round_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _kws_conv4_quantize_Div_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 1024);
    _kws_conv4_quantize_Div_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1024);
    _kws_conv4_Mul_2_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _kws_conv4_Mul_2_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _Reshape_output_0_to_chlast_output_array.data = AI_PTR(g_network_activations_map[0] + 1024);
    _Reshape_output_0_to_chlast_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1024);
    _fc_Gemm_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _fc_Gemm_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _fc_quantize_Round_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 84);
    _fc_quantize_Round_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 84);
    _fc_quantize_Div_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _fc_quantize_Div_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    kws_logits_output_array.data = AI_PTR(g_network_activations_map[0] + 84);
    kws_logits_output_array.data_start = AI_PTR(g_network_activations_map[0] + 84);
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
    
    _voice_conv1_quantize_Constant_1_output_0_4D_array.format |= AI_FMT_FLAG_CONST;
    _voice_conv1_quantize_Constant_1_output_0_4D_array.data = AI_PTR(g_network_weights_map[0] + 0);
    _voice_conv1_quantize_Constant_1_output_0_4D_array.data_start = AI_PTR(g_network_weights_map[0] + 0);
    _voice_conv2_quantize_Constant_1_output_0_4D_array.format |= AI_FMT_FLAG_CONST;
    _voice_conv2_quantize_Constant_1_output_0_4D_array.data = AI_PTR(g_network_weights_map[0] + 4);
    _voice_conv2_quantize_Constant_1_output_0_4D_array.data_start = AI_PTR(g_network_weights_map[0] + 4);
    _voice_conv3_quantize_Constant_1_output_0_4D_array.format |= AI_FMT_FLAG_CONST;
    _voice_conv3_quantize_Constant_1_output_0_4D_array.data = AI_PTR(g_network_weights_map[0] + 8);
    _voice_conv3_quantize_Constant_1_output_0_4D_array.data_start = AI_PTR(g_network_weights_map[0] + 8);
    _voice_conv4_quantize_Constant_1_output_0_4D_array.format |= AI_FMT_FLAG_CONST;
    _voice_conv4_quantize_Constant_1_output_0_4D_array.data = AI_PTR(g_network_weights_map[0] + 12);
    _voice_conv4_quantize_Constant_1_output_0_4D_array.data_start = AI_PTR(g_network_weights_map[0] + 12);
    _kws_conv1_quantize_Constant_1_output_0_4D_array.format |= AI_FMT_FLAG_CONST;
    _kws_conv1_quantize_Constant_1_output_0_4D_array.data = AI_PTR(g_network_weights_map[0] + 16);
    _kws_conv1_quantize_Constant_1_output_0_4D_array.data_start = AI_PTR(g_network_weights_map[0] + 16);
    _kws_conv2_quantize_Constant_1_output_0_4D_array.format |= AI_FMT_FLAG_CONST;
    _kws_conv2_quantize_Constant_1_output_0_4D_array.data = AI_PTR(g_network_weights_map[0] + 20);
    _kws_conv2_quantize_Constant_1_output_0_4D_array.data_start = AI_PTR(g_network_weights_map[0] + 20);
    _kws_conv3_quantize_pool_Constant_1_output_0_4D_array.format |= AI_FMT_FLAG_CONST;
    _kws_conv3_quantize_pool_Constant_1_output_0_4D_array.data = AI_PTR(g_network_weights_map[0] + 24);
    _kws_conv3_quantize_pool_Constant_1_output_0_4D_array.data_start = AI_PTR(g_network_weights_map[0] + 24);
    _kws_conv3_quantize_Constant_1_output_0_4D_array.format |= AI_FMT_FLAG_CONST;
    _kws_conv3_quantize_Constant_1_output_0_4D_array.data = AI_PTR(g_network_weights_map[0] + 28);
    _kws_conv3_quantize_Constant_1_output_0_4D_array.data_start = AI_PTR(g_network_weights_map[0] + 28);
    _kws_conv4_quantize_Constant_1_output_0_4D_array.format |= AI_FMT_FLAG_CONST;
    _kws_conv4_quantize_Constant_1_output_0_4D_array.data = AI_PTR(g_network_weights_map[0] + 32);
    _kws_conv4_quantize_Constant_1_output_0_4D_array.data_start = AI_PTR(g_network_weights_map[0] + 32);
    _fc_quantize_Constant_1_output_0_2D_array.format |= AI_FMT_FLAG_CONST;
    _fc_quantize_Constant_1_output_0_2D_array.data = AI_PTR(g_network_weights_map[0] + 36);
    _fc_quantize_Constant_1_output_0_2D_array.data_start = AI_PTR(g_network_weights_map[0] + 36);
    _voice_conv1_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _voice_conv1_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 40);
    _voice_conv1_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 40);
    _voice_conv1_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _voice_conv1_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 51240);
    _voice_conv1_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 51240);
    _voice_conv1_quantize_Mul_output_0_scale_array.format |= AI_FMT_FLAG_CONST;
    _voice_conv1_quantize_Mul_output_0_scale_array.data = AI_PTR(g_network_weights_map[0] + 51640);
    _voice_conv1_quantize_Mul_output_0_scale_array.data_start = AI_PTR(g_network_weights_map[0] + 51640);
    _voice_conv1_quantize_Mul_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _voice_conv1_quantize_Mul_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 52040);
    _voice_conv1_quantize_Mul_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 52040);
    _voice_conv2_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _voice_conv2_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 52440);
    _voice_conv2_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 52440);
    _voice_conv2_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _voice_conv2_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 167640);
    _voice_conv2_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 167640);
    _voice_conv2_quantize_Mul_output_0_scale_array.format |= AI_FMT_FLAG_CONST;
    _voice_conv2_quantize_Mul_output_0_scale_array.data = AI_PTR(g_network_weights_map[0] + 168024);
    _voice_conv2_quantize_Mul_output_0_scale_array.data_start = AI_PTR(g_network_weights_map[0] + 168024);
    _voice_conv2_quantize_Mul_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _voice_conv2_quantize_Mul_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 168408);
    _voice_conv2_quantize_Mul_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 168408);
    _voice_conv2_Mul_2_output_0_scale_array.format |= AI_FMT_FLAG_CONST;
    _voice_conv2_Mul_2_output_0_scale_array.data = AI_PTR(g_network_weights_map[0] + 168792);
    _voice_conv2_Mul_2_output_0_scale_array.data_start = AI_PTR(g_network_weights_map[0] + 168792);
    _voice_conv3_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _voice_conv3_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 169176);
    _voice_conv3_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 169176);
    _voice_conv3_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _voice_conv3_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 242904);
    _voice_conv3_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 242904);
    _voice_conv3_quantize_Mul_output_0_scale_array.format |= AI_FMT_FLAG_CONST;
    _voice_conv3_quantize_Mul_output_0_scale_array.data = AI_PTR(g_network_weights_map[0] + 243160);
    _voice_conv3_quantize_Mul_output_0_scale_array.data_start = AI_PTR(g_network_weights_map[0] + 243160);
    _voice_conv3_quantize_Mul_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _voice_conv3_quantize_Mul_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 243416);
    _voice_conv3_quantize_Mul_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 243416);
    _voice_conv4_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _voice_conv4_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 243672);
    _voice_conv4_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 243672);
    _voice_conv4_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _voice_conv4_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 280536);
    _voice_conv4_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 280536);
    _voice_conv4_quantize_Mul_output_0_scale_array.format |= AI_FMT_FLAG_CONST;
    _voice_conv4_quantize_Mul_output_0_scale_array.data = AI_PTR(g_network_weights_map[0] + 280728);
    _voice_conv4_quantize_Mul_output_0_scale_array.data_start = AI_PTR(g_network_weights_map[0] + 280728);
    _voice_conv4_quantize_Mul_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _voice_conv4_quantize_Mul_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 280920);
    _voice_conv4_quantize_Mul_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 280920);
    _voice_conv4_Mul_2_output_0_scale_array.format |= AI_FMT_FLAG_CONST;
    _voice_conv4_Mul_2_output_0_scale_array.data = AI_PTR(g_network_weights_map[0] + 281112);
    _voice_conv4_Mul_2_output_0_scale_array.data_start = AI_PTR(g_network_weights_map[0] + 281112);
    _kws_conv1_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _kws_conv1_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 281304);
    _kws_conv1_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 281304);
    _kws_conv1_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _kws_conv1_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 318168);
    _kws_conv1_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 318168);
    _kws_conv2_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _kws_conv2_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 318424);
    _kws_conv2_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 318424);
    _kws_conv2_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _kws_conv2_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 392152);
    _kws_conv2_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 392152);
    _kws_conv3_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _kws_conv3_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 392536);
    _kws_conv3_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 392536);
    _kws_conv3_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _kws_conv3_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 507736);
    _kws_conv3_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 507736);
    _kws_conv3_Mul_2_output_0_scale_array.format |= AI_FMT_FLAG_CONST;
    _kws_conv3_Mul_2_output_0_scale_array.data = AI_PTR(g_network_weights_map[0] + 508136);
    _kws_conv3_Mul_2_output_0_scale_array.data_start = AI_PTR(g_network_weights_map[0] + 508136);
    _kws_conv4_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _kws_conv4_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 508536);
    _kws_conv4_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 508536);
    _kws_conv4_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _kws_conv4_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 662136);
    _kws_conv4_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 662136);
    _kws_conv4_Mul_2_output_0_scale_array.format |= AI_FMT_FLAG_CONST;
    _kws_conv4_Mul_2_output_0_scale_array.data = AI_PTR(g_network_weights_map[0] + 662392);
    _kws_conv4_Mul_2_output_0_scale_array.data_start = AI_PTR(g_network_weights_map[0] + 662392);
    _fc_Gemm_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _fc_Gemm_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 662648);
    _fc_Gemm_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 662648);
    _fc_Gemm_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _fc_Gemm_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 684152);
    _fc_Gemm_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 684152);
    kws_logits_scale_array.format |= AI_FMT_FLAG_CONST;
    kws_logits_scale_array.data = AI_PTR(g_network_weights_map[0] + 684236);
    kws_logits_scale_array.data_start = AI_PTR(g_network_weights_map[0] + 684236);
    kws_logits_bias_array.format |= AI_FMT_FLAG_CONST;
    kws_logits_bias_array.data = AI_PTR(g_network_weights_map[0] + 684320);
    kws_logits_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 684320);
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
      
      .n_macc            = 8797410,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .params            = AI_STRUCT_INIT,
      .activations       = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0xd0f1b3ac,
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
      
      .n_macc            = 8797410,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .map_signature     = AI_MAGIC_SIGNATURE,
      .map_weights       = AI_STRUCT_INIT,
      .map_activations   = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0xd0f1b3ac,
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

