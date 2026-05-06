/**
  ******************************************************************************
  * @file    network.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-05-06T16:10:49+0200
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
#define AI_NETWORK_MODEL_SIGNATURE     "0x9863c81c8bec2f37cec2add657802fef"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     ""
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "2026-05-06T16:10:49+0200"

#undef AI_TOOLS_COMPILE_TIME
#define AI_TOOLS_COMPILE_TIME    __DATE__ " " __TIME__

#undef AI_NETWORK_N_BATCHES
#define AI_NETWORK_N_BATCHES         (1)

static ai_ptr g_network_activations_map[1] = AI_C_ARRAY_INIT;
static ai_ptr g_network_weights_map[1] = AI_C_ARRAY_INIT;



/**  Array declarations section  **********************************************/
/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  kws_input_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 16384, AI_STATIC)

/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv1_Conv_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 12800, AI_STATIC)

/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv1_activate_Relu_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 12800, AI_STATIC)

/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv1_quantize_Mul_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 12800, AI_STATIC)

/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv1_quantize_Mul_output_0_0_0__voice_conv1_quantize_Round_output_0_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12800, AI_STATIC)

/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv1_quantize_Round_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12800, AI_STATIC)

/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv1_quantize_Div_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12800, AI_STATIC)

/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv1_quantize_Div_output_0_0_0__voice_conv1_Mul_2_output_0_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 12800, AI_STATIC)

/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv1_Mul_2_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 12800, AI_STATIC)

/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv2_Conv_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 12096, AI_STATIC)

/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv2_activate_Relu_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 12096, AI_STATIC)

/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv2_quantize_Mul_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 12096, AI_STATIC)

/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv2_quantize_Mul_output_0_0_0__voice_conv2_quantize_Round_output_0_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12096, AI_STATIC)

/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv2_quantize_Round_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12096, AI_STATIC)

/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv2_quantize_Div_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12096, AI_STATIC)

/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv2_quantize_Div_output_0_0_0__voice_conv2_Mul_2_output_0_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 12096, AI_STATIC)

/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv2_Mul_2_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 12096, AI_STATIC)

/* Array#17 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv3_pool_MaxPool_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 6048, AI_STATIC)

/* Array#18 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv3_Conv_output_0_pad_before_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 6240, AI_STATIC)

/* Array#19 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv3_Conv_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4032, AI_STATIC)

/* Array#20 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv3_activate_Relu_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4032, AI_STATIC)

/* Array#21 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv3_quantize_Mul_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4032, AI_STATIC)

/* Array#22 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv3_quantize_Mul_output_0_0_0__voice_conv3_quantize_Round_output_0_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4032, AI_STATIC)

/* Array#23 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv3_quantize_Round_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4032, AI_STATIC)

/* Array#24 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv3_quantize_Div_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4032, AI_STATIC)

/* Array#25 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv3_quantize_Div_output_0_0_0__voice_conv3_Mul_2_output_0_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4032, AI_STATIC)

/* Array#26 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv3_Mul_2_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4032, AI_STATIC)

/* Array#27 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv4_Conv_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2928, AI_STATIC)

/* Array#28 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv4_activate_Relu_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2928, AI_STATIC)

/* Array#29 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv4_quantize_Mul_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2928, AI_STATIC)

/* Array#30 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv4_quantize_Mul_output_0_0_0__voice_conv4_quantize_Round_output_0_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2928, AI_STATIC)

/* Array#31 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv4_quantize_Round_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2928, AI_STATIC)

/* Array#32 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv4_quantize_Div_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2928, AI_STATIC)

/* Array#33 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv4_quantize_Div_output_0_0_0__voice_conv4_Mul_2_output_0_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2928, AI_STATIC)

/* Array#34 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv4_Mul_2_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2928, AI_STATIC)

/* Array#35 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv1_pool_MaxPool_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1440, AI_STATIC)

/* Array#36 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv1_Conv_output_0_pad_before_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1536, AI_STATIC)

/* Array#37 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv1_Conv_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1920, AI_STATIC)

/* Array#38 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv1_activate_Relu_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1920, AI_STATIC)

/* Array#39 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv1_quantize_Mul_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1920, AI_STATIC)

/* Array#40 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv1_quantize_Mul_output_0_0_0__kws_conv1_quantize_Round_output_0_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1920, AI_STATIC)

/* Array#41 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv1_quantize_Round_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1920, AI_STATIC)

/* Array#42 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv1_quantize_Div_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1920, AI_STATIC)

/* Array#43 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv1_quantize_Div_output_0_0_0__kws_conv1_Mul_2_output_0_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1920, AI_STATIC)

/* Array#44 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv1_Mul_2_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1920, AI_STATIC)

/* Array#45 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv2_Conv_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2688, AI_STATIC)

/* Array#46 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv2_activate_Relu_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2688, AI_STATIC)

/* Array#47 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv2_quantize_Mul_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2688, AI_STATIC)

/* Array#48 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv2_quantize_Mul_output_0_0_0__kws_conv2_quantize_Round_output_0_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2688, AI_STATIC)

/* Array#49 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv2_quantize_Round_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2688, AI_STATIC)

/* Array#50 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv2_quantize_Div_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2688, AI_STATIC)

/* Array#51 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv2_quantize_Div_output_0_0_0__kws_conv2_Mul_2_output_0_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2688, AI_STATIC)

/* Array#52 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv2_Mul_2_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2688, AI_STATIC)

/* Array#53 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv3_pool_AveragePool_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1344, AI_STATIC)

/* Array#54 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv3_quantize_pool_Mul_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1344, AI_STATIC)

/* Array#55 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv3_quantize_pool_Mul_output_0_0_0__kws_conv3_quantize_pool_Floor_output_0_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1344, AI_STATIC)

/* Array#56 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv3_quantize_pool_Floor_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1344, AI_STATIC)

/* Array#57 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv3_quantize_pool_Div_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1344, AI_STATIC)

/* Array#58 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv3_quantize_pool_Div_output_0_0_0__kws_conv3_Conv_output_0_pad_before_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1344, AI_STATIC)

/* Array#59 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv3_Conv_output_0_pad_before_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1536, AI_STATIC)

/* Array#60 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv3_Conv_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1400, AI_STATIC)

/* Array#61 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv3_activate_Relu_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1400, AI_STATIC)

/* Array#62 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv3_quantize_Mul_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1400, AI_STATIC)

/* Array#63 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv3_quantize_Mul_output_0_0_0__kws_conv3_quantize_Round_output_0_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1400, AI_STATIC)

/* Array#64 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv3_quantize_Round_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1400, AI_STATIC)

/* Array#65 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv3_quantize_Div_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1400, AI_STATIC)

/* Array#66 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv3_quantize_Div_output_0_0_0__kws_conv3_Mul_2_output_0_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1400, AI_STATIC)

/* Array#67 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv3_Mul_2_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1400, AI_STATIC)

/* Array#68 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv4_pool_MaxPool_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 700, AI_STATIC)

/* Array#69 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv4_Conv_output_0_pad_before_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 900, AI_STATIC)

/* Array#70 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv4_Conv_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#71 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv4_activate_Relu_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#72 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv4_quantize_Mul_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#73 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv4_quantize_Mul_output_0_0_0__kws_conv4_quantize_Round_output_0_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 256, AI_STATIC)

/* Array#74 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv4_quantize_Round_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 256, AI_STATIC)

/* Array#75 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv4_quantize_Div_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 256, AI_STATIC)

/* Array#76 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv4_quantize_Div_output_0_0_0__kws_conv4_Mul_2_output_0_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#77 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv4_Mul_2_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#78 */
AI_ARRAY_OBJ_DECLARE(
  _Reshape_output_0_to_chlast_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256, AI_STATIC)

/* Array#79 */
AI_ARRAY_OBJ_DECLARE(
  _fc_Gemm_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 21, AI_STATIC)

/* Array#80 */
AI_ARRAY_OBJ_DECLARE(
  _fc_quantize_Mul_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 21, AI_STATIC)

/* Array#81 */
AI_ARRAY_OBJ_DECLARE(
  _fc_quantize_Mul_output_0_0_0__fc_quantize_Round_output_0_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 21, AI_STATIC)

/* Array#82 */
AI_ARRAY_OBJ_DECLARE(
  _fc_quantize_Round_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 21, AI_STATIC)

/* Array#83 */
AI_ARRAY_OBJ_DECLARE(
  _fc_quantize_Div_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 21, AI_STATIC)

/* Array#84 */
AI_ARRAY_OBJ_DECLARE(
  _fc_quantize_Div_output_0_0_0_kws_logits_QuantizeLinear_Input_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 21, AI_STATIC)

/* Array#85 */
AI_ARRAY_OBJ_DECLARE(
  kws_logits_QuantizeLinear_Input_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 21, AI_STATIC)

/* Array#86 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv1_quantize_Constant_1_output_0_4D_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#87 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv2_quantize_Constant_1_output_0_4D_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#88 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv3_quantize_Constant_1_output_0_4D_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#89 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv4_quantize_Constant_1_output_0_4D_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#90 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv1_quantize_Constant_1_output_0_4D_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#91 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv2_quantize_Constant_1_output_0_4D_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#92 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv3_quantize_pool_Constant_1_output_0_4D_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#93 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv3_quantize_Constant_1_output_0_4D_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#94 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv4_quantize_Constant_1_output_0_4D_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#95 */
AI_ARRAY_OBJ_DECLARE(
  _fc_quantize_Constant_1_output_0_2D_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#96 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv4_quantize_Constant_output_0_DequantizeLinear_Output_const_4D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1, AI_STATIC)

/* Array#97 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv4_calc_out_scale_Pow_output_0_DequantizeLinear_Output_const_4D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1, AI_STATIC)

/* Array#98 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv4_Pow_output_0_DequantizeLinear_Output_const_4D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1, AI_STATIC)

/* Array#99 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv3_quantize_Constant_output_0_DequantizeLinear_Output_const_4D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1, AI_STATIC)

/* Array#100 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv3_calc_out_scale_Pow_output_0_DequantizeLinear_Output_const_4D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1, AI_STATIC)

/* Array#101 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv3_Pow_output_0_DequantizeLinear_Output_const_4D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1, AI_STATIC)

/* Array#102 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv2_quantize_Constant_output_0_DequantizeLinear_Output_const_4D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1, AI_STATIC)

/* Array#103 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv2_calc_out_scale_Pow_output_0_DequantizeLinear_Output_const_4D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1, AI_STATIC)

/* Array#104 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv2_Pow_output_0_DequantizeLinear_Output_const_4D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1, AI_STATIC)

/* Array#105 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv1_quantize_Constant_output_0_DequantizeLinear_Output_const_4D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1, AI_STATIC)

/* Array#106 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv1_calc_out_scale_Pow_output_0_DequantizeLinear_Output_const_4D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1, AI_STATIC)

/* Array#107 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv1_Pow_output_0_DequantizeLinear_Output_const_4D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1, AI_STATIC)

/* Array#108 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv4_quantize_Constant_output_0_DequantizeLinear_Output_const_4D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1, AI_STATIC)

/* Array#109 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv4_calc_out_scale_Pow_output_0_DequantizeLinear_Output_const_4D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1, AI_STATIC)

/* Array#110 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv4_Pow_output_0_DequantizeLinear_Output_const_4D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1, AI_STATIC)

/* Array#111 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv3_quantize_pool_Constant_output_0_DequantizeLinear_Output_const_4D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1, AI_STATIC)

/* Array#112 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv3_quantize_Constant_output_0_DequantizeLinear_Output_const_4D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1, AI_STATIC)

/* Array#113 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv3_calc_out_scale_Pow_output_0_DequantizeLinear_Output_const_4D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1, AI_STATIC)

/* Array#114 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv3_Pow_output_0_DequantizeLinear_Output_const_4D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1, AI_STATIC)

/* Array#115 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv2_quantize_Constant_output_0_DequantizeLinear_Output_const_4D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1, AI_STATIC)

/* Array#116 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv2_calc_out_scale_Pow_output_0_DequantizeLinear_Output_const_4D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1, AI_STATIC)

/* Array#117 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv2_Pow_output_0_DequantizeLinear_Output_const_4D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1, AI_STATIC)

/* Array#118 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv1_quantize_Constant_output_0_DequantizeLinear_Output_const_4D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1, AI_STATIC)

/* Array#119 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv1_calc_out_scale_Pow_output_0_DequantizeLinear_Output_const_4D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1, AI_STATIC)

/* Array#120 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv1_Pow_output_0_DequantizeLinear_Output_const_4D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1, AI_STATIC)

/* Array#121 */
AI_ARRAY_OBJ_DECLARE(
  _fc_quantize_Constant_output_0_DequantizeLinear_Output_const_2D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1, AI_STATIC)

/* Array#122 */
AI_ARRAY_OBJ_DECLARE(
  _fc_Pow_output_0_DequantizeLinear_Output_const_2D_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1, AI_STATIC)

/* Array#123 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv1_Conv_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 12800, AI_STATIC)

/* Array#124 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv1_Conv_output_0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 100, AI_STATIC)

/* Array#125 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv2_Conv_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 28800, AI_STATIC)

/* Array#126 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv2_Conv_output_0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 96, AI_STATIC)

/* Array#127 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv3_Conv_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 18432, AI_STATIC)

/* Array#128 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv3_Conv_output_0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 64, AI_STATIC)

/* Array#129 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv4_Conv_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9216, AI_STATIC)

/* Array#130 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv4_Conv_output_0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 48, AI_STATIC)

/* Array#131 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv1_Conv_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9216, AI_STATIC)

/* Array#132 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv1_Conv_output_0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 64, AI_STATIC)

/* Array#133 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv2_Conv_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 18432, AI_STATIC)

/* Array#134 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv2_Conv_output_0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 96, AI_STATIC)

/* Array#135 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv3_Conv_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 28800, AI_STATIC)

/* Array#136 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv3_Conv_output_0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 100, AI_STATIC)

/* Array#137 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv4_Conv_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 38400, AI_STATIC)

/* Array#138 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv4_Conv_output_0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 64, AI_STATIC)

/* Array#139 */
AI_ARRAY_OBJ_DECLARE(
  _fc_Gemm_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 5376, AI_STATIC)

/* Array#140 */
AI_ARRAY_OBJ_DECLARE(
  _fc_Gemm_output_0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 21, AI_STATIC)

/* Array#141 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv1_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 7032, AI_STATIC)

/* Array#142 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv2_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 7664, AI_STATIC)

/* Array#143 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv3_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 7168, AI_STATIC)

/* Array#144 */
AI_ARRAY_OBJ_DECLARE(
  _voice_conv4_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 6560, AI_STATIC)

/* Array#145 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv1_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 6592, AI_STATIC)

/* Array#146 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv2_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 7232, AI_STATIC)

/* Array#147 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv3_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 7672, AI_STATIC)

/* Array#148 */
AI_ARRAY_OBJ_DECLARE(
  _kws_conv4_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8416, AI_STATIC)

/* Array#149 */
AI_ARRAY_OBJ_DECLARE(
  _fc_Gemm_output_0_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 361, AI_STATIC)

/**  Array metadata declarations section  *************************************/
/* Int quant #0 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_Reshape_output_0_to_chlast_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(1.8253064155578613f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #1 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_fc_Gemm_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(9.46987247467041f),
    AI_PACK_INTQ_ZP(26)))

/* Int quant #2 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_fc_Gemm_output_0_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 21,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.003383366158232093f, 0.005044291261583567f, 0.004060039296746254f, 0.0054133860394358635f, 0.005044291261583567f, 0.0038139764219522476f, 0.004244586452841759f, 0.004490649793297052f, 0.004859744105488062f, 0.00479822838678956f, 0.005967027507722378f, 0.004552165511995554f, 0.005228838417679071f, 0.004675196949392557f, 0.004552165511995554f, 0.004736712668091059f, 0.004183070734143257f, 0.004552165511995554f, 0.0041215550154447556f, 0.006582185160368681f, 0.0049212598241865635f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #3 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_fc_Pow_output_0_DequantizeLinear_Output_const_2D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.003921568859368563f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #4 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_fc_quantize_Constant_output_0_DequantizeLinear_Output_const_2D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(64.25098419189453f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #5 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_fc_quantize_Div_output_0_0_0_kws_logits_QuantizeLinear_Input_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(9.46987247467041f),
    AI_PACK_INTQ_ZP(26)))

/* Int quant #6 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_fc_quantize_Mul_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(155154.390625f),
    AI_PACK_INTQ_ZP(26)))

/* Int quant #7 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_kws_conv1_Conv_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.7651934623718262f),
    AI_PACK_INTQ_ZP(38)))

/* Int quant #8 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_kws_conv1_Conv_output_0_pad_before_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.24843749403953552f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #9 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_kws_conv1_Conv_output_0_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 64,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.006336121819913387f, 0.004060039296746254f, 0.0020915353670716286f, 0.007074310909956694f, 0.00479822838678956f, 0.0038139764219522476f, 0.0017839566571637988f, 0.0061515746638178825f, 0.0030142716132104397f, 0.002276082755997777f, 0.003875492140650749f, 0.002276082755997777f, 0.003998523578047752f, 0.004060039296746254f, 0.0022145670372992754f, 0.0024606299120932817f, 0.0030757873319089413f, 0.005597933195531368f, 0.0041215550154447556f, 0.0025221456307917833f, 0.0032603347208350897f, 0.003875492140650749f, 0.004060039296746254f, 0.006274606101214886f, 0.004490649793297052f, 0.004367617890238762f, 0.0026451770681887865f, 0.005782480351626873f, 0.007566437125205994f, 0.005967027507722378f, 0.0036909449845552444f, 0.003875492140650749f, 0.00479822838678956f, 0.005290354136377573f, 0.0030142716132104397f, 0.003998523578047752f, 0.0027066930197179317f, 0.0038139764219522476f, 0.0033218504395335913f, 0.005967027507722378f, 0.0036909449845552444f, 0.004367617890238762f, 0.005351869855076075f, 0.0034448818769305944f, 0.006213090382516384f, 0.004736712668091059f, 0.0041215550154447556f, 0.003937007859349251f, 0.007074310909956694f, 0.003198819002136588f, 0.002952755894511938f, 0.00479822838678956f, 0.0027682087384164333f, 0.002583661349490285f, 0.0069512794725596905f, 0.0030142716132104397f, 0.0018454724922776222f, 0.004060039296746254f, 0.0035679133143275976f, 0.005228838417679071f, 0.005536417476832867f, 0.005351869855076075f, 0.006459153722971678f, 0.0054133860394358635f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #10 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_kws_conv1_Mul_2_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.5369485020637512f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #11 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_kws_conv1_Pow_output_0_DequantizeLinear_Output_const_4D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.003921568859368563f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #12 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_kws_conv1_activate_Relu_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.5369356274604797f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #13 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_kws_conv1_calc_out_scale_Pow_output_0_DequantizeLinear_Output_const_4D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #14 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_kws_conv1_pool_MaxPool_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.24843749403953552f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #15 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_kws_conv1_quantize_Constant_output_0_DequantizeLinear_Output_const_4D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.501960813999176f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #16 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_kws_conv1_quantize_Div_output_0_0_0__kws_conv1_Mul_2_output_0_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.5369485020637512f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #17 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_kws_conv1_quantize_Mul_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(68.7277603149414f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #18 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_kws_conv2_Conv_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.6333828568458557f),
    AI_PACK_INTQ_ZP(34)))

/* Int quant #19 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_kws_conv2_Conv_output_0_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 96,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0017839566571637988f, 0.0017839566571637988f, 0.0020915353670716286f, 0.0022145670372992754f, 0.0030142716132104397f, 0.001599409501068294f, 0.0020915353670716286f, 0.0028912401758134365f, 0.0020915353670716286f, 0.0019685039296746254f, 0.00239911419339478f, 0.00239911419339478f, 0.0013533465098589659f, 0.003137303050607443f, 0.0018454724922776222f, 0.0026451770681887865f, 0.0015378936659544706f, 0.002952755894511938f, 0.004183070734143257f, 0.00239911419339478f, 0.0015378936659544706f, 0.0028912401758134365f, 0.001476377947255969f, 0.0022145670372992754f, 0.0019685039296746254f, 0.001599409501068294f, 0.002276082755997777f, 0.0027066930197179317f, 0.0026451770681887865f, 0.0027682087384164333f, 0.00215305108577013f, 0.0025221456307917833f, 0.00215305108577013f, 0.0018454724922776222f, 0.002030019648373127f, 0.002583661349490285f, 0.0016609252197667956f, 0.0027682087384164333f, 0.0019069882109761238f, 0.001599409501068294f, 0.00239911419339478f, 0.0017224409384652972f, 0.00239911419339478f, 0.00215305108577013f, 0.0030757873319089413f, 0.0027682087384164333f, 0.0017224409384652972f, 0.0028912401758134365f, 0.002952755894511938f, 0.003137303050607443f, 0.0028912401758134365f, 0.0026451770681887865f, 0.002583661349490285f, 0.0025221456307917833f, 0.0017224409384652972f, 0.0016609252197667956f, 0.001599409501068294f, 0.0027682087384164333f, 0.002030019648373127f, 0.0019685039296746254f, 0.0022145670372992754f, 0.0017224409384652972f, 0.0023375984746962786f, 0.001599409501068294f, 0.0019069882109761238f, 0.0019069882109761238f, 0.0022145670372992754f, 0.002276082755997777f, 0.0025221456307917833f, 0.0026451770681887865f, 0.0017224409384652972f, 0.002276082755997777f, 0.0015378936659544706f, 0.0022145670372992754f, 0.0022145670372992754f, 0.0018454724922776222f, 0.0017224409384652972f, 0.0024606299120932817f, 0.0027066930197179317f, 0.0023375984746962786f, 0.00239911419339478f, 0.001599409501068294f, 0.0022145670372992754f, 0.0022145670372992754f, 0.0018454724922776222f, 0.001476377947255969f, 0.0030142716132104397f, 0.002030019648373127f, 0.0019685039296746254f, 0.0023375984746962786f, 0.0018454724922776222f, 0.003383366158232093f, 0.0017839566571637988f, 0.0019685039296746254f, 0.0020915353670716286f, 0.0005536417593248188f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #20 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_kws_conv2_Mul_2_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.9218137264251709f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #21 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_kws_conv2_Pow_output_0_DequantizeLinear_Output_const_4D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.003921568859368563f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #22 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_kws_conv2_activate_Relu_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.9218031764030457f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #23 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_kws_conv2_calc_out_scale_Pow_output_0_DequantizeLinear_Output_const_4D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.01568627543747425f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #24 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_kws_conv2_quantize_Constant_output_0_DequantizeLinear_Output_const_4D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.501960813999176f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #25 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_kws_conv2_quantize_Div_output_0_0_0__kws_conv2_Mul_2_output_0_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.9218137264251709f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #26 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_kws_conv2_quantize_Mul_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(117.99080657958984f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #27 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_kws_conv3_Conv_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.8656094670295715f),
    AI_PACK_INTQ_ZP(67)))

/* Int quant #28 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_kws_conv3_Conv_output_0_pad_before_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.8063418865203857f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #29 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_kws_conv3_Conv_output_0_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 100,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0020915353670716286f, 0.0025221456307917833f, 0.0033218504395335913f, 0.0023375984746962786f, 0.0035679133143275976f, 0.0019069882109761238f, 0.0024606299120932817f, 0.0024606299120932817f, 0.0023375984746962786f, 0.002952755894511938f, 0.0017224409384652972f, 0.003629429033026099f, 0.00215305108577013f, 0.0023375984746962786f, 0.0041215550154447556f, 0.002583661349490285f, 0.0022145670372992754f, 0.0015378936659544706f, 0.0022145670372992754f, 0.0024606299120932817f, 0.003137303050607443f, 0.002829724457114935f, 0.0026451770681887865f, 0.0034448818769305944f, 0.0025221456307917833f, 0.0017839566571637988f, 0.00215305108577013f, 0.0028912401758134365f, 0.0028912401758134365f, 0.005597933195531368f, 0.002829724457114935f, 0.0030757873319089413f, 0.0019685039296746254f, 0.002030019648373127f, 0.0024606299120932817f, 0.00239911419339478f, 0.0028912401758134365f, 0.003383366158232093f, 0.002030019648373127f, 0.0038139764219522476f, 0.0032603347208350897f, 0.0025221456307917833f, 0.002829724457114935f, 0.0017224409384652972f, 0.0012918306747451425f, 0.0020915353670716286f, 0.0030757873319089413f, 0.0023375984746962786f, 0.00215305108577013f, 0.0033218504395335913f, 0.0032603347208350897f, 0.0026451770681887865f, 0.0025221456307917833f, 0.0027066930197179317f, 0.0027066930197179317f, 0.0028912401758134365f, 0.002829724457114935f, 0.003629429033026099f, 0.002276082755997777f, 0.0027682087384164333f, 0.0027682087384164333f, 0.003506397595629096f, 0.0027682087384164333f, 0.0023375984746962786f, 0.00239911419339478f, 0.003506397595629096f, 0.0038139764219522476f, 0.003506397595629096f, 0.0023375984746962786f, 0.003875492140650749f, 0.0027066930197179317f, 0.00215305108577013f, 0.00239911419339478f, 0.002583661349490285f, 0.0025221456307917833f, 0.0025221456307917833f, 0.0019069882109761238f, 0.0026451770681887865f, 0.003506397595629096f, 0.003383366158232093f, 0.0025221456307917833f, 0.003137303050607443f, 0.0019685039296746254f, 0.002952755894511938f, 0.004244586452841759f, 0.0022145670372992754f, 0.0019685039296746254f, 0.0020915353670716286f, 0.0036909449845552444f, 0.004367617890238762f, 0.0023375984746962786f, 0.0027682087384164333f, 0.0020915353670716286f, 0.0030142716132104397f, 0.002952755894511938f, 0.0018454724922776222f, 0.002276082755997777f, 0.0025221456307917833f, 0.0024606299120932817f, 0.0027682087384164333f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #30 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_kws_conv3_Mul_2_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.8172487616539001f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #31 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_kws_conv3_Pow_output_0_DequantizeLinear_Output_const_4D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.003921568859368563f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #32 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_kws_conv3_activate_Relu_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.8172363042831421f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #33 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_kws_conv3_calc_out_scale_Pow_output_0_DequantizeLinear_Output_const_4D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.01568627543747425f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #34 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_kws_conv3_pool_AveragePool_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.9218137264251709f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #35 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_kws_conv3_quantize_Constant_output_0_DequantizeLinear_Output_const_4D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.501960813999176f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #36 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_kws_conv3_quantize_Div_output_0_0_0__kws_conv3_Mul_2_output_0_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.8172487616539001f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #37 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_kws_conv3_quantize_Mul_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(104.60624694824219f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #38 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_kws_conv3_quantize_pool_Constant_output_0_DequantizeLinear_Output_const_4D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.501960813999176f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #39 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_kws_conv3_quantize_pool_Div_output_0_0_0__kws_conv3_Conv_output_0_pad_before_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.8063418865203857f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #40 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_kws_conv3_quantize_pool_Mul_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(103.2137222290039f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #41 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_kws_conv4_Conv_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.9560812711715698f),
    AI_PACK_INTQ_ZP(5)))

/* Int quant #42 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_kws_conv4_Conv_output_0_pad_before_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.8172487616539001f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #43 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_kws_conv4_Conv_output_0_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 64,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.004982775542885065f, 0.0025221456307917833f, 0.0020915353670716286f, 0.003752460703253746f, 0.003383366158232093f, 0.004490649793297052f, 0.0028912401758134365f, 0.0027066930197179317f, 0.0032603347208350897f, 0.0030757873319089413f, 0.003383366158232093f, 0.005905511789023876f, 0.0030142716132104397f, 0.0027066930197179317f, 0.0034448818769305944f, 0.0033218504395335913f, 0.0033218504395335913f, 0.006397638004273176f, 0.0030142716132104397f, 0.003198819002136588f, 0.005782480351626873f, 0.0033218504395335913f, 0.0030757873319089413f, 0.002952755894511938f, 0.0038139764219522476f, 0.0034448818769305944f, 0.006274606101214886f, 0.006028543226420879f, 0.0027682087384164333f, 0.004060039296746254f, 0.002583661349490285f, 0.002952755894511938f, 0.0035679133143275976f, 0.002952755894511938f, 0.0033218504395335913f, 0.002583661349490285f, 0.003137303050607443f, 0.0030142716132104397f, 0.0027066930197179317f, 0.0032603347208350897f, 0.003198819002136588f, 0.0049212598241865635f, 0.004060039296746254f, 0.004183070734143257f, 0.004982775542885065f, 0.002583661349490285f, 0.0046136812306940556f, 0.0032603347208350897f, 0.0035679133143275976f, 0.003752460703253746f, 0.0027066930197179317f, 0.0030142716132104397f, 0.0038139764219522476f, 0.004490649793297052f, 0.003506397595629096f, 0.002952755894511938f, 0.004982775542885065f, 0.003629429033026099f, 0.004859744105488062f, 0.0027066930197179317f, 0.0003075787390116602f, 0.0032603347208350897f, 0.003629429033026099f, 0.003383366158232093f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #44 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_kws_conv4_Mul_2_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(1.8253064155578613f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #45 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_kws_conv4_Pow_output_0_DequantizeLinear_Output_const_4D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.003921568859368563f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #46 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_kws_conv4_activate_Relu_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(1.8253140449523926f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #47 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_kws_conv4_calc_out_scale_Pow_output_0_DequantizeLinear_Output_const_4D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.01568627543747425f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #48 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_kws_conv4_pool_MaxPool_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.8172487616539001f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #49 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_kws_conv4_quantize_Constant_output_0_DequantizeLinear_Output_const_4D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.501960813999176f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #50 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_kws_conv4_quantize_Div_output_0_0_0__kws_conv4_Mul_2_output_0_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(1.8253064155578613f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #51 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_kws_conv4_quantize_Mul_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(233.64019775390625f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #52 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_voice_conv1_Conv_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05821796506643295f),
    AI_PACK_INTQ_ZP(7)))

/* Int quant #53 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_voice_conv1_Conv_output_0_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 100,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0030757873319089413f, 0.0013533465098589659f, 0.0019685039296746254f, 0.0012303149560466409f, 0.0016609252197667956f, 0.0026451770681887865f, 0.0009227362461388111f, 0.0013533465098589659f, 0.0025221456307917833f, 0.002829724457114935f, 0.0009227362461388111f, 0.0009842519648373127f, 0.0006766732549294829f, 0.0025221456307917833f, 0.0028912401758134365f, 0.0011072835186496377f, 0.004060039296746254f, 0.0020915353670716286f, 0.0012918306747451425f, 0.0032603347208350897f, 0.0033218504395335913f, 0.0011687992373481393f, 0.0009842519648373127f, 0.0010457676835358143f, 0.003752460703253746f, 0.0006151574780233204f, 0.0017224409384652972f, 0.002952755894511938f, 0.0012918306747451425f, 0.0008612204692326486f, 0.0030757873319089413f, 0.0010457676835358143f, 0.0027682087384164333f, 0.0019685039296746254f, 0.0013533465098589659f, 0.0022145670372992754f, 0.003752460703253746f, 0.0019069882109761238f, 0.004675196949392557f, 0.0019069882109761238f, 0.003137303050607443f, 0.0030142716132104397f, 0.000799704750534147f, 0.0010457676835358143f, 0.003506397595629096f, 0.0010457676835358143f, 0.0015378936659544706f, 0.0008612204692326486f, 0.0018454724922776222f, 0.0027066930197179317f, 0.0018454724922776222f, 0.0017839566571637988f, 0.00215305108577013f, 0.0012303149560466409f, 0.0035679133143275976f, 0.003506397595629096f, 0.0009842519648373127f, 0.0009842519648373127f, 0.0014148622285574675f, 0.001476377947255969f, 0.0011687992373481393f, 0.0009842519648373127f, 0.003137303050607443f, 0.0016609252197667956f, 0.0008612204692326486f, 0.0032603347208350897f, 0.003198819002136588f, 0.003198819002136588f, 0.0016609252197667956f, 0.003937007859349251f, 0.0010457676835358143f, 0.0030142716132104397f, 0.002829724457114935f, 0.0011687992373481393f, 0.0012303149560466409f, 0.00215305108577013f, 0.00239911419339478f, 0.0019069882109761238f, 0.0022145670372992754f, 0.0008612204692326486f, 0.000799704750534147f, 0.0012303149560466409f, 0.000799704750534147f, 0.0028912401758134365f, 0.0014148622285574675f, 0.0032603347208350897f, 0.0030142716132104397f, 0.0027066930197179317f, 0.0027066930197179317f, 0.0009227362461388111f, 0.0030757873319089413f, 0.0010457676835358143f, 0.0030142716132104397f, 0.0006766732549294829f, 0.0030757873319089413f, 0.0009227362461388111f, 0.0006766732549294829f, 0.003198819002136588f, 0.002829724457114935f, 0.0011072835186496377f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #54 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_voice_conv1_Mul_2_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05459558963775635f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #55 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_voice_conv1_Pow_output_0_DequantizeLinear_Output_const_4D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.003921568859368563f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #56 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_voice_conv1_activate_Relu_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05458793044090271f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #57 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_voice_conv1_calc_out_scale_Pow_output_0_DequantizeLinear_Output_const_4D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007843137718737125f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #58 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_voice_conv1_quantize_Constant_output_0_DequantizeLinear_Output_const_4D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.501960813999176f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #59 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_voice_conv1_quantize_Div_output_0_0_0__voice_conv1_Mul_2_output_0_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05459558963775635f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #60 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_voice_conv1_quantize_Mul_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(6.987255096435547f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #61 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_voice_conv2_Conv_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.11856234818696976f),
    AI_PACK_INTQ_ZP(57)))

/* Int quant #62 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_voice_conv2_Conv_output_0_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 96,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0017224409384652972f, 0.0030142716132104397f, 0.0012918306747451425f, 0.002030019648373127f, 0.0022145670372992754f, 0.0022145670372992754f, 0.0012918306747451425f, 0.00215305108577013f, 0.001476377947255969f, 0.003506397595629096f, 0.0012918306747451425f, 0.0027682087384164333f, 0.0008612204692326486f, 0.001476377947255969f, 0.0024606299120932817f, 0.0041215550154447556f, 0.0011072835186496377f, 0.003629429033026099f, 0.001476377947255969f, 0.0019069882109761238f, 0.0023375984746962786f, 0.00215305108577013f, 0.0009842519648373127f, 0.0019069882109761238f, 0.0027682087384164333f, 0.0016609252197667956f, 0.0017839566571637988f, 0.0014148622285574675f, 0.002276082755997777f, 0.0027682087384164333f, 0.0018454724922776222f, 0.00239911419339478f, 0.0009842519648373127f, 0.0012303149560466409f, 0.0019685039296746254f, 0.001599409501068294f, 0.002583661349490285f, 0.003198819002136588f, 0.0026451770681887865f, 0.0020915353670716286f, 0.0027682087384164333f, 0.0009842519648373127f, 0.0023375984746962786f, 0.0011687992373481393f, 0.0018454724922776222f, 0.0006151574780233204f, 0.0022145670372992754f, 0.0013533465098589659f, 0.0017224409384652972f, 0.0015378936659544706f, 0.0017224409384652972f, 0.0019069882109761238f, 0.0018454724922776222f, 0.0012303149560466409f, 0.0019685039296746254f, 0.001599409501068294f, 0.0027682087384164333f, 0.001476377947255969f, 0.0011072835186496377f, 0.001599409501068294f, 0.0012918306747451425f, 0.002030019648373127f, 0.001476377947255969f, 0.001599409501068294f, 0.001599409501068294f, 0.00239911419339478f, 0.0019069882109761238f, 0.0022145670372992754f, 0.0015378936659544706f, 0.0012303149560466409f, 0.0020915353670716286f, 0.0017224409384652972f, 0.0016609252197667956f, 0.001599409501068294f, 0.0023375984746962786f, 0.0020915353670716286f, 0.0027066930197179317f, 0.0026451770681887865f, 0.0017224409384652972f, 0.0022145670372992754f, 0.0012303149560466409f, 0.0012918306747451425f, 0.00239911419339478f, 0.0011687992373481393f, 0.002276082755997777f, 0.001599409501068294f, 0.003198819002136588f, 0.0019069882109761238f, 0.0020915353670716286f, 0.0017224409384652972f, 0.0011687992373481393f, 0.001476377947255969f, 0.002030019648373127f, 0.001599409501068294f, 0.0017224409384652972f, 0.0011072835186496377f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #63 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_voice_conv2_Mul_2_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.1299632340669632f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #64 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_voice_conv2_Pow_output_0_DequantizeLinear_Output_const_4D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.003921568859368563f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #65 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_voice_conv2_activate_Relu_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.12996035814285278f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #66 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_voice_conv2_calc_out_scale_Pow_output_0_DequantizeLinear_Output_const_4D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.01568627543747425f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #67 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_voice_conv2_quantize_Constant_output_0_DequantizeLinear_Output_const_4D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.501960813999176f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #68 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_voice_conv2_quantize_Div_output_0_0_0__voice_conv2_Mul_2_output_0_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.1299632340669632f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #69 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_voice_conv2_quantize_Mul_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(16.634925842285156f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #70 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_voice_conv3_Conv_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.3473328948020935f),
    AI_PACK_INTQ_ZP(33)))

/* Int quant #71 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_voice_conv3_Conv_output_0_pad_before_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.1299632340669632f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #72 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_voice_conv3_Conv_output_0_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 64,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0022145670372992754f, 0.003629429033026099f, 0.0019685039296746254f, 0.0028912401758134365f, 0.0024606299120932817f, 0.003198819002136588f, 0.00239911419339478f, 0.0016609252197667956f, 0.0035679133143275976f, 0.001599409501068294f, 0.0030142716132104397f, 0.0019685039296746254f, 0.00215305108577013f, 0.0014148622285574675f, 0.0018454724922776222f, 0.00215305108577013f, 0.0033218504395335913f, 0.0034448818769305944f, 0.0020915353670716286f, 0.0017224409384652972f, 0.0030142716132104397f, 0.003875492140650749f, 0.001599409501068294f, 0.003198819002136588f, 0.0030757873319089413f, 0.0034448818769305944f, 0.0020915353670716286f, 0.00215305108577013f, 0.0026451770681887865f, 0.004367617890238762f, 0.003137303050607443f, 0.00215305108577013f, 0.0022145670372992754f, 0.0024606299120932817f, 0.002276082755997777f, 0.0026451770681887865f, 0.002952755894511938f, 0.002276082755997777f, 0.0030142716132104397f, 0.0022145670372992754f, 0.0030142716132104397f, 0.0019685039296746254f, 0.002952755894511938f, 0.002583661349490285f, 0.0025221456307917833f, 0.005474901758134365f, 0.0026451770681887865f, 0.0020915353670716286f, 0.004060039296746254f, 0.004244586452841759f, 0.00215305108577013f, 0.0017839566571637988f, 0.002952755894511938f, 0.0012303149560466409f, 0.0016609252197667956f, 0.005105806980282068f, 0.0024606299120932817f, 0.0019685039296746254f, 0.0027682087384164333f, 0.002276082755997777f, 0.0017839566571637988f, 0.002276082755997777f, 0.0025221456307917833f, 0.0016609252197667956f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #73 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_voice_conv3_Mul_2_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.513051450252533f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #74 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_voice_conv3_Pow_output_0_DequantizeLinear_Output_const_4D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.003921568859368563f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #75 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_voice_conv3_activate_Relu_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.5130466818809509f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #76 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_voice_conv3_calc_out_scale_Pow_output_0_DequantizeLinear_Output_const_4D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.01568627543747425f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #77 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_voice_conv3_pool_MaxPool_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.1299632340669632f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #78 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_voice_conv3_quantize_Constant_output_0_DequantizeLinear_Output_const_4D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.501960813999176f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #79 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_voice_conv3_quantize_Div_output_0_0_0__voice_conv3_Mul_2_output_0_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.513051450252533f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #80 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_voice_conv3_quantize_Mul_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(65.66997528076172f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #81 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_voice_conv4_Conv_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.24441181123256683f),
    AI_PACK_INTQ_ZP(62)))

/* Int quant #82 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_voice_conv4_Conv_output_0_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 48,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0019685039296746254f, 0.00215305108577013f, 0.0014148622285574675f, 0.0014148622285574675f, 0.0018454724922776222f, 0.0019069882109761238f, 0.0017224409384652972f, 0.0024606299120932817f, 0.0012918306747451425f, 0.0026451770681887865f, 0.0028912401758134365f, 0.00215305108577013f, 0.002030019648373127f, 0.0025221456307917833f, 0.0012918306747451425f, 0.0015378936659544706f, 0.0019685039296746254f, 0.00239911419339478f, 0.0012303149560466409f, 0.0011687992373481393f, 0.0012918306747451425f, 0.0017224409384652972f, 0.0011072835186496377f, 0.0017839566571637988f, 0.0020915353670716286f, 0.002030019648373127f, 0.0026451770681887865f, 0.0015378936659544706f, 0.0012918306747451425f, 0.004060039296746254f, 0.0012303149560466409f, 0.0020915353670716286f, 0.0017839566571637988f, 0.0016609252197667956f, 0.0015378936659544706f, 0.0011072835186496377f, 0.0026451770681887865f, 0.00215305108577013f, 0.0019069882109761238f, 0.002583661349490285f, 0.0017839566571637988f, 0.0012918306747451425f, 0.0018454724922776222f, 0.001476377947255969f, 0.0019685039296746254f, 0.0020915353670716286f, 0.002829724457114935f, 0.0019685039296746254f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #83 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_voice_conv4_Mul_2_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.24843749403953552f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #84 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_voice_conv4_Pow_output_0_DequantizeLinear_Output_const_4D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.003921568859368563f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #85 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_voice_conv4_activate_Relu_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.24842792749404907f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #86 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_voice_conv4_calc_out_scale_Pow_output_0_DequantizeLinear_Output_const_4D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.01568627543747425f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #87 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_voice_conv4_quantize_Constant_output_0_DequantizeLinear_Output_const_4D_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.501960813999176f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #88 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_voice_conv4_quantize_Div_output_0_0_0__voice_conv4_Mul_2_output_0_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.24843749403953552f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #89 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_voice_conv4_quantize_Mul_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(31.79877471923828f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #90 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(kws_input_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #91 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(kws_logits_QuantizeLinear_Input_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(9.46987247467041f),
    AI_PACK_INTQ_ZP(26)))

/**  Tensor declarations section  *********************************************/
/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  _Reshape_output_0_to_chlast_output, AI_STATIC,
  0, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 4, 64), AI_STRIDE_INIT(4, 1, 1, 1, 4),
  1, &_Reshape_output_0_to_chlast_output_array, &_Reshape_output_0_to_chlast_output_array_intq)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  _Reshape_output_0_to_chlast_output0, AI_STATIC,
  1, 0x1,
  AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &_Reshape_output_0_to_chlast_output_array, &_Reshape_output_0_to_chlast_output_array_intq)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  _fc_Gemm_output_0_bias, AI_STATIC,
  2, 0x0,
  AI_SHAPE_INIT(4, 1, 21, 1, 1), AI_STRIDE_INIT(4, 4, 4, 84, 84),
  1, &_fc_Gemm_output_0_bias_array, NULL)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  _fc_Gemm_output_0_output, AI_STATIC,
  3, 0x1,
  AI_SHAPE_INIT(4, 1, 21, 1, 1), AI_STRIDE_INIT(4, 1, 1, 21, 21),
  1, &_fc_Gemm_output_0_output_array, &_fc_Gemm_output_0_output_array_intq)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  _fc_Gemm_output_0_scratch0, AI_STATIC,
  4, 0x0,
  AI_SHAPE_INIT(4, 1, 361, 1, 1), AI_STRIDE_INIT(4, 2, 2, 722, 722),
  1, &_fc_Gemm_output_0_scratch0_array, NULL)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  _fc_Gemm_output_0_weights, AI_STATIC,
  5, 0x1,
  AI_SHAPE_INIT(4, 256, 21, 1, 1), AI_STRIDE_INIT(4, 1, 256, 5376, 5376),
  1, &_fc_Gemm_output_0_weights_array, &_fc_Gemm_output_0_weights_array_intq)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  _fc_Pow_output_0_DequantizeLinear_Output_const_2D, AI_STATIC,
  6, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &_fc_Pow_output_0_DequantizeLinear_Output_const_2D_array, &_fc_Pow_output_0_DequantizeLinear_Output_const_2D_array_intq)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  _fc_quantize_Constant_1_output_0_2D, AI_STATIC,
  7, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &_fc_quantize_Constant_1_output_0_2D_array, NULL)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  _fc_quantize_Constant_output_0_DequantizeLinear_Output_const_2D, AI_STATIC,
  8, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &_fc_quantize_Constant_output_0_DequantizeLinear_Output_const_2D_array, &_fc_quantize_Constant_output_0_DequantizeLinear_Output_const_2D_array_intq)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  _fc_quantize_Div_output_0_0_0_kws_logits_QuantizeLinear_Input_conversion_output, AI_STATIC,
  9, 0x1,
  AI_SHAPE_INIT(4, 1, 21, 1, 1), AI_STRIDE_INIT(4, 1, 1, 21, 21),
  1, &_fc_quantize_Div_output_0_0_0_kws_logits_QuantizeLinear_Input_conversion_output_array, &_fc_quantize_Div_output_0_0_0_kws_logits_QuantizeLinear_Input_conversion_output_array_intq)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  _fc_quantize_Div_output_0_output, AI_STATIC,
  10, 0x0,
  AI_SHAPE_INIT(4, 1, 21, 1, 1), AI_STRIDE_INIT(4, 4, 4, 84, 84),
  1, &_fc_quantize_Div_output_0_output_array, NULL)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  _fc_quantize_Mul_output_0_0_0__fc_quantize_Round_output_0_conversion_output, AI_STATIC,
  11, 0x0,
  AI_SHAPE_INIT(4, 1, 21, 1, 1), AI_STRIDE_INIT(4, 4, 4, 84, 84),
  1, &_fc_quantize_Mul_output_0_0_0__fc_quantize_Round_output_0_conversion_output_array, NULL)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  _fc_quantize_Mul_output_0_output, AI_STATIC,
  12, 0x1,
  AI_SHAPE_INIT(4, 1, 21, 1, 1), AI_STRIDE_INIT(4, 1, 1, 21, 21),
  1, &_fc_quantize_Mul_output_0_output_array, &_fc_quantize_Mul_output_0_output_array_intq)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  _fc_quantize_Round_output_0_output, AI_STATIC,
  13, 0x0,
  AI_SHAPE_INIT(4, 1, 21, 1, 1), AI_STRIDE_INIT(4, 4, 4, 84, 84),
  1, &_fc_quantize_Round_output_0_output_array, NULL)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv1_Conv_output_0_bias, AI_STATIC,
  14, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_kws_conv1_Conv_output_0_bias_array, NULL)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv1_Conv_output_0_output, AI_STATIC,
  15, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 30), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &_kws_conv1_Conv_output_0_output_array, &_kws_conv1_Conv_output_0_output_array_intq)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv1_Conv_output_0_pad_before_output, AI_STATIC,
  16, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 1, 32), AI_STRIDE_INIT(4, 1, 1, 48, 48),
  1, &_kws_conv1_Conv_output_0_pad_before_output_array, &_kws_conv1_Conv_output_0_pad_before_output_array_intq)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv1_Conv_output_0_scratch0, AI_STATIC,
  17, 0x0,
  AI_SHAPE_INIT(4, 1, 6592, 1, 1), AI_STRIDE_INIT(4, 1, 1, 6592, 6592),
  1, &_kws_conv1_Conv_output_0_scratch0_array, NULL)

/* Tensor #18 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv1_Conv_output_0_weights, AI_STATIC,
  18, 0x1,
  AI_SHAPE_INIT(4, 48, 1, 3, 64), AI_STRIDE_INIT(4, 1, 48, 3072, 3072),
  1, &_kws_conv1_Conv_output_0_weights_array, &_kws_conv1_Conv_output_0_weights_array_intq)

/* Tensor #19 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv1_Mul_2_output_0_output, AI_STATIC,
  19, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 30), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &_kws_conv1_Mul_2_output_0_output_array, &_kws_conv1_Mul_2_output_0_output_array_intq)

/* Tensor #20 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv1_Pow_output_0_DequantizeLinear_Output_const_4D, AI_STATIC,
  20, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &_kws_conv1_Pow_output_0_DequantizeLinear_Output_const_4D_array, &_kws_conv1_Pow_output_0_DequantizeLinear_Output_const_4D_array_intq)

/* Tensor #21 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv1_activate_Relu_output_0_output, AI_STATIC,
  21, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 30), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &_kws_conv1_activate_Relu_output_0_output_array, &_kws_conv1_activate_Relu_output_0_output_array_intq)

/* Tensor #22 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv1_calc_out_scale_Pow_output_0_DequantizeLinear_Output_const_4D, AI_STATIC,
  22, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &_kws_conv1_calc_out_scale_Pow_output_0_DequantizeLinear_Output_const_4D_array, &_kws_conv1_calc_out_scale_Pow_output_0_DequantizeLinear_Output_const_4D_array_intq)

/* Tensor #23 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv1_pool_MaxPool_output_0_output, AI_STATIC,
  23, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 1, 30), AI_STRIDE_INIT(4, 1, 1, 48, 48),
  1, &_kws_conv1_pool_MaxPool_output_0_output_array, &_kws_conv1_pool_MaxPool_output_0_output_array_intq)

/* Tensor #24 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv1_quantize_Constant_1_output_0_4D, AI_STATIC,
  24, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &_kws_conv1_quantize_Constant_1_output_0_4D_array, NULL)

/* Tensor #25 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv1_quantize_Constant_output_0_DequantizeLinear_Output_const_4D, AI_STATIC,
  25, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &_kws_conv1_quantize_Constant_output_0_DequantizeLinear_Output_const_4D_array, &_kws_conv1_quantize_Constant_output_0_DequantizeLinear_Output_const_4D_array_intq)

/* Tensor #26 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv1_quantize_Div_output_0_0_0__kws_conv1_Mul_2_output_0_conversion_output, AI_STATIC,
  26, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 30), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &_kws_conv1_quantize_Div_output_0_0_0__kws_conv1_Mul_2_output_0_conversion_output_array, &_kws_conv1_quantize_Div_output_0_0_0__kws_conv1_Mul_2_output_0_conversion_output_array_intq)

/* Tensor #27 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv1_quantize_Div_output_0_output, AI_STATIC,
  27, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 30), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_kws_conv1_quantize_Div_output_0_output_array, NULL)

/* Tensor #28 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv1_quantize_Mul_output_0_0_0__kws_conv1_quantize_Round_output_0_conversion_output, AI_STATIC,
  28, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 30), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_kws_conv1_quantize_Mul_output_0_0_0__kws_conv1_quantize_Round_output_0_conversion_output_array, NULL)

/* Tensor #29 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv1_quantize_Mul_output_0_output, AI_STATIC,
  29, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 30), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &_kws_conv1_quantize_Mul_output_0_output_array, &_kws_conv1_quantize_Mul_output_0_output_array_intq)

/* Tensor #30 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv1_quantize_Round_output_0_output, AI_STATIC,
  30, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 30), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_kws_conv1_quantize_Round_output_0_output_array, NULL)

/* Tensor #31 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv2_Conv_output_0_bias, AI_STATIC,
  31, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 4, 4, 384, 384),
  1, &_kws_conv2_Conv_output_0_bias_array, NULL)

/* Tensor #32 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv2_Conv_output_0_output, AI_STATIC,
  32, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 1, 28), AI_STRIDE_INIT(4, 1, 1, 96, 96),
  1, &_kws_conv2_Conv_output_0_output_array, &_kws_conv2_Conv_output_0_output_array_intq)

/* Tensor #33 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv2_Conv_output_0_scratch0, AI_STATIC,
  33, 0x0,
  AI_SHAPE_INIT(4, 1, 7232, 1, 1), AI_STRIDE_INIT(4, 1, 1, 7232, 7232),
  1, &_kws_conv2_Conv_output_0_scratch0_array, NULL)

/* Tensor #34 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv2_Conv_output_0_weights, AI_STATIC,
  34, 0x1,
  AI_SHAPE_INIT(4, 64, 1, 3, 96), AI_STRIDE_INIT(4, 1, 64, 6144, 6144),
  1, &_kws_conv2_Conv_output_0_weights_array, &_kws_conv2_Conv_output_0_weights_array_intq)

/* Tensor #35 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv2_Mul_2_output_0_output, AI_STATIC,
  35, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 1, 28), AI_STRIDE_INIT(4, 1, 1, 96, 96),
  1, &_kws_conv2_Mul_2_output_0_output_array, &_kws_conv2_Mul_2_output_0_output_array_intq)

/* Tensor #36 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv2_Pow_output_0_DequantizeLinear_Output_const_4D, AI_STATIC,
  36, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &_kws_conv2_Pow_output_0_DequantizeLinear_Output_const_4D_array, &_kws_conv2_Pow_output_0_DequantizeLinear_Output_const_4D_array_intq)

/* Tensor #37 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv2_activate_Relu_output_0_output, AI_STATIC,
  37, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 1, 28), AI_STRIDE_INIT(4, 1, 1, 96, 96),
  1, &_kws_conv2_activate_Relu_output_0_output_array, &_kws_conv2_activate_Relu_output_0_output_array_intq)

/* Tensor #38 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv2_calc_out_scale_Pow_output_0_DequantizeLinear_Output_const_4D, AI_STATIC,
  38, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &_kws_conv2_calc_out_scale_Pow_output_0_DequantizeLinear_Output_const_4D_array, &_kws_conv2_calc_out_scale_Pow_output_0_DequantizeLinear_Output_const_4D_array_intq)

/* Tensor #39 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv2_quantize_Constant_1_output_0_4D, AI_STATIC,
  39, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &_kws_conv2_quantize_Constant_1_output_0_4D_array, NULL)

/* Tensor #40 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv2_quantize_Constant_output_0_DequantizeLinear_Output_const_4D, AI_STATIC,
  40, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &_kws_conv2_quantize_Constant_output_0_DequantizeLinear_Output_const_4D_array, &_kws_conv2_quantize_Constant_output_0_DequantizeLinear_Output_const_4D_array_intq)

/* Tensor #41 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv2_quantize_Div_output_0_0_0__kws_conv2_Mul_2_output_0_conversion_output, AI_STATIC,
  41, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 1, 28), AI_STRIDE_INIT(4, 1, 1, 96, 96),
  1, &_kws_conv2_quantize_Div_output_0_0_0__kws_conv2_Mul_2_output_0_conversion_output_array, &_kws_conv2_quantize_Div_output_0_0_0__kws_conv2_Mul_2_output_0_conversion_output_array_intq)

/* Tensor #42 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv2_quantize_Div_output_0_output, AI_STATIC,
  42, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 28), AI_STRIDE_INIT(4, 4, 4, 384, 384),
  1, &_kws_conv2_quantize_Div_output_0_output_array, NULL)

/* Tensor #43 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv2_quantize_Mul_output_0_0_0__kws_conv2_quantize_Round_output_0_conversion_output, AI_STATIC,
  43, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 28), AI_STRIDE_INIT(4, 4, 4, 384, 384),
  1, &_kws_conv2_quantize_Mul_output_0_0_0__kws_conv2_quantize_Round_output_0_conversion_output_array, NULL)

/* Tensor #44 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv2_quantize_Mul_output_0_output, AI_STATIC,
  44, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 1, 28), AI_STRIDE_INIT(4, 1, 1, 96, 96),
  1, &_kws_conv2_quantize_Mul_output_0_output_array, &_kws_conv2_quantize_Mul_output_0_output_array_intq)

/* Tensor #45 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv2_quantize_Round_output_0_output, AI_STATIC,
  45, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 28), AI_STRIDE_INIT(4, 4, 4, 384, 384),
  1, &_kws_conv2_quantize_Round_output_0_output_array, NULL)

/* Tensor #46 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv3_Conv_output_0_bias, AI_STATIC,
  46, 0x0,
  AI_SHAPE_INIT(4, 1, 100, 1, 1), AI_STRIDE_INIT(4, 4, 4, 400, 400),
  1, &_kws_conv3_Conv_output_0_bias_array, NULL)

/* Tensor #47 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv3_Conv_output_0_output, AI_STATIC,
  47, 0x1,
  AI_SHAPE_INIT(4, 1, 100, 1, 14), AI_STRIDE_INIT(4, 1, 1, 100, 100),
  1, &_kws_conv3_Conv_output_0_output_array, &_kws_conv3_Conv_output_0_output_array_intq)

/* Tensor #48 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv3_Conv_output_0_pad_before_output, AI_STATIC,
  48, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 1, 16), AI_STRIDE_INIT(4, 1, 1, 96, 96),
  1, &_kws_conv3_Conv_output_0_pad_before_output_array, &_kws_conv3_Conv_output_0_pad_before_output_array_intq)

/* Tensor #49 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv3_Conv_output_0_scratch0, AI_STATIC,
  49, 0x0,
  AI_SHAPE_INIT(4, 1, 7672, 1, 1), AI_STRIDE_INIT(4, 1, 1, 7672, 7672),
  1, &_kws_conv3_Conv_output_0_scratch0_array, NULL)

/* Tensor #50 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv3_Conv_output_0_weights, AI_STATIC,
  50, 0x1,
  AI_SHAPE_INIT(4, 96, 1, 3, 100), AI_STRIDE_INIT(4, 1, 96, 9600, 9600),
  1, &_kws_conv3_Conv_output_0_weights_array, &_kws_conv3_Conv_output_0_weights_array_intq)

/* Tensor #51 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv3_Mul_2_output_0_output, AI_STATIC,
  51, 0x1,
  AI_SHAPE_INIT(4, 1, 100, 1, 14), AI_STRIDE_INIT(4, 1, 1, 100, 100),
  1, &_kws_conv3_Mul_2_output_0_output_array, &_kws_conv3_Mul_2_output_0_output_array_intq)

/* Tensor #52 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv3_Pow_output_0_DequantizeLinear_Output_const_4D, AI_STATIC,
  52, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &_kws_conv3_Pow_output_0_DequantizeLinear_Output_const_4D_array, &_kws_conv3_Pow_output_0_DequantizeLinear_Output_const_4D_array_intq)

/* Tensor #53 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv3_activate_Relu_output_0_output, AI_STATIC,
  53, 0x1,
  AI_SHAPE_INIT(4, 1, 100, 1, 14), AI_STRIDE_INIT(4, 1, 1, 100, 100),
  1, &_kws_conv3_activate_Relu_output_0_output_array, &_kws_conv3_activate_Relu_output_0_output_array_intq)

/* Tensor #54 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv3_calc_out_scale_Pow_output_0_DequantizeLinear_Output_const_4D, AI_STATIC,
  54, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &_kws_conv3_calc_out_scale_Pow_output_0_DequantizeLinear_Output_const_4D_array, &_kws_conv3_calc_out_scale_Pow_output_0_DequantizeLinear_Output_const_4D_array_intq)

/* Tensor #55 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv3_pool_AveragePool_output_0_output, AI_STATIC,
  55, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 1, 14), AI_STRIDE_INIT(4, 1, 1, 96, 96),
  1, &_kws_conv3_pool_AveragePool_output_0_output_array, &_kws_conv3_pool_AveragePool_output_0_output_array_intq)

/* Tensor #56 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv3_quantize_Constant_1_output_0_4D, AI_STATIC,
  56, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &_kws_conv3_quantize_Constant_1_output_0_4D_array, NULL)

/* Tensor #57 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv3_quantize_Constant_output_0_DequantizeLinear_Output_const_4D, AI_STATIC,
  57, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &_kws_conv3_quantize_Constant_output_0_DequantizeLinear_Output_const_4D_array, &_kws_conv3_quantize_Constant_output_0_DequantizeLinear_Output_const_4D_array_intq)

/* Tensor #58 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv3_quantize_Div_output_0_0_0__kws_conv3_Mul_2_output_0_conversion_output, AI_STATIC,
  58, 0x1,
  AI_SHAPE_INIT(4, 1, 100, 1, 14), AI_STRIDE_INIT(4, 1, 1, 100, 100),
  1, &_kws_conv3_quantize_Div_output_0_0_0__kws_conv3_Mul_2_output_0_conversion_output_array, &_kws_conv3_quantize_Div_output_0_0_0__kws_conv3_Mul_2_output_0_conversion_output_array_intq)

/* Tensor #59 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv3_quantize_Div_output_0_output, AI_STATIC,
  59, 0x0,
  AI_SHAPE_INIT(4, 1, 100, 1, 14), AI_STRIDE_INIT(4, 4, 4, 400, 400),
  1, &_kws_conv3_quantize_Div_output_0_output_array, NULL)

/* Tensor #60 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv3_quantize_Mul_output_0_0_0__kws_conv3_quantize_Round_output_0_conversion_output, AI_STATIC,
  60, 0x0,
  AI_SHAPE_INIT(4, 1, 100, 1, 14), AI_STRIDE_INIT(4, 4, 4, 400, 400),
  1, &_kws_conv3_quantize_Mul_output_0_0_0__kws_conv3_quantize_Round_output_0_conversion_output_array, NULL)

/* Tensor #61 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv3_quantize_Mul_output_0_output, AI_STATIC,
  61, 0x1,
  AI_SHAPE_INIT(4, 1, 100, 1, 14), AI_STRIDE_INIT(4, 1, 1, 100, 100),
  1, &_kws_conv3_quantize_Mul_output_0_output_array, &_kws_conv3_quantize_Mul_output_0_output_array_intq)

/* Tensor #62 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv3_quantize_Round_output_0_output, AI_STATIC,
  62, 0x0,
  AI_SHAPE_INIT(4, 1, 100, 1, 14), AI_STRIDE_INIT(4, 4, 4, 400, 400),
  1, &_kws_conv3_quantize_Round_output_0_output_array, NULL)

/* Tensor #63 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv3_quantize_pool_Constant_1_output_0_4D, AI_STATIC,
  63, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &_kws_conv3_quantize_pool_Constant_1_output_0_4D_array, NULL)

/* Tensor #64 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv3_quantize_pool_Constant_output_0_DequantizeLinear_Output_const_4D, AI_STATIC,
  64, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &_kws_conv3_quantize_pool_Constant_output_0_DequantizeLinear_Output_const_4D_array, &_kws_conv3_quantize_pool_Constant_output_0_DequantizeLinear_Output_const_4D_array_intq)

/* Tensor #65 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv3_quantize_pool_Div_output_0_0_0__kws_conv3_Conv_output_0_pad_before_conversion_output, AI_STATIC,
  65, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 1, 14), AI_STRIDE_INIT(4, 1, 1, 96, 96),
  1, &_kws_conv3_quantize_pool_Div_output_0_0_0__kws_conv3_Conv_output_0_pad_before_conversion_output_array, &_kws_conv3_quantize_pool_Div_output_0_0_0__kws_conv3_Conv_output_0_pad_before_conversion_output_array_intq)

/* Tensor #66 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv3_quantize_pool_Div_output_0_output, AI_STATIC,
  66, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 14), AI_STRIDE_INIT(4, 4, 4, 384, 384),
  1, &_kws_conv3_quantize_pool_Div_output_0_output_array, NULL)

/* Tensor #67 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv3_quantize_pool_Floor_output_0_output, AI_STATIC,
  67, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 14), AI_STRIDE_INIT(4, 4, 4, 384, 384),
  1, &_kws_conv3_quantize_pool_Floor_output_0_output_array, NULL)

/* Tensor #68 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv3_quantize_pool_Mul_output_0_0_0__kws_conv3_quantize_pool_Floor_output_0_conversion_output, AI_STATIC,
  68, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 14), AI_STRIDE_INIT(4, 4, 4, 384, 384),
  1, &_kws_conv3_quantize_pool_Mul_output_0_0_0__kws_conv3_quantize_pool_Floor_output_0_conversion_output_array, NULL)

/* Tensor #69 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv3_quantize_pool_Mul_output_0_output, AI_STATIC,
  69, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 1, 14), AI_STRIDE_INIT(4, 1, 1, 96, 96),
  1, &_kws_conv3_quantize_pool_Mul_output_0_output_array, &_kws_conv3_quantize_pool_Mul_output_0_output_array_intq)

/* Tensor #70 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv4_Conv_output_0_bias, AI_STATIC,
  70, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_kws_conv4_Conv_output_0_bias_array, NULL)

/* Tensor #71 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv4_Conv_output_0_output, AI_STATIC,
  71, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 4), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &_kws_conv4_Conv_output_0_output_array, &_kws_conv4_Conv_output_0_output_array_intq)

/* Tensor #72 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv4_Conv_output_0_pad_before_output, AI_STATIC,
  72, 0x1,
  AI_SHAPE_INIT(4, 1, 100, 1, 9), AI_STRIDE_INIT(4, 1, 1, 100, 100),
  1, &_kws_conv4_Conv_output_0_pad_before_output_array, &_kws_conv4_Conv_output_0_pad_before_output_array_intq)

/* Tensor #73 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv4_Conv_output_0_scratch0, AI_STATIC,
  73, 0x0,
  AI_SHAPE_INIT(4, 1, 8416, 1, 1), AI_STRIDE_INIT(4, 1, 1, 8416, 8416),
  1, &_kws_conv4_Conv_output_0_scratch0_array, NULL)

/* Tensor #74 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv4_Conv_output_0_weights, AI_STATIC,
  74, 0x1,
  AI_SHAPE_INIT(4, 100, 1, 6, 64), AI_STRIDE_INIT(4, 1, 100, 6400, 6400),
  1, &_kws_conv4_Conv_output_0_weights_array, &_kws_conv4_Conv_output_0_weights_array_intq)

/* Tensor #75 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv4_Mul_2_output_0_output, AI_STATIC,
  75, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 4), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &_kws_conv4_Mul_2_output_0_output_array, &_kws_conv4_Mul_2_output_0_output_array_intq)

/* Tensor #76 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv4_Pow_output_0_DequantizeLinear_Output_const_4D, AI_STATIC,
  76, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &_kws_conv4_Pow_output_0_DequantizeLinear_Output_const_4D_array, &_kws_conv4_Pow_output_0_DequantizeLinear_Output_const_4D_array_intq)

/* Tensor #77 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv4_activate_Relu_output_0_output, AI_STATIC,
  77, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 4), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &_kws_conv4_activate_Relu_output_0_output_array, &_kws_conv4_activate_Relu_output_0_output_array_intq)

/* Tensor #78 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv4_calc_out_scale_Pow_output_0_DequantizeLinear_Output_const_4D, AI_STATIC,
  78, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &_kws_conv4_calc_out_scale_Pow_output_0_DequantizeLinear_Output_const_4D_array, &_kws_conv4_calc_out_scale_Pow_output_0_DequantizeLinear_Output_const_4D_array_intq)

/* Tensor #79 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv4_pool_MaxPool_output_0_output, AI_STATIC,
  79, 0x1,
  AI_SHAPE_INIT(4, 1, 100, 1, 7), AI_STRIDE_INIT(4, 1, 1, 100, 100),
  1, &_kws_conv4_pool_MaxPool_output_0_output_array, &_kws_conv4_pool_MaxPool_output_0_output_array_intq)

/* Tensor #80 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv4_quantize_Constant_1_output_0_4D, AI_STATIC,
  80, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &_kws_conv4_quantize_Constant_1_output_0_4D_array, NULL)

/* Tensor #81 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv4_quantize_Constant_output_0_DequantizeLinear_Output_const_4D, AI_STATIC,
  81, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &_kws_conv4_quantize_Constant_output_0_DequantizeLinear_Output_const_4D_array, &_kws_conv4_quantize_Constant_output_0_DequantizeLinear_Output_const_4D_array_intq)

/* Tensor #82 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv4_quantize_Div_output_0_0_0__kws_conv4_Mul_2_output_0_conversion_output, AI_STATIC,
  82, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 4), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &_kws_conv4_quantize_Div_output_0_0_0__kws_conv4_Mul_2_output_0_conversion_output_array, &_kws_conv4_quantize_Div_output_0_0_0__kws_conv4_Mul_2_output_0_conversion_output_array_intq)

/* Tensor #83 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv4_quantize_Div_output_0_output, AI_STATIC,
  83, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 4), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_kws_conv4_quantize_Div_output_0_output_array, NULL)

/* Tensor #84 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv4_quantize_Mul_output_0_0_0__kws_conv4_quantize_Round_output_0_conversion_output, AI_STATIC,
  84, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 4), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_kws_conv4_quantize_Mul_output_0_0_0__kws_conv4_quantize_Round_output_0_conversion_output_array, NULL)

/* Tensor #85 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv4_quantize_Mul_output_0_output, AI_STATIC,
  85, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 4), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &_kws_conv4_quantize_Mul_output_0_output_array, &_kws_conv4_quantize_Mul_output_0_output_array_intq)

/* Tensor #86 */
AI_TENSOR_OBJ_DECLARE(
  _kws_conv4_quantize_Round_output_0_output, AI_STATIC,
  86, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 4), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_kws_conv4_quantize_Round_output_0_output_array, NULL)

/* Tensor #87 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv1_Conv_output_0_bias, AI_STATIC,
  87, 0x0,
  AI_SHAPE_INIT(4, 1, 100, 1, 1), AI_STRIDE_INIT(4, 4, 4, 400, 400),
  1, &_voice_conv1_Conv_output_0_bias_array, NULL)

/* Tensor #88 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv1_Conv_output_0_output, AI_STATIC,
  88, 0x1,
  AI_SHAPE_INIT(4, 1, 100, 1, 128), AI_STRIDE_INIT(4, 1, 1, 100, 100),
  1, &_voice_conv1_Conv_output_0_output_array, &_voice_conv1_Conv_output_0_output_array_intq)

/* Tensor #89 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv1_Conv_output_0_scratch0, AI_STATIC,
  89, 0x0,
  AI_SHAPE_INIT(4, 1, 7032, 1, 1), AI_STRIDE_INIT(4, 1, 1, 7032, 7032),
  1, &_voice_conv1_Conv_output_0_scratch0_array, NULL)

/* Tensor #90 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv1_Conv_output_0_weights, AI_STATIC,
  90, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 1, 100), AI_STRIDE_INIT(4, 1, 1, 100, 12800),
  1, &_voice_conv1_Conv_output_0_weights_array, &_voice_conv1_Conv_output_0_weights_array_intq)

/* Tensor #91 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv1_Mul_2_output_0_output, AI_STATIC,
  91, 0x1,
  AI_SHAPE_INIT(4, 1, 100, 1, 128), AI_STRIDE_INIT(4, 1, 1, 100, 100),
  1, &_voice_conv1_Mul_2_output_0_output_array, &_voice_conv1_Mul_2_output_0_output_array_intq)

/* Tensor #92 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv1_Pow_output_0_DequantizeLinear_Output_const_4D, AI_STATIC,
  92, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &_voice_conv1_Pow_output_0_DequantizeLinear_Output_const_4D_array, &_voice_conv1_Pow_output_0_DequantizeLinear_Output_const_4D_array_intq)

/* Tensor #93 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv1_activate_Relu_output_0_output, AI_STATIC,
  93, 0x1,
  AI_SHAPE_INIT(4, 1, 100, 1, 128), AI_STRIDE_INIT(4, 1, 1, 100, 100),
  1, &_voice_conv1_activate_Relu_output_0_output_array, &_voice_conv1_activate_Relu_output_0_output_array_intq)

/* Tensor #94 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv1_calc_out_scale_Pow_output_0_DequantizeLinear_Output_const_4D, AI_STATIC,
  94, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &_voice_conv1_calc_out_scale_Pow_output_0_DequantizeLinear_Output_const_4D_array, &_voice_conv1_calc_out_scale_Pow_output_0_DequantizeLinear_Output_const_4D_array_intq)

/* Tensor #95 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv1_quantize_Constant_1_output_0_4D, AI_STATIC,
  95, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &_voice_conv1_quantize_Constant_1_output_0_4D_array, NULL)

/* Tensor #96 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv1_quantize_Constant_output_0_DequantizeLinear_Output_const_4D, AI_STATIC,
  96, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &_voice_conv1_quantize_Constant_output_0_DequantizeLinear_Output_const_4D_array, &_voice_conv1_quantize_Constant_output_0_DequantizeLinear_Output_const_4D_array_intq)

/* Tensor #97 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv1_quantize_Div_output_0_0_0__voice_conv1_Mul_2_output_0_conversion_output, AI_STATIC,
  97, 0x1,
  AI_SHAPE_INIT(4, 1, 100, 1, 128), AI_STRIDE_INIT(4, 1, 1, 100, 100),
  1, &_voice_conv1_quantize_Div_output_0_0_0__voice_conv1_Mul_2_output_0_conversion_output_array, &_voice_conv1_quantize_Div_output_0_0_0__voice_conv1_Mul_2_output_0_conversion_output_array_intq)

/* Tensor #98 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv1_quantize_Div_output_0_output, AI_STATIC,
  98, 0x0,
  AI_SHAPE_INIT(4, 1, 100, 1, 128), AI_STRIDE_INIT(4, 4, 4, 400, 400),
  1, &_voice_conv1_quantize_Div_output_0_output_array, NULL)

/* Tensor #99 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv1_quantize_Mul_output_0_0_0__voice_conv1_quantize_Round_output_0_conversion_output, AI_STATIC,
  99, 0x0,
  AI_SHAPE_INIT(4, 1, 100, 1, 128), AI_STRIDE_INIT(4, 4, 4, 400, 400),
  1, &_voice_conv1_quantize_Mul_output_0_0_0__voice_conv1_quantize_Round_output_0_conversion_output_array, NULL)

/* Tensor #100 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv1_quantize_Mul_output_0_output, AI_STATIC,
  100, 0x1,
  AI_SHAPE_INIT(4, 1, 100, 1, 128), AI_STRIDE_INIT(4, 1, 1, 100, 100),
  1, &_voice_conv1_quantize_Mul_output_0_output_array, &_voice_conv1_quantize_Mul_output_0_output_array_intq)

/* Tensor #101 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv1_quantize_Round_output_0_output, AI_STATIC,
  101, 0x0,
  AI_SHAPE_INIT(4, 1, 100, 1, 128), AI_STRIDE_INIT(4, 4, 4, 400, 400),
  1, &_voice_conv1_quantize_Round_output_0_output_array, NULL)

/* Tensor #102 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv2_Conv_output_0_bias, AI_STATIC,
  102, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 1), AI_STRIDE_INIT(4, 4, 4, 384, 384),
  1, &_voice_conv2_Conv_output_0_bias_array, NULL)

/* Tensor #103 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv2_Conv_output_0_output, AI_STATIC,
  103, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 1, 126), AI_STRIDE_INIT(4, 1, 1, 96, 96),
  1, &_voice_conv2_Conv_output_0_output_array, &_voice_conv2_Conv_output_0_output_array_intq)

/* Tensor #104 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv2_Conv_output_0_scratch0, AI_STATIC,
  104, 0x0,
  AI_SHAPE_INIT(4, 1, 7664, 1, 1), AI_STRIDE_INIT(4, 1, 1, 7664, 7664),
  1, &_voice_conv2_Conv_output_0_scratch0_array, NULL)

/* Tensor #105 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv2_Conv_output_0_weights, AI_STATIC,
  105, 0x1,
  AI_SHAPE_INIT(4, 100, 1, 3, 96), AI_STRIDE_INIT(4, 1, 100, 9600, 9600),
  1, &_voice_conv2_Conv_output_0_weights_array, &_voice_conv2_Conv_output_0_weights_array_intq)

/* Tensor #106 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv2_Mul_2_output_0_output, AI_STATIC,
  106, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 1, 126), AI_STRIDE_INIT(4, 1, 1, 96, 96),
  1, &_voice_conv2_Mul_2_output_0_output_array, &_voice_conv2_Mul_2_output_0_output_array_intq)

/* Tensor #107 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv2_Pow_output_0_DequantizeLinear_Output_const_4D, AI_STATIC,
  107, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &_voice_conv2_Pow_output_0_DequantizeLinear_Output_const_4D_array, &_voice_conv2_Pow_output_0_DequantizeLinear_Output_const_4D_array_intq)

/* Tensor #108 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv2_activate_Relu_output_0_output, AI_STATIC,
  108, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 1, 126), AI_STRIDE_INIT(4, 1, 1, 96, 96),
  1, &_voice_conv2_activate_Relu_output_0_output_array, &_voice_conv2_activate_Relu_output_0_output_array_intq)

/* Tensor #109 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv2_calc_out_scale_Pow_output_0_DequantizeLinear_Output_const_4D, AI_STATIC,
  109, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &_voice_conv2_calc_out_scale_Pow_output_0_DequantizeLinear_Output_const_4D_array, &_voice_conv2_calc_out_scale_Pow_output_0_DequantizeLinear_Output_const_4D_array_intq)

/* Tensor #110 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv2_quantize_Constant_1_output_0_4D, AI_STATIC,
  110, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &_voice_conv2_quantize_Constant_1_output_0_4D_array, NULL)

/* Tensor #111 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv2_quantize_Constant_output_0_DequantizeLinear_Output_const_4D, AI_STATIC,
  111, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &_voice_conv2_quantize_Constant_output_0_DequantizeLinear_Output_const_4D_array, &_voice_conv2_quantize_Constant_output_0_DequantizeLinear_Output_const_4D_array_intq)

/* Tensor #112 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv2_quantize_Div_output_0_0_0__voice_conv2_Mul_2_output_0_conversion_output, AI_STATIC,
  112, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 1, 126), AI_STRIDE_INIT(4, 1, 1, 96, 96),
  1, &_voice_conv2_quantize_Div_output_0_0_0__voice_conv2_Mul_2_output_0_conversion_output_array, &_voice_conv2_quantize_Div_output_0_0_0__voice_conv2_Mul_2_output_0_conversion_output_array_intq)

/* Tensor #113 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv2_quantize_Div_output_0_output, AI_STATIC,
  113, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 126), AI_STRIDE_INIT(4, 4, 4, 384, 384),
  1, &_voice_conv2_quantize_Div_output_0_output_array, NULL)

/* Tensor #114 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv2_quantize_Mul_output_0_0_0__voice_conv2_quantize_Round_output_0_conversion_output, AI_STATIC,
  114, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 126), AI_STRIDE_INIT(4, 4, 4, 384, 384),
  1, &_voice_conv2_quantize_Mul_output_0_0_0__voice_conv2_quantize_Round_output_0_conversion_output_array, NULL)

/* Tensor #115 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv2_quantize_Mul_output_0_output, AI_STATIC,
  115, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 1, 126), AI_STRIDE_INIT(4, 1, 1, 96, 96),
  1, &_voice_conv2_quantize_Mul_output_0_output_array, &_voice_conv2_quantize_Mul_output_0_output_array_intq)

/* Tensor #116 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv2_quantize_Round_output_0_output, AI_STATIC,
  116, 0x0,
  AI_SHAPE_INIT(4, 1, 96, 1, 126), AI_STRIDE_INIT(4, 4, 4, 384, 384),
  1, &_voice_conv2_quantize_Round_output_0_output_array, NULL)

/* Tensor #117 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv3_Conv_output_0_bias, AI_STATIC,
  117, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_voice_conv3_Conv_output_0_bias_array, NULL)

/* Tensor #118 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv3_Conv_output_0_output, AI_STATIC,
  118, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 63), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &_voice_conv3_Conv_output_0_output_array, &_voice_conv3_Conv_output_0_output_array_intq)

/* Tensor #119 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv3_Conv_output_0_pad_before_output, AI_STATIC,
  119, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 1, 65), AI_STRIDE_INIT(4, 1, 1, 96, 96),
  1, &_voice_conv3_Conv_output_0_pad_before_output_array, &_voice_conv3_Conv_output_0_pad_before_output_array_intq)

/* Tensor #120 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv3_Conv_output_0_scratch0, AI_STATIC,
  120, 0x0,
  AI_SHAPE_INIT(4, 1, 7168, 1, 1), AI_STRIDE_INIT(4, 1, 1, 7168, 7168),
  1, &_voice_conv3_Conv_output_0_scratch0_array, NULL)

/* Tensor #121 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv3_Conv_output_0_weights, AI_STATIC,
  121, 0x1,
  AI_SHAPE_INIT(4, 96, 1, 3, 64), AI_STRIDE_INIT(4, 1, 96, 6144, 6144),
  1, &_voice_conv3_Conv_output_0_weights_array, &_voice_conv3_Conv_output_0_weights_array_intq)

/* Tensor #122 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv3_Mul_2_output_0_output, AI_STATIC,
  122, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 63), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &_voice_conv3_Mul_2_output_0_output_array, &_voice_conv3_Mul_2_output_0_output_array_intq)

/* Tensor #123 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv3_Pow_output_0_DequantizeLinear_Output_const_4D, AI_STATIC,
  123, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &_voice_conv3_Pow_output_0_DequantizeLinear_Output_const_4D_array, &_voice_conv3_Pow_output_0_DequantizeLinear_Output_const_4D_array_intq)

/* Tensor #124 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv3_activate_Relu_output_0_output, AI_STATIC,
  124, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 63), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &_voice_conv3_activate_Relu_output_0_output_array, &_voice_conv3_activate_Relu_output_0_output_array_intq)

/* Tensor #125 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv3_calc_out_scale_Pow_output_0_DequantizeLinear_Output_const_4D, AI_STATIC,
  125, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &_voice_conv3_calc_out_scale_Pow_output_0_DequantizeLinear_Output_const_4D_array, &_voice_conv3_calc_out_scale_Pow_output_0_DequantizeLinear_Output_const_4D_array_intq)

/* Tensor #126 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv3_pool_MaxPool_output_0_output, AI_STATIC,
  126, 0x1,
  AI_SHAPE_INIT(4, 1, 96, 1, 63), AI_STRIDE_INIT(4, 1, 1, 96, 96),
  1, &_voice_conv3_pool_MaxPool_output_0_output_array, &_voice_conv3_pool_MaxPool_output_0_output_array_intq)

/* Tensor #127 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv3_quantize_Constant_1_output_0_4D, AI_STATIC,
  127, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &_voice_conv3_quantize_Constant_1_output_0_4D_array, NULL)

/* Tensor #128 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv3_quantize_Constant_output_0_DequantizeLinear_Output_const_4D, AI_STATIC,
  128, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &_voice_conv3_quantize_Constant_output_0_DequantizeLinear_Output_const_4D_array, &_voice_conv3_quantize_Constant_output_0_DequantizeLinear_Output_const_4D_array_intq)

/* Tensor #129 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv3_quantize_Div_output_0_0_0__voice_conv3_Mul_2_output_0_conversion_output, AI_STATIC,
  129, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 63), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &_voice_conv3_quantize_Div_output_0_0_0__voice_conv3_Mul_2_output_0_conversion_output_array, &_voice_conv3_quantize_Div_output_0_0_0__voice_conv3_Mul_2_output_0_conversion_output_array_intq)

/* Tensor #130 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv3_quantize_Div_output_0_output, AI_STATIC,
  130, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 63), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_voice_conv3_quantize_Div_output_0_output_array, NULL)

/* Tensor #131 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv3_quantize_Mul_output_0_0_0__voice_conv3_quantize_Round_output_0_conversion_output, AI_STATIC,
  131, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 63), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_voice_conv3_quantize_Mul_output_0_0_0__voice_conv3_quantize_Round_output_0_conversion_output_array, NULL)

/* Tensor #132 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv3_quantize_Mul_output_0_output, AI_STATIC,
  132, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 63), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &_voice_conv3_quantize_Mul_output_0_output_array, &_voice_conv3_quantize_Mul_output_0_output_array_intq)

/* Tensor #133 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv3_quantize_Round_output_0_output, AI_STATIC,
  133, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 63), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_voice_conv3_quantize_Round_output_0_output_array, NULL)

/* Tensor #134 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv4_Conv_output_0_bias, AI_STATIC,
  134, 0x0,
  AI_SHAPE_INIT(4, 1, 48, 1, 1), AI_STRIDE_INIT(4, 4, 4, 192, 192),
  1, &_voice_conv4_Conv_output_0_bias_array, NULL)

/* Tensor #135 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv4_Conv_output_0_output, AI_STATIC,
  135, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 1, 61), AI_STRIDE_INIT(4, 1, 1, 48, 48),
  1, &_voice_conv4_Conv_output_0_output_array, &_voice_conv4_Conv_output_0_output_array_intq)

/* Tensor #136 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv4_Conv_output_0_scratch0, AI_STATIC,
  136, 0x0,
  AI_SHAPE_INIT(4, 1, 6560, 1, 1), AI_STRIDE_INIT(4, 1, 1, 6560, 6560),
  1, &_voice_conv4_Conv_output_0_scratch0_array, NULL)

/* Tensor #137 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv4_Conv_output_0_weights, AI_STATIC,
  137, 0x1,
  AI_SHAPE_INIT(4, 64, 1, 3, 48), AI_STRIDE_INIT(4, 1, 64, 3072, 3072),
  1, &_voice_conv4_Conv_output_0_weights_array, &_voice_conv4_Conv_output_0_weights_array_intq)

/* Tensor #138 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv4_Mul_2_output_0_output, AI_STATIC,
  138, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 1, 61), AI_STRIDE_INIT(4, 1, 1, 48, 48),
  1, &_voice_conv4_Mul_2_output_0_output_array, &_voice_conv4_Mul_2_output_0_output_array_intq)

/* Tensor #139 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv4_Pow_output_0_DequantizeLinear_Output_const_4D, AI_STATIC,
  139, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &_voice_conv4_Pow_output_0_DequantizeLinear_Output_const_4D_array, &_voice_conv4_Pow_output_0_DequantizeLinear_Output_const_4D_array_intq)

/* Tensor #140 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv4_activate_Relu_output_0_output, AI_STATIC,
  140, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 1, 61), AI_STRIDE_INIT(4, 1, 1, 48, 48),
  1, &_voice_conv4_activate_Relu_output_0_output_array, &_voice_conv4_activate_Relu_output_0_output_array_intq)

/* Tensor #141 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv4_calc_out_scale_Pow_output_0_DequantizeLinear_Output_const_4D, AI_STATIC,
  141, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &_voice_conv4_calc_out_scale_Pow_output_0_DequantizeLinear_Output_const_4D_array, &_voice_conv4_calc_out_scale_Pow_output_0_DequantizeLinear_Output_const_4D_array_intq)

/* Tensor #142 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv4_quantize_Constant_1_output_0_4D, AI_STATIC,
  142, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &_voice_conv4_quantize_Constant_1_output_0_4D_array, NULL)

/* Tensor #143 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv4_quantize_Constant_output_0_DequantizeLinear_Output_const_4D, AI_STATIC,
  143, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &_voice_conv4_quantize_Constant_output_0_DequantizeLinear_Output_const_4D_array, &_voice_conv4_quantize_Constant_output_0_DequantizeLinear_Output_const_4D_array_intq)

/* Tensor #144 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv4_quantize_Div_output_0_0_0__voice_conv4_Mul_2_output_0_conversion_output, AI_STATIC,
  144, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 1, 61), AI_STRIDE_INIT(4, 1, 1, 48, 48),
  1, &_voice_conv4_quantize_Div_output_0_0_0__voice_conv4_Mul_2_output_0_conversion_output_array, &_voice_conv4_quantize_Div_output_0_0_0__voice_conv4_Mul_2_output_0_conversion_output_array_intq)

/* Tensor #145 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv4_quantize_Div_output_0_output, AI_STATIC,
  145, 0x0,
  AI_SHAPE_INIT(4, 1, 48, 1, 61), AI_STRIDE_INIT(4, 4, 4, 192, 192),
  1, &_voice_conv4_quantize_Div_output_0_output_array, NULL)

/* Tensor #146 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv4_quantize_Mul_output_0_0_0__voice_conv4_quantize_Round_output_0_conversion_output, AI_STATIC,
  146, 0x0,
  AI_SHAPE_INIT(4, 1, 48, 1, 61), AI_STRIDE_INIT(4, 4, 4, 192, 192),
  1, &_voice_conv4_quantize_Mul_output_0_0_0__voice_conv4_quantize_Round_output_0_conversion_output_array, NULL)

/* Tensor #147 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv4_quantize_Mul_output_0_output, AI_STATIC,
  147, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 1, 61), AI_STRIDE_INIT(4, 1, 1, 48, 48),
  1, &_voice_conv4_quantize_Mul_output_0_output_array, &_voice_conv4_quantize_Mul_output_0_output_array_intq)

/* Tensor #148 */
AI_TENSOR_OBJ_DECLARE(
  _voice_conv4_quantize_Round_output_0_output, AI_STATIC,
  148, 0x0,
  AI_SHAPE_INIT(4, 1, 48, 1, 61), AI_STRIDE_INIT(4, 4, 4, 192, 192),
  1, &_voice_conv4_quantize_Round_output_0_output_array, NULL)

/* Tensor #149 */
AI_TENSOR_OBJ_DECLARE(
  kws_input_output, AI_STATIC,
  149, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 128, 128), AI_STRIDE_INIT(4, 1, 1, 1, 128),
  1, &kws_input_output_array, &kws_input_output_array_intq)

/* Tensor #150 */
AI_TENSOR_OBJ_DECLARE(
  kws_logits_QuantizeLinear_Input_output, AI_STATIC,
  150, 0x1,
  AI_SHAPE_INIT(4, 1, 21, 1, 1), AI_STRIDE_INIT(4, 1, 1, 21, 21),
  1, &kws_logits_QuantizeLinear_Input_output_array, &kws_logits_QuantizeLinear_Input_output_array_intq)



/**  Layer declarations section  **********************************************/


AI_TENSOR_CHAIN_OBJ_DECLARE(
  kws_logits_QuantizeLinear_Input_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_fc_quantize_Div_output_0_0_0_kws_logits_QuantizeLinear_Input_conversion_output, &_fc_Pow_output_0_DequantizeLinear_Output_const_2D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &kws_logits_QuantizeLinear_Input_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  kws_logits_QuantizeLinear_Input_layer, 208,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &kws_logits_QuantizeLinear_Input_chain,
  NULL, &kws_logits_QuantizeLinear_Input_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _fc_quantize_Div_output_0_0_0_kws_logits_QuantizeLinear_Input_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_fc_quantize_Div_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_fc_quantize_Div_output_0_0_0_kws_logits_QuantizeLinear_Input_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _fc_quantize_Div_output_0_0_0_kws_logits_QuantizeLinear_Input_conversion_layer, 205,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &_fc_quantize_Div_output_0_0_0_kws_logits_QuantizeLinear_Input_conversion_chain,
  NULL, &kws_logits_QuantizeLinear_Input_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _fc_quantize_Div_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_fc_quantize_Round_output_0_output, &_fc_quantize_Constant_1_output_0_2D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_fc_quantize_Div_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _fc_quantize_Div_output_0_layer, 205,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_fc_quantize_Div_output_0_chain,
  NULL, &_fc_quantize_Div_output_0_0_0_kws_logits_QuantizeLinear_Input_conversion_layer, AI_STATIC, 
  .operation = ai_div_f32, 
  .buffer_operation = ai_div_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _fc_quantize_Round_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_fc_quantize_Mul_output_0_0_0__fc_quantize_Round_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_fc_quantize_Round_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _fc_quantize_Round_output_0_layer, 204,
  NL_TYPE, 0x0, NULL,
  nl, forward_round,
  &_fc_quantize_Round_output_0_chain,
  NULL, &_fc_quantize_Div_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _fc_quantize_Mul_output_0_0_0__fc_quantize_Round_output_0_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_fc_quantize_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_fc_quantize_Mul_output_0_0_0__fc_quantize_Round_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _fc_quantize_Mul_output_0_0_0__fc_quantize_Round_output_0_conversion_layer, 201,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &_fc_quantize_Mul_output_0_0_0__fc_quantize_Round_output_0_conversion_chain,
  NULL, &_fc_quantize_Round_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _fc_quantize_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_fc_Gemm_output_0_output, &_fc_quantize_Constant_output_0_DequantizeLinear_Output_const_2D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_fc_quantize_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _fc_quantize_Mul_output_0_layer, 201,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_fc_quantize_Mul_output_0_chain,
  NULL, &_fc_quantize_Mul_output_0_0_0__fc_quantize_Round_output_0_conversion_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _fc_Gemm_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Reshape_output_0_to_chlast_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_fc_Gemm_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_fc_Gemm_output_0_weights, &_fc_Gemm_output_0_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_fc_Gemm_output_0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  _fc_Gemm_output_0_layer, 198,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &_fc_Gemm_output_0_chain,
  NULL, &_fc_quantize_Mul_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Reshape_output_0_to_chlast_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv4_Mul_2_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Reshape_output_0_to_chlast_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _Reshape_output_0_to_chlast_layer, 195,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &_Reshape_output_0_to_chlast_chain,
  NULL, &_fc_Gemm_output_0_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_WIDTH, AI_SHAPE_HEIGHT, AI_SHAPE_CHANNEL, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _kws_conv4_Mul_2_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_kws_conv4_quantize_Div_output_0_0_0__kws_conv4_Mul_2_output_0_conversion_output, &_kws_conv4_Pow_output_0_DequantizeLinear_Output_const_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv4_Mul_2_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _kws_conv4_Mul_2_output_0_layer, 192,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_kws_conv4_Mul_2_output_0_chain,
  NULL, &_Reshape_output_0_to_chlast_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _kws_conv4_quantize_Div_output_0_0_0__kws_conv4_Mul_2_output_0_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv4_quantize_Div_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv4_quantize_Div_output_0_0_0__kws_conv4_Mul_2_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _kws_conv4_quantize_Div_output_0_0_0__kws_conv4_Mul_2_output_0_conversion_layer, 189,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &_kws_conv4_quantize_Div_output_0_0_0__kws_conv4_Mul_2_output_0_conversion_chain,
  NULL, &_kws_conv4_Mul_2_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _kws_conv4_quantize_Div_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_kws_conv4_quantize_Round_output_0_output, &_kws_conv4_quantize_Constant_1_output_0_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv4_quantize_Div_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _kws_conv4_quantize_Div_output_0_layer, 189,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_kws_conv4_quantize_Div_output_0_chain,
  NULL, &_kws_conv4_quantize_Div_output_0_0_0__kws_conv4_Mul_2_output_0_conversion_layer, AI_STATIC, 
  .operation = ai_div_f32, 
  .buffer_operation = ai_div_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _kws_conv4_quantize_Round_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv4_quantize_Mul_output_0_0_0__kws_conv4_quantize_Round_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv4_quantize_Round_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _kws_conv4_quantize_Round_output_0_layer, 188,
  NL_TYPE, 0x0, NULL,
  nl, forward_round,
  &_kws_conv4_quantize_Round_output_0_chain,
  NULL, &_kws_conv4_quantize_Div_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _kws_conv4_quantize_Mul_output_0_0_0__kws_conv4_quantize_Round_output_0_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv4_quantize_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv4_quantize_Mul_output_0_0_0__kws_conv4_quantize_Round_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _kws_conv4_quantize_Mul_output_0_0_0__kws_conv4_quantize_Round_output_0_conversion_layer, 185,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &_kws_conv4_quantize_Mul_output_0_0_0__kws_conv4_quantize_Round_output_0_conversion_chain,
  NULL, &_kws_conv4_quantize_Round_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _kws_conv4_quantize_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_kws_conv4_activate_Relu_output_0_output, &_kws_conv4_quantize_Constant_output_0_DequantizeLinear_Output_const_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv4_quantize_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _kws_conv4_quantize_Mul_output_0_layer, 185,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_kws_conv4_quantize_Mul_output_0_chain,
  NULL, &_kws_conv4_quantize_Mul_output_0_0_0__kws_conv4_quantize_Round_output_0_conversion_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _kws_conv4_activate_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_kws_conv4_Conv_output_0_output, &_kws_conv4_calc_out_scale_Pow_output_0_DequantizeLinear_Output_const_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv4_activate_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _kws_conv4_activate_Relu_output_0_layer, 182,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_kws_conv4_activate_Relu_output_0_chain,
  NULL, &_kws_conv4_quantize_Mul_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _kws_conv4_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv4_Conv_output_0_pad_before_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv4_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_kws_conv4_Conv_output_0_weights, &_kws_conv4_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv4_Conv_output_0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  _kws_conv4_Conv_output_0_layer, 179,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_deep_sssa8_ch,
  &_kws_conv4_Conv_output_0_chain,
  NULL, &_kws_conv4_activate_Relu_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_i8 _kws_conv4_Conv_output_0_pad_before_value_data[] = { -128 };
AI_ARRAY_OBJ_DECLARE(
    _kws_conv4_Conv_output_0_pad_before_value, AI_ARRAY_FORMAT_S8,
    _kws_conv4_Conv_output_0_pad_before_value_data, _kws_conv4_Conv_output_0_pad_before_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  _kws_conv4_Conv_output_0_pad_before_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv4_pool_MaxPool_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv4_Conv_output_0_pad_before_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _kws_conv4_Conv_output_0_pad_before_layer, 179,
  PAD_TYPE, 0x0, NULL,
  pad, forward_pad,
  &_kws_conv4_Conv_output_0_pad_before_chain,
  NULL, &_kws_conv4_Conv_output_0_layer, AI_STATIC, 
  .value = &_kws_conv4_Conv_output_0_pad_before_value, 
  .mode = AI_PAD_CONSTANT, 
  .pads = AI_SHAPE_INIT(4, 1, 0, 1, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _kws_conv4_pool_MaxPool_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv3_Mul_2_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv4_pool_MaxPool_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _kws_conv4_pool_MaxPool_output_0_layer, 176,
  POOL_TYPE, 0x0, NULL,
  pool, forward_mp_integer_INT8,
  &_kws_conv4_pool_MaxPool_output_0_chain,
  NULL, &_kws_conv4_Conv_output_0_pad_before_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(1, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(1, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _kws_conv3_Mul_2_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_kws_conv3_quantize_Div_output_0_0_0__kws_conv3_Mul_2_output_0_conversion_output, &_kws_conv3_Pow_output_0_DequantizeLinear_Output_const_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv3_Mul_2_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _kws_conv3_Mul_2_output_0_layer, 173,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_kws_conv3_Mul_2_output_0_chain,
  NULL, &_kws_conv4_pool_MaxPool_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _kws_conv3_quantize_Div_output_0_0_0__kws_conv3_Mul_2_output_0_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv3_quantize_Div_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv3_quantize_Div_output_0_0_0__kws_conv3_Mul_2_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _kws_conv3_quantize_Div_output_0_0_0__kws_conv3_Mul_2_output_0_conversion_layer, 170,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &_kws_conv3_quantize_Div_output_0_0_0__kws_conv3_Mul_2_output_0_conversion_chain,
  NULL, &_kws_conv3_Mul_2_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _kws_conv3_quantize_Div_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_kws_conv3_quantize_Round_output_0_output, &_kws_conv3_quantize_Constant_1_output_0_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv3_quantize_Div_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _kws_conv3_quantize_Div_output_0_layer, 170,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_kws_conv3_quantize_Div_output_0_chain,
  NULL, &_kws_conv3_quantize_Div_output_0_0_0__kws_conv3_Mul_2_output_0_conversion_layer, AI_STATIC, 
  .operation = ai_div_f32, 
  .buffer_operation = ai_div_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _kws_conv3_quantize_Round_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv3_quantize_Mul_output_0_0_0__kws_conv3_quantize_Round_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv3_quantize_Round_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _kws_conv3_quantize_Round_output_0_layer, 169,
  NL_TYPE, 0x0, NULL,
  nl, forward_round,
  &_kws_conv3_quantize_Round_output_0_chain,
  NULL, &_kws_conv3_quantize_Div_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _kws_conv3_quantize_Mul_output_0_0_0__kws_conv3_quantize_Round_output_0_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv3_quantize_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv3_quantize_Mul_output_0_0_0__kws_conv3_quantize_Round_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _kws_conv3_quantize_Mul_output_0_0_0__kws_conv3_quantize_Round_output_0_conversion_layer, 166,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &_kws_conv3_quantize_Mul_output_0_0_0__kws_conv3_quantize_Round_output_0_conversion_chain,
  NULL, &_kws_conv3_quantize_Round_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _kws_conv3_quantize_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_kws_conv3_activate_Relu_output_0_output, &_kws_conv3_quantize_Constant_output_0_DequantizeLinear_Output_const_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv3_quantize_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _kws_conv3_quantize_Mul_output_0_layer, 166,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_kws_conv3_quantize_Mul_output_0_chain,
  NULL, &_kws_conv3_quantize_Mul_output_0_0_0__kws_conv3_quantize_Round_output_0_conversion_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _kws_conv3_activate_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_kws_conv3_Conv_output_0_output, &_kws_conv3_calc_out_scale_Pow_output_0_DequantizeLinear_Output_const_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv3_activate_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _kws_conv3_activate_Relu_output_0_layer, 163,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_kws_conv3_activate_Relu_output_0_chain,
  NULL, &_kws_conv3_quantize_Mul_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _kws_conv3_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv3_Conv_output_0_pad_before_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv3_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_kws_conv3_Conv_output_0_weights, &_kws_conv3_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv3_Conv_output_0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  _kws_conv3_Conv_output_0_layer, 160,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_deep_sssa8_ch,
  &_kws_conv3_Conv_output_0_chain,
  NULL, &_kws_conv3_activate_Relu_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_i8 _kws_conv3_Conv_output_0_pad_before_value_data[] = { -128 };
AI_ARRAY_OBJ_DECLARE(
    _kws_conv3_Conv_output_0_pad_before_value, AI_ARRAY_FORMAT_S8,
    _kws_conv3_Conv_output_0_pad_before_value_data, _kws_conv3_Conv_output_0_pad_before_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  _kws_conv3_Conv_output_0_pad_before_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv3_quantize_pool_Div_output_0_0_0__kws_conv3_Conv_output_0_pad_before_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv3_Conv_output_0_pad_before_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _kws_conv3_Conv_output_0_pad_before_layer, 160,
  PAD_TYPE, 0x0, NULL,
  pad, forward_pad,
  &_kws_conv3_Conv_output_0_pad_before_chain,
  NULL, &_kws_conv3_Conv_output_0_layer, AI_STATIC, 
  .value = &_kws_conv3_Conv_output_0_pad_before_value, 
  .mode = AI_PAD_CONSTANT, 
  .pads = AI_SHAPE_INIT(4, 1, 0, 1, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _kws_conv3_quantize_pool_Div_output_0_0_0__kws_conv3_Conv_output_0_pad_before_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv3_quantize_pool_Div_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv3_quantize_pool_Div_output_0_0_0__kws_conv3_Conv_output_0_pad_before_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _kws_conv3_quantize_pool_Div_output_0_0_0__kws_conv3_Conv_output_0_pad_before_conversion_layer, 157,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &_kws_conv3_quantize_pool_Div_output_0_0_0__kws_conv3_Conv_output_0_pad_before_conversion_chain,
  NULL, &_kws_conv3_Conv_output_0_pad_before_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _kws_conv3_quantize_pool_Div_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_kws_conv3_quantize_pool_Floor_output_0_output, &_kws_conv3_quantize_pool_Constant_1_output_0_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv3_quantize_pool_Div_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _kws_conv3_quantize_pool_Div_output_0_layer, 157,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_kws_conv3_quantize_pool_Div_output_0_chain,
  NULL, &_kws_conv3_quantize_pool_Div_output_0_0_0__kws_conv3_Conv_output_0_pad_before_conversion_layer, AI_STATIC, 
  .operation = ai_div_f32, 
  .buffer_operation = ai_div_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _kws_conv3_quantize_pool_Floor_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv3_quantize_pool_Mul_output_0_0_0__kws_conv3_quantize_pool_Floor_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv3_quantize_pool_Floor_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _kws_conv3_quantize_pool_Floor_output_0_layer, 156,
  NL_TYPE, 0x0, NULL,
  nl, forward_floor,
  &_kws_conv3_quantize_pool_Floor_output_0_chain,
  NULL, &_kws_conv3_quantize_pool_Div_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _kws_conv3_quantize_pool_Mul_output_0_0_0__kws_conv3_quantize_pool_Floor_output_0_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv3_quantize_pool_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv3_quantize_pool_Mul_output_0_0_0__kws_conv3_quantize_pool_Floor_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _kws_conv3_quantize_pool_Mul_output_0_0_0__kws_conv3_quantize_pool_Floor_output_0_conversion_layer, 153,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &_kws_conv3_quantize_pool_Mul_output_0_0_0__kws_conv3_quantize_pool_Floor_output_0_conversion_chain,
  NULL, &_kws_conv3_quantize_pool_Floor_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _kws_conv3_quantize_pool_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_kws_conv3_pool_AveragePool_output_0_output, &_kws_conv3_quantize_pool_Constant_output_0_DequantizeLinear_Output_const_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv3_quantize_pool_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _kws_conv3_quantize_pool_Mul_output_0_layer, 153,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_kws_conv3_quantize_pool_Mul_output_0_chain,
  NULL, &_kws_conv3_quantize_pool_Mul_output_0_0_0__kws_conv3_quantize_pool_Floor_output_0_conversion_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _kws_conv3_pool_AveragePool_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv2_Mul_2_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv3_pool_AveragePool_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _kws_conv3_pool_AveragePool_output_0_layer, 150,
  POOL_TYPE, 0x0, NULL,
  pool, forward_ap_integer_INT8,
  &_kws_conv3_pool_AveragePool_output_0_chain,
  NULL, &_kws_conv3_quantize_pool_Mul_output_0_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(1, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(1, 2), 
  .count_include_pad = 1, 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _kws_conv2_Mul_2_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_kws_conv2_quantize_Div_output_0_0_0__kws_conv2_Mul_2_output_0_conversion_output, &_kws_conv2_Pow_output_0_DequantizeLinear_Output_const_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv2_Mul_2_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _kws_conv2_Mul_2_output_0_layer, 147,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_kws_conv2_Mul_2_output_0_chain,
  NULL, &_kws_conv3_pool_AveragePool_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _kws_conv2_quantize_Div_output_0_0_0__kws_conv2_Mul_2_output_0_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv2_quantize_Div_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv2_quantize_Div_output_0_0_0__kws_conv2_Mul_2_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _kws_conv2_quantize_Div_output_0_0_0__kws_conv2_Mul_2_output_0_conversion_layer, 144,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &_kws_conv2_quantize_Div_output_0_0_0__kws_conv2_Mul_2_output_0_conversion_chain,
  NULL, &_kws_conv2_Mul_2_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _kws_conv2_quantize_Div_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_kws_conv2_quantize_Round_output_0_output, &_kws_conv2_quantize_Constant_1_output_0_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv2_quantize_Div_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _kws_conv2_quantize_Div_output_0_layer, 144,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_kws_conv2_quantize_Div_output_0_chain,
  NULL, &_kws_conv2_quantize_Div_output_0_0_0__kws_conv2_Mul_2_output_0_conversion_layer, AI_STATIC, 
  .operation = ai_div_f32, 
  .buffer_operation = ai_div_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _kws_conv2_quantize_Round_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv2_quantize_Mul_output_0_0_0__kws_conv2_quantize_Round_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv2_quantize_Round_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _kws_conv2_quantize_Round_output_0_layer, 143,
  NL_TYPE, 0x0, NULL,
  nl, forward_round,
  &_kws_conv2_quantize_Round_output_0_chain,
  NULL, &_kws_conv2_quantize_Div_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _kws_conv2_quantize_Mul_output_0_0_0__kws_conv2_quantize_Round_output_0_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv2_quantize_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv2_quantize_Mul_output_0_0_0__kws_conv2_quantize_Round_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _kws_conv2_quantize_Mul_output_0_0_0__kws_conv2_quantize_Round_output_0_conversion_layer, 140,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &_kws_conv2_quantize_Mul_output_0_0_0__kws_conv2_quantize_Round_output_0_conversion_chain,
  NULL, &_kws_conv2_quantize_Round_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _kws_conv2_quantize_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_kws_conv2_activate_Relu_output_0_output, &_kws_conv2_quantize_Constant_output_0_DequantizeLinear_Output_const_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv2_quantize_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _kws_conv2_quantize_Mul_output_0_layer, 140,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_kws_conv2_quantize_Mul_output_0_chain,
  NULL, &_kws_conv2_quantize_Mul_output_0_0_0__kws_conv2_quantize_Round_output_0_conversion_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _kws_conv2_activate_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_kws_conv2_Conv_output_0_output, &_kws_conv2_calc_out_scale_Pow_output_0_DequantizeLinear_Output_const_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv2_activate_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _kws_conv2_activate_Relu_output_0_layer, 137,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_kws_conv2_activate_Relu_output_0_chain,
  NULL, &_kws_conv2_quantize_Mul_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _kws_conv2_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv1_Mul_2_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv2_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_kws_conv2_Conv_output_0_weights, &_kws_conv2_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv2_Conv_output_0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  _kws_conv2_Conv_output_0_layer, 134,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_deep_sssa8_ch,
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
  _kws_conv1_Mul_2_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_kws_conv1_quantize_Div_output_0_0_0__kws_conv1_Mul_2_output_0_conversion_output, &_kws_conv1_Pow_output_0_DequantizeLinear_Output_const_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv1_Mul_2_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _kws_conv1_Mul_2_output_0_layer, 131,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_kws_conv1_Mul_2_output_0_chain,
  NULL, &_kws_conv2_Conv_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _kws_conv1_quantize_Div_output_0_0_0__kws_conv1_Mul_2_output_0_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv1_quantize_Div_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv1_quantize_Div_output_0_0_0__kws_conv1_Mul_2_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _kws_conv1_quantize_Div_output_0_0_0__kws_conv1_Mul_2_output_0_conversion_layer, 128,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &_kws_conv1_quantize_Div_output_0_0_0__kws_conv1_Mul_2_output_0_conversion_chain,
  NULL, &_kws_conv1_Mul_2_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _kws_conv1_quantize_Div_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_kws_conv1_quantize_Round_output_0_output, &_kws_conv1_quantize_Constant_1_output_0_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv1_quantize_Div_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _kws_conv1_quantize_Div_output_0_layer, 128,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_kws_conv1_quantize_Div_output_0_chain,
  NULL, &_kws_conv1_quantize_Div_output_0_0_0__kws_conv1_Mul_2_output_0_conversion_layer, AI_STATIC, 
  .operation = ai_div_f32, 
  .buffer_operation = ai_div_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _kws_conv1_quantize_Round_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv1_quantize_Mul_output_0_0_0__kws_conv1_quantize_Round_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv1_quantize_Round_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _kws_conv1_quantize_Round_output_0_layer, 127,
  NL_TYPE, 0x0, NULL,
  nl, forward_round,
  &_kws_conv1_quantize_Round_output_0_chain,
  NULL, &_kws_conv1_quantize_Div_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _kws_conv1_quantize_Mul_output_0_0_0__kws_conv1_quantize_Round_output_0_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv1_quantize_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv1_quantize_Mul_output_0_0_0__kws_conv1_quantize_Round_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _kws_conv1_quantize_Mul_output_0_0_0__kws_conv1_quantize_Round_output_0_conversion_layer, 124,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &_kws_conv1_quantize_Mul_output_0_0_0__kws_conv1_quantize_Round_output_0_conversion_chain,
  NULL, &_kws_conv1_quantize_Round_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _kws_conv1_quantize_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_kws_conv1_activate_Relu_output_0_output, &_kws_conv1_quantize_Constant_output_0_DequantizeLinear_Output_const_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv1_quantize_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _kws_conv1_quantize_Mul_output_0_layer, 124,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_kws_conv1_quantize_Mul_output_0_chain,
  NULL, &_kws_conv1_quantize_Mul_output_0_0_0__kws_conv1_quantize_Round_output_0_conversion_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _kws_conv1_activate_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_kws_conv1_Conv_output_0_output, &_kws_conv1_calc_out_scale_Pow_output_0_DequantizeLinear_Output_const_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv1_activate_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _kws_conv1_activate_Relu_output_0_layer, 121,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_kws_conv1_activate_Relu_output_0_chain,
  NULL, &_kws_conv1_quantize_Mul_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _kws_conv1_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv1_Conv_output_0_pad_before_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv1_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_kws_conv1_Conv_output_0_weights, &_kws_conv1_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv1_Conv_output_0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  _kws_conv1_Conv_output_0_layer, 118,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_deep_sssa8_ch,
  &_kws_conv1_Conv_output_0_chain,
  NULL, &_kws_conv1_activate_Relu_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_i8 _kws_conv1_Conv_output_0_pad_before_value_data[] = { -128 };
AI_ARRAY_OBJ_DECLARE(
    _kws_conv1_Conv_output_0_pad_before_value, AI_ARRAY_FORMAT_S8,
    _kws_conv1_Conv_output_0_pad_before_value_data, _kws_conv1_Conv_output_0_pad_before_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  _kws_conv1_Conv_output_0_pad_before_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv1_pool_MaxPool_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv1_Conv_output_0_pad_before_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _kws_conv1_Conv_output_0_pad_before_layer, 118,
  PAD_TYPE, 0x0, NULL,
  pad, forward_pad,
  &_kws_conv1_Conv_output_0_pad_before_chain,
  NULL, &_kws_conv1_Conv_output_0_layer, AI_STATIC, 
  .value = &_kws_conv1_Conv_output_0_pad_before_value, 
  .mode = AI_PAD_CONSTANT, 
  .pads = AI_SHAPE_INIT(4, 1, 0, 1, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _kws_conv1_pool_MaxPool_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv4_Mul_2_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_kws_conv1_pool_MaxPool_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _kws_conv1_pool_MaxPool_output_0_layer, 115,
  POOL_TYPE, 0x0, NULL,
  pool, forward_mp_integer_INT8,
  &_kws_conv1_pool_MaxPool_output_0_chain,
  NULL, &_kws_conv1_Conv_output_0_pad_before_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(1, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(1, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _voice_conv4_Mul_2_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_voice_conv4_quantize_Div_output_0_0_0__voice_conv4_Mul_2_output_0_conversion_output, &_voice_conv4_Pow_output_0_DequantizeLinear_Output_const_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv4_Mul_2_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _voice_conv4_Mul_2_output_0_layer, 112,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_voice_conv4_Mul_2_output_0_chain,
  NULL, &_kws_conv1_pool_MaxPool_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _voice_conv4_quantize_Div_output_0_0_0__voice_conv4_Mul_2_output_0_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv4_quantize_Div_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv4_quantize_Div_output_0_0_0__voice_conv4_Mul_2_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _voice_conv4_quantize_Div_output_0_0_0__voice_conv4_Mul_2_output_0_conversion_layer, 109,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &_voice_conv4_quantize_Div_output_0_0_0__voice_conv4_Mul_2_output_0_conversion_chain,
  NULL, &_voice_conv4_Mul_2_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _voice_conv4_quantize_Div_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_voice_conv4_quantize_Round_output_0_output, &_voice_conv4_quantize_Constant_1_output_0_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv4_quantize_Div_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _voice_conv4_quantize_Div_output_0_layer, 109,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_voice_conv4_quantize_Div_output_0_chain,
  NULL, &_voice_conv4_quantize_Div_output_0_0_0__voice_conv4_Mul_2_output_0_conversion_layer, AI_STATIC, 
  .operation = ai_div_f32, 
  .buffer_operation = ai_div_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _voice_conv4_quantize_Round_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv4_quantize_Mul_output_0_0_0__voice_conv4_quantize_Round_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv4_quantize_Round_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _voice_conv4_quantize_Round_output_0_layer, 108,
  NL_TYPE, 0x0, NULL,
  nl, forward_round,
  &_voice_conv4_quantize_Round_output_0_chain,
  NULL, &_voice_conv4_quantize_Div_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _voice_conv4_quantize_Mul_output_0_0_0__voice_conv4_quantize_Round_output_0_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv4_quantize_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv4_quantize_Mul_output_0_0_0__voice_conv4_quantize_Round_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _voice_conv4_quantize_Mul_output_0_0_0__voice_conv4_quantize_Round_output_0_conversion_layer, 105,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &_voice_conv4_quantize_Mul_output_0_0_0__voice_conv4_quantize_Round_output_0_conversion_chain,
  NULL, &_voice_conv4_quantize_Round_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _voice_conv4_quantize_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_voice_conv4_activate_Relu_output_0_output, &_voice_conv4_quantize_Constant_output_0_DequantizeLinear_Output_const_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv4_quantize_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _voice_conv4_quantize_Mul_output_0_layer, 105,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_voice_conv4_quantize_Mul_output_0_chain,
  NULL, &_voice_conv4_quantize_Mul_output_0_0_0__voice_conv4_quantize_Round_output_0_conversion_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _voice_conv4_activate_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_voice_conv4_Conv_output_0_output, &_voice_conv4_calc_out_scale_Pow_output_0_DequantizeLinear_Output_const_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv4_activate_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _voice_conv4_activate_Relu_output_0_layer, 102,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_voice_conv4_activate_Relu_output_0_chain,
  NULL, &_voice_conv4_quantize_Mul_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _voice_conv4_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv3_Mul_2_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv4_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_voice_conv4_Conv_output_0_weights, &_voice_conv4_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv4_Conv_output_0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  _voice_conv4_Conv_output_0_layer, 99,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_deep_sssa8_ch,
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
  _voice_conv3_Mul_2_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_voice_conv3_quantize_Div_output_0_0_0__voice_conv3_Mul_2_output_0_conversion_output, &_voice_conv3_Pow_output_0_DequantizeLinear_Output_const_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv3_Mul_2_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _voice_conv3_Mul_2_output_0_layer, 96,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_voice_conv3_Mul_2_output_0_chain,
  NULL, &_voice_conv4_Conv_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _voice_conv3_quantize_Div_output_0_0_0__voice_conv3_Mul_2_output_0_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv3_quantize_Div_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv3_quantize_Div_output_0_0_0__voice_conv3_Mul_2_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _voice_conv3_quantize_Div_output_0_0_0__voice_conv3_Mul_2_output_0_conversion_layer, 93,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &_voice_conv3_quantize_Div_output_0_0_0__voice_conv3_Mul_2_output_0_conversion_chain,
  NULL, &_voice_conv3_Mul_2_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _voice_conv3_quantize_Div_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_voice_conv3_quantize_Round_output_0_output, &_voice_conv3_quantize_Constant_1_output_0_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv3_quantize_Div_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _voice_conv3_quantize_Div_output_0_layer, 93,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_voice_conv3_quantize_Div_output_0_chain,
  NULL, &_voice_conv3_quantize_Div_output_0_0_0__voice_conv3_Mul_2_output_0_conversion_layer, AI_STATIC, 
  .operation = ai_div_f32, 
  .buffer_operation = ai_div_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _voice_conv3_quantize_Round_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv3_quantize_Mul_output_0_0_0__voice_conv3_quantize_Round_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv3_quantize_Round_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _voice_conv3_quantize_Round_output_0_layer, 92,
  NL_TYPE, 0x0, NULL,
  nl, forward_round,
  &_voice_conv3_quantize_Round_output_0_chain,
  NULL, &_voice_conv3_quantize_Div_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _voice_conv3_quantize_Mul_output_0_0_0__voice_conv3_quantize_Round_output_0_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv3_quantize_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv3_quantize_Mul_output_0_0_0__voice_conv3_quantize_Round_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _voice_conv3_quantize_Mul_output_0_0_0__voice_conv3_quantize_Round_output_0_conversion_layer, 89,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &_voice_conv3_quantize_Mul_output_0_0_0__voice_conv3_quantize_Round_output_0_conversion_chain,
  NULL, &_voice_conv3_quantize_Round_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _voice_conv3_quantize_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_voice_conv3_activate_Relu_output_0_output, &_voice_conv3_quantize_Constant_output_0_DequantizeLinear_Output_const_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv3_quantize_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _voice_conv3_quantize_Mul_output_0_layer, 89,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_voice_conv3_quantize_Mul_output_0_chain,
  NULL, &_voice_conv3_quantize_Mul_output_0_0_0__voice_conv3_quantize_Round_output_0_conversion_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _voice_conv3_activate_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_voice_conv3_Conv_output_0_output, &_voice_conv3_calc_out_scale_Pow_output_0_DequantizeLinear_Output_const_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv3_activate_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _voice_conv3_activate_Relu_output_0_layer, 86,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_voice_conv3_activate_Relu_output_0_chain,
  NULL, &_voice_conv3_quantize_Mul_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _voice_conv3_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv3_Conv_output_0_pad_before_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv3_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_voice_conv3_Conv_output_0_weights, &_voice_conv3_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv3_Conv_output_0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  _voice_conv3_Conv_output_0_layer, 83,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_deep_sssa8_ch,
  &_voice_conv3_Conv_output_0_chain,
  NULL, &_voice_conv3_activate_Relu_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_i8 _voice_conv3_Conv_output_0_pad_before_value_data[] = { -128 };
AI_ARRAY_OBJ_DECLARE(
    _voice_conv3_Conv_output_0_pad_before_value, AI_ARRAY_FORMAT_S8,
    _voice_conv3_Conv_output_0_pad_before_value_data, _voice_conv3_Conv_output_0_pad_before_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  _voice_conv3_Conv_output_0_pad_before_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv3_pool_MaxPool_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv3_Conv_output_0_pad_before_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _voice_conv3_Conv_output_0_pad_before_layer, 83,
  PAD_TYPE, 0x0, NULL,
  pad, forward_pad,
  &_voice_conv3_Conv_output_0_pad_before_chain,
  NULL, &_voice_conv3_Conv_output_0_layer, AI_STATIC, 
  .value = &_voice_conv3_Conv_output_0_pad_before_value, 
  .mode = AI_PAD_CONSTANT, 
  .pads = AI_SHAPE_INIT(4, 1, 0, 1, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _voice_conv3_pool_MaxPool_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv2_Mul_2_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv3_pool_MaxPool_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _voice_conv3_pool_MaxPool_output_0_layer, 80,
  POOL_TYPE, 0x0, NULL,
  pool, forward_mp_integer_INT8,
  &_voice_conv3_pool_MaxPool_output_0_chain,
  NULL, &_voice_conv3_Conv_output_0_pad_before_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(1, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(1, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _voice_conv2_Mul_2_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_voice_conv2_quantize_Div_output_0_0_0__voice_conv2_Mul_2_output_0_conversion_output, &_voice_conv2_Pow_output_0_DequantizeLinear_Output_const_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv2_Mul_2_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _voice_conv2_Mul_2_output_0_layer, 77,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_voice_conv2_Mul_2_output_0_chain,
  NULL, &_voice_conv3_pool_MaxPool_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _voice_conv2_quantize_Div_output_0_0_0__voice_conv2_Mul_2_output_0_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv2_quantize_Div_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv2_quantize_Div_output_0_0_0__voice_conv2_Mul_2_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _voice_conv2_quantize_Div_output_0_0_0__voice_conv2_Mul_2_output_0_conversion_layer, 74,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &_voice_conv2_quantize_Div_output_0_0_0__voice_conv2_Mul_2_output_0_conversion_chain,
  NULL, &_voice_conv2_Mul_2_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _voice_conv2_quantize_Div_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_voice_conv2_quantize_Round_output_0_output, &_voice_conv2_quantize_Constant_1_output_0_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv2_quantize_Div_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _voice_conv2_quantize_Div_output_0_layer, 74,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_voice_conv2_quantize_Div_output_0_chain,
  NULL, &_voice_conv2_quantize_Div_output_0_0_0__voice_conv2_Mul_2_output_0_conversion_layer, AI_STATIC, 
  .operation = ai_div_f32, 
  .buffer_operation = ai_div_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _voice_conv2_quantize_Round_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv2_quantize_Mul_output_0_0_0__voice_conv2_quantize_Round_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv2_quantize_Round_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _voice_conv2_quantize_Round_output_0_layer, 73,
  NL_TYPE, 0x0, NULL,
  nl, forward_round,
  &_voice_conv2_quantize_Round_output_0_chain,
  NULL, &_voice_conv2_quantize_Div_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _voice_conv2_quantize_Mul_output_0_0_0__voice_conv2_quantize_Round_output_0_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv2_quantize_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv2_quantize_Mul_output_0_0_0__voice_conv2_quantize_Round_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _voice_conv2_quantize_Mul_output_0_0_0__voice_conv2_quantize_Round_output_0_conversion_layer, 70,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &_voice_conv2_quantize_Mul_output_0_0_0__voice_conv2_quantize_Round_output_0_conversion_chain,
  NULL, &_voice_conv2_quantize_Round_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _voice_conv2_quantize_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_voice_conv2_activate_Relu_output_0_output, &_voice_conv2_quantize_Constant_output_0_DequantizeLinear_Output_const_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv2_quantize_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _voice_conv2_quantize_Mul_output_0_layer, 70,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_voice_conv2_quantize_Mul_output_0_chain,
  NULL, &_voice_conv2_quantize_Mul_output_0_0_0__voice_conv2_quantize_Round_output_0_conversion_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _voice_conv2_activate_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_voice_conv2_Conv_output_0_output, &_voice_conv2_calc_out_scale_Pow_output_0_DequantizeLinear_Output_const_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv2_activate_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _voice_conv2_activate_Relu_output_0_layer, 67,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_voice_conv2_activate_Relu_output_0_chain,
  NULL, &_voice_conv2_quantize_Mul_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _voice_conv2_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv1_Mul_2_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv2_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_voice_conv2_Conv_output_0_weights, &_voice_conv2_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv2_Conv_output_0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  _voice_conv2_Conv_output_0_layer, 64,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_deep_sssa8_ch,
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
  _voice_conv1_Mul_2_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_voice_conv1_quantize_Div_output_0_0_0__voice_conv1_Mul_2_output_0_conversion_output, &_voice_conv1_Pow_output_0_DequantizeLinear_Output_const_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv1_Mul_2_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _voice_conv1_Mul_2_output_0_layer, 61,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_voice_conv1_Mul_2_output_0_chain,
  NULL, &_voice_conv2_Conv_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _voice_conv1_quantize_Div_output_0_0_0__voice_conv1_Mul_2_output_0_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv1_quantize_Div_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv1_quantize_Div_output_0_0_0__voice_conv1_Mul_2_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _voice_conv1_quantize_Div_output_0_0_0__voice_conv1_Mul_2_output_0_conversion_layer, 58,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &_voice_conv1_quantize_Div_output_0_0_0__voice_conv1_Mul_2_output_0_conversion_chain,
  NULL, &_voice_conv1_Mul_2_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _voice_conv1_quantize_Div_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_voice_conv1_quantize_Round_output_0_output, &_voice_conv1_quantize_Constant_1_output_0_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv1_quantize_Div_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _voice_conv1_quantize_Div_output_0_layer, 58,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_voice_conv1_quantize_Div_output_0_chain,
  NULL, &_voice_conv1_quantize_Div_output_0_0_0__voice_conv1_Mul_2_output_0_conversion_layer, AI_STATIC, 
  .operation = ai_div_f32, 
  .buffer_operation = ai_div_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _voice_conv1_quantize_Round_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv1_quantize_Mul_output_0_0_0__voice_conv1_quantize_Round_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv1_quantize_Round_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _voice_conv1_quantize_Round_output_0_layer, 57,
  NL_TYPE, 0x0, NULL,
  nl, forward_round,
  &_voice_conv1_quantize_Round_output_0_chain,
  NULL, &_voice_conv1_quantize_Div_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _voice_conv1_quantize_Mul_output_0_0_0__voice_conv1_quantize_Round_output_0_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv1_quantize_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv1_quantize_Mul_output_0_0_0__voice_conv1_quantize_Round_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _voice_conv1_quantize_Mul_output_0_0_0__voice_conv1_quantize_Round_output_0_conversion_layer, 54,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &_voice_conv1_quantize_Mul_output_0_0_0__voice_conv1_quantize_Round_output_0_conversion_chain,
  NULL, &_voice_conv1_quantize_Round_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _voice_conv1_quantize_Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_voice_conv1_activate_Relu_output_0_output, &_voice_conv1_quantize_Constant_output_0_DequantizeLinear_Output_const_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv1_quantize_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _voice_conv1_quantize_Mul_output_0_layer, 54,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_voice_conv1_quantize_Mul_output_0_chain,
  NULL, &_voice_conv1_quantize_Mul_output_0_0_0__voice_conv1_quantize_Round_output_0_conversion_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _voice_conv1_activate_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_voice_conv1_Conv_output_0_output, &_voice_conv1_calc_out_scale_Pow_output_0_DequantizeLinear_Output_const_4D),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv1_activate_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _voice_conv1_activate_Relu_output_0_layer, 51,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &_voice_conv1_activate_Relu_output_0_chain,
  NULL, &_voice_conv1_quantize_Mul_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _voice_conv1_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &kws_input_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv1_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_voice_conv1_Conv_output_0_weights, &_voice_conv1_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_voice_conv1_Conv_output_0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  _voice_conv1_Conv_output_0_layer, 48,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_sssa8_ch,
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
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 172232, 1, 1),
    172232, NULL, NULL),
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 51200, 1, 1),
    51200, NULL, NULL),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_IN_NUM, &kws_input_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_OUT_NUM, &kws_logits_QuantizeLinear_Input_output),
  &_voice_conv1_Conv_output_0_layer, 0xc3398bbd, NULL)

#else

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 172232, 1, 1),
      172232, NULL, NULL)
  ),
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 51200, 1, 1),
      51200, NULL, NULL)
  ),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_IN_NUM, &kws_input_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_OUT_NUM, &kws_logits_QuantizeLinear_Input_output),
  &_voice_conv1_Conv_output_0_layer, 0xc3398bbd, NULL)

#endif	/*(AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)*/



/******************************************************************************/
AI_DECLARE_STATIC
ai_bool network_configure_activations(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_activations_map(g_network_activations_map, 1, params)) {
    /* Updating activations (byte) offsets */
    
    kws_input_output_array.data = AI_PTR(g_network_activations_map[0] + 9216);
    kws_input_output_array.data_start = AI_PTR(g_network_activations_map[0] + 9216);
    _voice_conv1_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 2184);
    _voice_conv1_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 2184);
    _voice_conv1_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 25600);
    _voice_conv1_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 25600);
    _voice_conv1_activate_Relu_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 25600);
    _voice_conv1_activate_Relu_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 25600);
    _voice_conv1_quantize_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 38400);
    _voice_conv1_quantize_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 38400);
    _voice_conv1_quantize_Mul_output_0_0_0__voice_conv1_quantize_Round_output_0_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _voice_conv1_quantize_Mul_output_0_0_0__voice_conv1_quantize_Round_output_0_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _voice_conv1_quantize_Round_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _voice_conv1_quantize_Round_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _voice_conv1_quantize_Div_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _voice_conv1_quantize_Div_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _voice_conv1_quantize_Div_output_0_0_0__voice_conv1_Mul_2_output_0_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _voice_conv1_quantize_Div_output_0_0_0__voice_conv1_Mul_2_output_0_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _voice_conv1_Mul_2_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 12800);
    _voice_conv1_Mul_2_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 12800);
    _voice_conv2_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _voice_conv2_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _voice_conv2_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 25600);
    _voice_conv2_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 25600);
    _voice_conv2_activate_Relu_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _voice_conv2_activate_Relu_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _voice_conv2_quantize_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 39104);
    _voice_conv2_quantize_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 39104);
    _voice_conv2_quantize_Mul_output_0_0_0__voice_conv2_quantize_Round_output_0_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 2816);
    _voice_conv2_quantize_Mul_output_0_0_0__voice_conv2_quantize_Round_output_0_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2816);
    _voice_conv2_quantize_Round_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 2816);
    _voice_conv2_quantize_Round_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2816);
    _voice_conv2_quantize_Div_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 2816);
    _voice_conv2_quantize_Div_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2816);
    _voice_conv2_quantize_Div_output_0_0_0__voice_conv2_Mul_2_output_0_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 2816);
    _voice_conv2_quantize_Div_output_0_0_0__voice_conv2_Mul_2_output_0_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2816);
    _voice_conv2_Mul_2_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 14912);
    _voice_conv2_Mul_2_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 14912);
    _voice_conv3_pool_MaxPool_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _voice_conv3_pool_MaxPool_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _voice_conv3_Conv_output_0_pad_before_output_array.data = AI_PTR(g_network_activations_map[0] + 6048);
    _voice_conv3_Conv_output_0_pad_before_output_array.data_start = AI_PTR(g_network_activations_map[0] + 6048);
    _voice_conv3_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 12288);
    _voice_conv3_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 12288);
    _voice_conv3_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _voice_conv3_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _voice_conv3_activate_Relu_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 4032);
    _voice_conv3_activate_Relu_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 4032);
    _voice_conv3_quantize_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _voice_conv3_quantize_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _voice_conv3_quantize_Mul_output_0_0_0__voice_conv3_quantize_Round_output_0_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 4032);
    _voice_conv3_quantize_Mul_output_0_0_0__voice_conv3_quantize_Round_output_0_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 4032);
    _voice_conv3_quantize_Round_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 20160);
    _voice_conv3_quantize_Round_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 20160);
    _voice_conv3_quantize_Div_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _voice_conv3_quantize_Div_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _voice_conv3_quantize_Div_output_0_0_0__voice_conv3_Mul_2_output_0_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 16128);
    _voice_conv3_quantize_Div_output_0_0_0__voice_conv3_Mul_2_output_0_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 16128);
    _voice_conv3_Mul_2_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _voice_conv3_Mul_2_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _voice_conv4_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 4032);
    _voice_conv4_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 4032);
    _voice_conv4_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 10592);
    _voice_conv4_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 10592);
    _voice_conv4_activate_Relu_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _voice_conv4_activate_Relu_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _voice_conv4_quantize_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 2928);
    _voice_conv4_quantize_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2928);
    _voice_conv4_quantize_Mul_output_0_0_0__voice_conv4_quantize_Round_output_0_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 5856);
    _voice_conv4_quantize_Mul_output_0_0_0__voice_conv4_quantize_Round_output_0_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 5856);
    _voice_conv4_quantize_Round_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 17568);
    _voice_conv4_quantize_Round_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 17568);
    _voice_conv4_quantize_Div_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _voice_conv4_quantize_Div_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _voice_conv4_quantize_Div_output_0_0_0__voice_conv4_Mul_2_output_0_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 11712);
    _voice_conv4_quantize_Div_output_0_0_0__voice_conv4_Mul_2_output_0_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 11712);
    _voice_conv4_Mul_2_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _voice_conv4_Mul_2_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _kws_conv1_pool_MaxPool_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 2928);
    _kws_conv1_pool_MaxPool_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2928);
    _kws_conv1_Conv_output_0_pad_before_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _kws_conv1_Conv_output_0_pad_before_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _kws_conv1_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 1536);
    _kws_conv1_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 1536);
    _kws_conv1_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 8128);
    _kws_conv1_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 8128);
    _kws_conv1_activate_Relu_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _kws_conv1_activate_Relu_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _kws_conv1_quantize_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 1920);
    _kws_conv1_quantize_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1920);
    _kws_conv1_quantize_Mul_output_0_0_0__kws_conv1_quantize_Round_output_0_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 3840);
    _kws_conv1_quantize_Mul_output_0_0_0__kws_conv1_quantize_Round_output_0_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3840);
    _kws_conv1_quantize_Round_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 11520);
    _kws_conv1_quantize_Round_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 11520);
    _kws_conv1_quantize_Div_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _kws_conv1_quantize_Div_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _kws_conv1_quantize_Div_output_0_0_0__kws_conv1_Mul_2_output_0_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 7680);
    _kws_conv1_quantize_Div_output_0_0_0__kws_conv1_Mul_2_output_0_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 7680);
    _kws_conv1_Mul_2_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _kws_conv1_Mul_2_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _kws_conv2_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 1920);
    _kws_conv2_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 1920);
    _kws_conv2_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 9152);
    _kws_conv2_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 9152);
    _kws_conv2_activate_Relu_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _kws_conv2_activate_Relu_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _kws_conv2_quantize_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 2688);
    _kws_conv2_quantize_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2688);
    _kws_conv2_quantize_Mul_output_0_0_0__kws_conv2_quantize_Round_output_0_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 5376);
    _kws_conv2_quantize_Mul_output_0_0_0__kws_conv2_quantize_Round_output_0_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 5376);
    _kws_conv2_quantize_Round_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 16128);
    _kws_conv2_quantize_Round_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 16128);
    _kws_conv2_quantize_Div_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _kws_conv2_quantize_Div_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _kws_conv2_quantize_Div_output_0_0_0__kws_conv2_Mul_2_output_0_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 10752);
    _kws_conv2_quantize_Div_output_0_0_0__kws_conv2_Mul_2_output_0_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 10752);
    _kws_conv2_Mul_2_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _kws_conv2_Mul_2_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _kws_conv3_pool_AveragePool_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 2688);
    _kws_conv3_pool_AveragePool_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2688);
    _kws_conv3_quantize_pool_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _kws_conv3_quantize_pool_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _kws_conv3_quantize_pool_Mul_output_0_0_0__kws_conv3_quantize_pool_Floor_output_0_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 1344);
    _kws_conv3_quantize_pool_Mul_output_0_0_0__kws_conv3_quantize_pool_Floor_output_0_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1344);
    _kws_conv3_quantize_pool_Floor_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 6720);
    _kws_conv3_quantize_pool_Floor_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 6720);
    _kws_conv3_quantize_pool_Div_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _kws_conv3_quantize_pool_Div_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _kws_conv3_quantize_pool_Div_output_0_0_0__kws_conv3_Conv_output_0_pad_before_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 5376);
    _kws_conv3_quantize_pool_Div_output_0_0_0__kws_conv3_Conv_output_0_pad_before_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 5376);
    _kws_conv3_Conv_output_0_pad_before_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _kws_conv3_Conv_output_0_pad_before_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _kws_conv3_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 1536);
    _kws_conv3_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 1536);
    _kws_conv3_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 9208);
    _kws_conv3_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 9208);
    _kws_conv3_activate_Relu_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _kws_conv3_activate_Relu_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _kws_conv3_quantize_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 1400);
    _kws_conv3_quantize_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1400);
    _kws_conv3_quantize_Mul_output_0_0_0__kws_conv3_quantize_Round_output_0_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 2800);
    _kws_conv3_quantize_Mul_output_0_0_0__kws_conv3_quantize_Round_output_0_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2800);
    _kws_conv3_quantize_Round_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 8400);
    _kws_conv3_quantize_Round_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 8400);
    _kws_conv3_quantize_Div_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _kws_conv3_quantize_Div_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _kws_conv3_quantize_Div_output_0_0_0__kws_conv3_Mul_2_output_0_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 5600);
    _kws_conv3_quantize_Div_output_0_0_0__kws_conv3_Mul_2_output_0_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 5600);
    _kws_conv3_Mul_2_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _kws_conv3_Mul_2_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _kws_conv4_pool_MaxPool_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 1400);
    _kws_conv4_pool_MaxPool_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1400);
    _kws_conv4_Conv_output_0_pad_before_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _kws_conv4_Conv_output_0_pad_before_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _kws_conv4_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 900);
    _kws_conv4_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 900);
    _kws_conv4_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 9316);
    _kws_conv4_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 9316);
    _kws_conv4_activate_Relu_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _kws_conv4_activate_Relu_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _kws_conv4_quantize_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 256);
    _kws_conv4_quantize_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 256);
    _kws_conv4_quantize_Mul_output_0_0_0__kws_conv4_quantize_Round_output_0_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 512);
    _kws_conv4_quantize_Mul_output_0_0_0__kws_conv4_quantize_Round_output_0_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 512);
    _kws_conv4_quantize_Round_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 1536);
    _kws_conv4_quantize_Round_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1536);
    _kws_conv4_quantize_Div_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _kws_conv4_quantize_Div_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _kws_conv4_quantize_Div_output_0_0_0__kws_conv4_Mul_2_output_0_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 1024);
    _kws_conv4_quantize_Div_output_0_0_0__kws_conv4_Mul_2_output_0_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1024);
    _kws_conv4_Mul_2_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _kws_conv4_Mul_2_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _Reshape_output_0_to_chlast_output_array.data = AI_PTR(g_network_activations_map[0] + 256);
    _Reshape_output_0_to_chlast_output_array.data_start = AI_PTR(g_network_activations_map[0] + 256);
    _fc_Gemm_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 512);
    _fc_Gemm_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 512);
    _fc_Gemm_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _fc_Gemm_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _fc_quantize_Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 24);
    _fc_quantize_Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 24);
    _fc_quantize_Mul_output_0_0_0__fc_quantize_Round_output_0_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 48);
    _fc_quantize_Mul_output_0_0_0__fc_quantize_Round_output_0_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 48);
    _fc_quantize_Round_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 132);
    _fc_quantize_Round_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 132);
    _fc_quantize_Div_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _fc_quantize_Div_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _fc_quantize_Div_output_0_0_0_kws_logits_QuantizeLinear_Input_conversion_output_array.data = AI_PTR(g_network_activations_map[0] + 84);
    _fc_quantize_Div_output_0_0_0_kws_logits_QuantizeLinear_Input_conversion_output_array.data_start = AI_PTR(g_network_activations_map[0] + 84);
    kws_logits_QuantizeLinear_Input_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    kws_logits_QuantizeLinear_Input_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
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
    _voice_conv4_quantize_Constant_output_0_DequantizeLinear_Output_const_4D_array.format |= AI_FMT_FLAG_CONST;
    _voice_conv4_quantize_Constant_output_0_DequantizeLinear_Output_const_4D_array.data = AI_PTR(g_network_weights_map[0] + 40);
    _voice_conv4_quantize_Constant_output_0_DequantizeLinear_Output_const_4D_array.data_start = AI_PTR(g_network_weights_map[0] + 40);
    _voice_conv4_calc_out_scale_Pow_output_0_DequantizeLinear_Output_const_4D_array.format |= AI_FMT_FLAG_CONST;
    _voice_conv4_calc_out_scale_Pow_output_0_DequantizeLinear_Output_const_4D_array.data = AI_PTR(g_network_weights_map[0] + 44);
    _voice_conv4_calc_out_scale_Pow_output_0_DequantizeLinear_Output_const_4D_array.data_start = AI_PTR(g_network_weights_map[0] + 44);
    _voice_conv4_Pow_output_0_DequantizeLinear_Output_const_4D_array.format |= AI_FMT_FLAG_CONST;
    _voice_conv4_Pow_output_0_DequantizeLinear_Output_const_4D_array.data = AI_PTR(g_network_weights_map[0] + 48);
    _voice_conv4_Pow_output_0_DequantizeLinear_Output_const_4D_array.data_start = AI_PTR(g_network_weights_map[0] + 48);
    _voice_conv3_quantize_Constant_output_0_DequantizeLinear_Output_const_4D_array.format |= AI_FMT_FLAG_CONST;
    _voice_conv3_quantize_Constant_output_0_DequantizeLinear_Output_const_4D_array.data = AI_PTR(g_network_weights_map[0] + 52);
    _voice_conv3_quantize_Constant_output_0_DequantizeLinear_Output_const_4D_array.data_start = AI_PTR(g_network_weights_map[0] + 52);
    _voice_conv3_calc_out_scale_Pow_output_0_DequantizeLinear_Output_const_4D_array.format |= AI_FMT_FLAG_CONST;
    _voice_conv3_calc_out_scale_Pow_output_0_DequantizeLinear_Output_const_4D_array.data = AI_PTR(g_network_weights_map[0] + 56);
    _voice_conv3_calc_out_scale_Pow_output_0_DequantizeLinear_Output_const_4D_array.data_start = AI_PTR(g_network_weights_map[0] + 56);
    _voice_conv3_Pow_output_0_DequantizeLinear_Output_const_4D_array.format |= AI_FMT_FLAG_CONST;
    _voice_conv3_Pow_output_0_DequantizeLinear_Output_const_4D_array.data = AI_PTR(g_network_weights_map[0] + 60);
    _voice_conv3_Pow_output_0_DequantizeLinear_Output_const_4D_array.data_start = AI_PTR(g_network_weights_map[0] + 60);
    _voice_conv2_quantize_Constant_output_0_DequantizeLinear_Output_const_4D_array.format |= AI_FMT_FLAG_CONST;
    _voice_conv2_quantize_Constant_output_0_DequantizeLinear_Output_const_4D_array.data = AI_PTR(g_network_weights_map[0] + 64);
    _voice_conv2_quantize_Constant_output_0_DequantizeLinear_Output_const_4D_array.data_start = AI_PTR(g_network_weights_map[0] + 64);
    _voice_conv2_calc_out_scale_Pow_output_0_DequantizeLinear_Output_const_4D_array.format |= AI_FMT_FLAG_CONST;
    _voice_conv2_calc_out_scale_Pow_output_0_DequantizeLinear_Output_const_4D_array.data = AI_PTR(g_network_weights_map[0] + 68);
    _voice_conv2_calc_out_scale_Pow_output_0_DequantizeLinear_Output_const_4D_array.data_start = AI_PTR(g_network_weights_map[0] + 68);
    _voice_conv2_Pow_output_0_DequantizeLinear_Output_const_4D_array.format |= AI_FMT_FLAG_CONST;
    _voice_conv2_Pow_output_0_DequantizeLinear_Output_const_4D_array.data = AI_PTR(g_network_weights_map[0] + 72);
    _voice_conv2_Pow_output_0_DequantizeLinear_Output_const_4D_array.data_start = AI_PTR(g_network_weights_map[0] + 72);
    _voice_conv1_quantize_Constant_output_0_DequantizeLinear_Output_const_4D_array.format |= AI_FMT_FLAG_CONST;
    _voice_conv1_quantize_Constant_output_0_DequantizeLinear_Output_const_4D_array.data = AI_PTR(g_network_weights_map[0] + 76);
    _voice_conv1_quantize_Constant_output_0_DequantizeLinear_Output_const_4D_array.data_start = AI_PTR(g_network_weights_map[0] + 76);
    _voice_conv1_calc_out_scale_Pow_output_0_DequantizeLinear_Output_const_4D_array.format |= AI_FMT_FLAG_CONST;
    _voice_conv1_calc_out_scale_Pow_output_0_DequantizeLinear_Output_const_4D_array.data = AI_PTR(g_network_weights_map[0] + 80);
    _voice_conv1_calc_out_scale_Pow_output_0_DequantizeLinear_Output_const_4D_array.data_start = AI_PTR(g_network_weights_map[0] + 80);
    _voice_conv1_Pow_output_0_DequantizeLinear_Output_const_4D_array.format |= AI_FMT_FLAG_CONST;
    _voice_conv1_Pow_output_0_DequantizeLinear_Output_const_4D_array.data = AI_PTR(g_network_weights_map[0] + 84);
    _voice_conv1_Pow_output_0_DequantizeLinear_Output_const_4D_array.data_start = AI_PTR(g_network_weights_map[0] + 84);
    _kws_conv4_quantize_Constant_output_0_DequantizeLinear_Output_const_4D_array.format |= AI_FMT_FLAG_CONST;
    _kws_conv4_quantize_Constant_output_0_DequantizeLinear_Output_const_4D_array.data = AI_PTR(g_network_weights_map[0] + 88);
    _kws_conv4_quantize_Constant_output_0_DequantizeLinear_Output_const_4D_array.data_start = AI_PTR(g_network_weights_map[0] + 88);
    _kws_conv4_calc_out_scale_Pow_output_0_DequantizeLinear_Output_const_4D_array.format |= AI_FMT_FLAG_CONST;
    _kws_conv4_calc_out_scale_Pow_output_0_DequantizeLinear_Output_const_4D_array.data = AI_PTR(g_network_weights_map[0] + 92);
    _kws_conv4_calc_out_scale_Pow_output_0_DequantizeLinear_Output_const_4D_array.data_start = AI_PTR(g_network_weights_map[0] + 92);
    _kws_conv4_Pow_output_0_DequantizeLinear_Output_const_4D_array.format |= AI_FMT_FLAG_CONST;
    _kws_conv4_Pow_output_0_DequantizeLinear_Output_const_4D_array.data = AI_PTR(g_network_weights_map[0] + 96);
    _kws_conv4_Pow_output_0_DequantizeLinear_Output_const_4D_array.data_start = AI_PTR(g_network_weights_map[0] + 96);
    _kws_conv3_quantize_pool_Constant_output_0_DequantizeLinear_Output_const_4D_array.format |= AI_FMT_FLAG_CONST;
    _kws_conv3_quantize_pool_Constant_output_0_DequantizeLinear_Output_const_4D_array.data = AI_PTR(g_network_weights_map[0] + 100);
    _kws_conv3_quantize_pool_Constant_output_0_DequantizeLinear_Output_const_4D_array.data_start = AI_PTR(g_network_weights_map[0] + 100);
    _kws_conv3_quantize_Constant_output_0_DequantizeLinear_Output_const_4D_array.format |= AI_FMT_FLAG_CONST;
    _kws_conv3_quantize_Constant_output_0_DequantizeLinear_Output_const_4D_array.data = AI_PTR(g_network_weights_map[0] + 104);
    _kws_conv3_quantize_Constant_output_0_DequantizeLinear_Output_const_4D_array.data_start = AI_PTR(g_network_weights_map[0] + 104);
    _kws_conv3_calc_out_scale_Pow_output_0_DequantizeLinear_Output_const_4D_array.format |= AI_FMT_FLAG_CONST;
    _kws_conv3_calc_out_scale_Pow_output_0_DequantizeLinear_Output_const_4D_array.data = AI_PTR(g_network_weights_map[0] + 108);
    _kws_conv3_calc_out_scale_Pow_output_0_DequantizeLinear_Output_const_4D_array.data_start = AI_PTR(g_network_weights_map[0] + 108);
    _kws_conv3_Pow_output_0_DequantizeLinear_Output_const_4D_array.format |= AI_FMT_FLAG_CONST;
    _kws_conv3_Pow_output_0_DequantizeLinear_Output_const_4D_array.data = AI_PTR(g_network_weights_map[0] + 112);
    _kws_conv3_Pow_output_0_DequantizeLinear_Output_const_4D_array.data_start = AI_PTR(g_network_weights_map[0] + 112);
    _kws_conv2_quantize_Constant_output_0_DequantizeLinear_Output_const_4D_array.format |= AI_FMT_FLAG_CONST;
    _kws_conv2_quantize_Constant_output_0_DequantizeLinear_Output_const_4D_array.data = AI_PTR(g_network_weights_map[0] + 116);
    _kws_conv2_quantize_Constant_output_0_DequantizeLinear_Output_const_4D_array.data_start = AI_PTR(g_network_weights_map[0] + 116);
    _kws_conv2_calc_out_scale_Pow_output_0_DequantizeLinear_Output_const_4D_array.format |= AI_FMT_FLAG_CONST;
    _kws_conv2_calc_out_scale_Pow_output_0_DequantizeLinear_Output_const_4D_array.data = AI_PTR(g_network_weights_map[0] + 120);
    _kws_conv2_calc_out_scale_Pow_output_0_DequantizeLinear_Output_const_4D_array.data_start = AI_PTR(g_network_weights_map[0] + 120);
    _kws_conv2_Pow_output_0_DequantizeLinear_Output_const_4D_array.format |= AI_FMT_FLAG_CONST;
    _kws_conv2_Pow_output_0_DequantizeLinear_Output_const_4D_array.data = AI_PTR(g_network_weights_map[0] + 124);
    _kws_conv2_Pow_output_0_DequantizeLinear_Output_const_4D_array.data_start = AI_PTR(g_network_weights_map[0] + 124);
    _kws_conv1_quantize_Constant_output_0_DequantizeLinear_Output_const_4D_array.format |= AI_FMT_FLAG_CONST;
    _kws_conv1_quantize_Constant_output_0_DequantizeLinear_Output_const_4D_array.data = AI_PTR(g_network_weights_map[0] + 128);
    _kws_conv1_quantize_Constant_output_0_DequantizeLinear_Output_const_4D_array.data_start = AI_PTR(g_network_weights_map[0] + 128);
    _kws_conv1_calc_out_scale_Pow_output_0_DequantizeLinear_Output_const_4D_array.format |= AI_FMT_FLAG_CONST;
    _kws_conv1_calc_out_scale_Pow_output_0_DequantizeLinear_Output_const_4D_array.data = AI_PTR(g_network_weights_map[0] + 132);
    _kws_conv1_calc_out_scale_Pow_output_0_DequantizeLinear_Output_const_4D_array.data_start = AI_PTR(g_network_weights_map[0] + 132);
    _kws_conv1_Pow_output_0_DequantizeLinear_Output_const_4D_array.format |= AI_FMT_FLAG_CONST;
    _kws_conv1_Pow_output_0_DequantizeLinear_Output_const_4D_array.data = AI_PTR(g_network_weights_map[0] + 136);
    _kws_conv1_Pow_output_0_DequantizeLinear_Output_const_4D_array.data_start = AI_PTR(g_network_weights_map[0] + 136);
    _fc_quantize_Constant_output_0_DequantizeLinear_Output_const_2D_array.format |= AI_FMT_FLAG_CONST;
    _fc_quantize_Constant_output_0_DequantizeLinear_Output_const_2D_array.data = AI_PTR(g_network_weights_map[0] + 140);
    _fc_quantize_Constant_output_0_DequantizeLinear_Output_const_2D_array.data_start = AI_PTR(g_network_weights_map[0] + 140);
    _fc_Pow_output_0_DequantizeLinear_Output_const_2D_array.format |= AI_FMT_FLAG_CONST;
    _fc_Pow_output_0_DequantizeLinear_Output_const_2D_array.data = AI_PTR(g_network_weights_map[0] + 144);
    _fc_Pow_output_0_DequantizeLinear_Output_const_2D_array.data_start = AI_PTR(g_network_weights_map[0] + 144);
    _voice_conv1_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _voice_conv1_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 148);
    _voice_conv1_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 148);
    _voice_conv1_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _voice_conv1_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 12948);
    _voice_conv1_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 12948);
    _voice_conv2_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _voice_conv2_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 13348);
    _voice_conv2_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 13348);
    _voice_conv2_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _voice_conv2_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 42148);
    _voice_conv2_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 42148);
    _voice_conv3_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _voice_conv3_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 42532);
    _voice_conv3_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 42532);
    _voice_conv3_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _voice_conv3_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 60964);
    _voice_conv3_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 60964);
    _voice_conv4_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _voice_conv4_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 61220);
    _voice_conv4_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 61220);
    _voice_conv4_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _voice_conv4_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 70436);
    _voice_conv4_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 70436);
    _kws_conv1_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _kws_conv1_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 70628);
    _kws_conv1_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 70628);
    _kws_conv1_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _kws_conv1_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 79844);
    _kws_conv1_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 79844);
    _kws_conv2_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _kws_conv2_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 80100);
    _kws_conv2_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 80100);
    _kws_conv2_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _kws_conv2_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 98532);
    _kws_conv2_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 98532);
    _kws_conv3_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _kws_conv3_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 98916);
    _kws_conv3_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 98916);
    _kws_conv3_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _kws_conv3_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 127716);
    _kws_conv3_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 127716);
    _kws_conv4_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _kws_conv4_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 128116);
    _kws_conv4_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 128116);
    _kws_conv4_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _kws_conv4_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 166516);
    _kws_conv4_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 166516);
    _fc_Gemm_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _fc_Gemm_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 166772);
    _fc_Gemm_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 166772);
    _fc_Gemm_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _fc_Gemm_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 172148);
    _fc_Gemm_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 172148);
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
      
      .n_macc            = 8915270,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .params            = AI_STRUCT_INIT,
      .activations       = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0xc3398bbd,
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
      
      .n_macc            = 8915270,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .map_signature     = AI_MAGIC_SIGNATURE,
      .map_weights       = AI_STRUCT_INIT,
      .map_activations   = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0xc3398bbd,
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

