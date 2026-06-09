#!/usr/bin/env python3
"""
Convert ds_cnn_l_static.tflite (INT8) to Deeploy fake-quantized ONNX for GAP9.

Architecture (NCHW, same as MLPerf KWS reference):
  uint8 [1,1,49,10]
  → Pad (top/left/bottom/right for SAME 10×4 conv with stride 2)
  → Conv (weights [64,1,10,4])
  → RequantShift (uint8 out, ReLU fused)
  → 4 × [ DWConv [64,1,3,3] → RequantShift (uint8),
           PWConv [64,64,1,1] → RequantShift (uint8) ]
  → ReduceMean axes=[2,3] keepdims=0 (→ uint8 [1,64])
  → Gemm (weights [12,64], transB=1)
  → RequantShift (int8 signed out)
  → int8 [1,12]

Quantization math (div=D=65536=2^16):
  kappa[oc]    = round(scale_in * scale_w[oc] / scale_out * D)
  bias_eff[oc] = b_int32[oc] - (128 + zp_in) * sum_w[oc]
  add_onnx[oc] = bias_eff[oc] * kappa[oc] + zp_out_uint8 * D
  (Deeploy adds round(D/2) internally when building the merged PULP kernel.)

  For first Conv:   zp_in = -9  → correction = (128 + (-9)) = 119
  For RELU layers:  zp_in = -128 → correction = 0 (no zp correction)
  For RELU output:  zp_out_uint8 = 0   (TFLite zp_out_int8 = -128 → uint8 zero)
  For FC output:    zp_out = zp_out_tflite = 34 (signed int8, non-RELU)

Input/output encoding:
  ONNX: float32 (all values are integers stored as float — "fake quantized")
  NPZ:  int64 (actual integer values to load into hardware)
  Input: tflite_int8 + 128 → uint8 (stored as int64 in npz)
  Output: tflite_int8 (stored as int64 in npz)
"""

import os
import re
import struct
import numpy as np
import onnx
from onnx import numpy_helper, TensorProto, helper, AttributeProto

TFLITE_PATH = os.path.join(os.path.dirname(__file__),
                           "../../models/ds_cnn_l_static.tflite")
HEADER_PATH = os.path.join(os.path.dirname(__file__),
                           "../../platforms/max78000/keyword_spotting_max78000_ds_cnn"
                           "/ds_cnn_test_input_left.h")
DEEPLOY_TEST_DIR = os.path.join(os.path.dirname(__file__),
                                "Deeploy/DeeployTest")
OUT_DIR = os.path.join(DEEPLOY_TEST_DIR, "Tests/Models/DSCNNL_INT8")
os.makedirs(OUT_DIR, exist_ok=True)

D = 16         # requant shift bits
DIV = 1 << D  # 65536


# ── 1. Parse TFLite ──────────────────────────────────────────────────────────

import tflite
from tflite.Model import Model

with open(TFLITE_PATH, "rb") as f:
    buf = bytearray(f.read())
model_tf = Model.GetRootAsModel(buf, 0)
subgraph = model_tf.Subgraphs(0)


def _tf_tensor_data(tidx):
    """Return numpy array for TFLite tensor by index."""
    t = subgraph.Tensors(tidx)
    buf_idx = t.Buffer()
    raw = model_tf.Buffers(buf_idx).DataAsNumpy()
    dtype_map = {9: np.int8, 2: np.int32, 0: np.float32}
    dtype = dtype_map.get(t.Type(), np.uint8)
    return raw.view(dtype).reshape([t.Shape(j) for j in range(t.ShapeLength())])


def _tf_tensor_quant(tidx):
    """Return (scales_array, zp_array) for TFLite tensor."""
    q = subgraph.Tensors(tidx).Quantization()
    scales = np.array([q.Scale(i) for i in range(q.ScaleLength())], dtype=np.float64)
    zps    = np.array([q.ZeroPoint(i) for i in range(q.ZeroPointLength())], dtype=np.int64)
    return scales, zps


# Collect op list: each op = (opcode_idx, inputs, outputs)
ops = []
for i in range(subgraph.OperatorsLength()):
    op = subgraph.Operators(i)
    ops.append((op.OpcodeIndex(),
                [op.Inputs(j) for j in range(op.InputsLength())],
                [op.Outputs(j) for j in range(op.OutputsLength())]))

# Builtin op codes: 1=AVG_POOL, 3=CONV_2D, 4=DW_CONV, 9=FC, 25=RESHAPE
opcodes_builtin = {}
for i in range(model_tf.OperatorCodesLength()):
    oc = model_tf.OperatorCodes(i)
    opcodes_builtin[i] = oc.BuiltinCode()

print("TFLite ops:")
for i, (oc_idx, ins, outs) in enumerate(ops):
    bc = opcodes_builtin[oc_idx]
    name = {1: "AVG_POOL", 3: "CONV_2D", 4: "DW_CONV", 9: "FC", 25: "RESHAPE"}.get(bc, str(bc))
    print(f"  {i:2d}: {name:10s} in={ins} out={outs}")

# Input tensor
in_tidx = subgraph.Inputs(0)
s_in_scales, s_in_zps = _tf_tensor_quant(in_tidx)
s_in = float(s_in_scales[0])
zp_in_first = int(s_in_zps[0])   # -9 for model input
print(f"\nModel input: scale={s_in:.4e}, zp={zp_in_first}")

# Output tensor (before reshape / softmax)
out_tidx = subgraph.Outputs(0)   # = 32 (softmax output)


# ── 2. RequantShift helpers ──────────────────────────────────────────────────

def make_rqs_attr_tensors(div_val, n_levels_val, signed_val):
    """Create RequantShift ONNX node attributes (tensor type)."""
    return [
        helper.make_attribute('div',
            numpy_helper.from_array(np.array(float(div_val), dtype=np.float32))),
        helper.make_attribute('n_levels_out',
            numpy_helper.from_array(np.array([float(n_levels_val)], dtype=np.float32))),
        helper.make_attribute('signed',
            numpy_helper.from_array(np.array([float(signed_val)], dtype=np.float32))),
    ]


def compute_rqs_params(s_in_val, s_w_arr, s_out_val, zp_in_val,
                        b_int32_arr, w_data_nchw, zp_out_tflite,
                        is_relu_out=True, input_is_uint8=True):
    """
    Compute per-channel RequantShift mul and add for a Conv/DW/PW layer.

    s_in_val      : scalar input activation scale
    s_w_arr       : per-channel weight scales, shape [OC]
    s_out_val     : scalar output activation scale
    zp_in_val     : input activation zp (int8) — -9 for model input, -128 for RELU
    b_int32_arr   : bias in int32, shape [OC]
    w_data_nchw   : weight data in NCHW int8, shape [OC, IC, kH, kW]
    zp_out_tflite : output zp (tflite signed int8)
    is_relu_out   : True if output is RELU (uint8 unsigned output)
    input_is_uint8: True for uint8 input (x_uint8 = x_int8+128), False for raw int8

    Returns mul_arr, add_arr (float32 arrays, int-valued).

    For uint8 input: acc_pulp = sum((x_int8+128)*w) = acc_tflite + (128+zp_in)*sum_w
    For int8 input:  acc_pulp = sum(x_int8*w)       = acc_tflite + zp_in*sum_w
    Effective bias:  bias_eff = b_int32 - (offset)*sum_w
    where offset = (128+zp_in) for uint8, zp_in for int8.
    """
    oc = s_w_arr.shape[0]
    M = s_in_val * s_w_arr / s_out_val           # shape [OC]
    kappa = np.round(M * DIV).astype(np.int64)   # int32-valued

    # Sum of weights per output channel (over IC, kH, kW)
    sum_w = w_data_nchw.reshape(oc, -1).sum(axis=1).astype(np.int64)  # [OC]

    # Bias correction depends on input encoding
    if input_is_uint8:
        correction = (128 + zp_in_val) * sum_w   # uint8: acc_pulp includes 128*sum_w extra
    else:
        correction = zp_in_val * sum_w           # int8: acc_pulp = acc_tflite + zp_in*sum_w
    bias_eff = b_int32_arr.astype(np.int64) - correction  # [OC]

    # zp_out_uint8: for RELU/unsigned output (TFLite zp=-128 → uint8 zero)
    # the PULP kernel produces uint8 directly, so zp_out_tflite=-128 → offset=0.
    # For signed FC output (zp_out_tflite=34), offset = zp_out_tflite * DIV.
    if is_relu_out:
        zp_term = 0
    else:
        # Signed output: offset = zp_out_tflite * DIV
        zp_term = int(zp_out_tflite) * DIV

    add_onnx = bias_eff * kappa + zp_term        # [OC]

    return kappa.astype(np.float32), add_onnx.astype(np.float32)


# ── 3. Build ONNX graph ───────────────────────────────────────────────────────

nodes        = []
initializers = []
value_infos  = []
node_cnt     = [0]


def uid():
    node_cnt[0] += 1
    return node_cnt[0]


def add_init(name, arr):
    initializers.append(numpy_helper.from_array(arr.astype(np.float32), name=name))


def add_vi(name, shape):
    value_infos.append(
        helper.make_tensor_value_info(name, TensorProto.FLOAT, shape))


INPUT_NAME = "input_0"
input_shape = [1, 1, 49, 10]   # NCHW
add_vi(INPUT_NAME, input_shape)

prev = INPUT_NAME
current_shape = input_shape[:]   # track [N, C, H, W]

# Helper: add attribute-style Pad + Conv(zero-pads)
def add_padded_conv(inp, w_nchw, b_int32, s_in_val, s_w, s_out, zp_in_val,
                     zp_out, strides, pads_tflite_nhwc, is_dw, is_relu,
                     tag, input_is_uint8=True):
    """Insert optional Pad then Conv/DWConv + RequantShift.
    pads_tflite_nhwc: [top, left, bottom, right] (computed for SAME padding).
    Returns output tensor name and output NCHW shape.
    """

    oc, ic, kh, kw = w_nchw.shape

    # Pad check
    top, left, bot, right = pads_tflite_nhwc
    need_pad = (top != bot) or (left != right)

    if need_pad:
        padded_name = f"padded_{tag}"
        pad_node = helper.make_node(
            'Pad', inputs=[inp], outputs=[padded_name], name=f"pad_{tag}",
            mode='constant',
            pads=[0, 0, top, left, 0, 0, bot, right],
            value=0.0,
        )
        nodes.append(pad_node)
        in_sh = current_shape[:]
        pad_shape = [in_sh[0], in_sh[1], in_sh[2] + top + bot, in_sh[3] + left + right]
        add_vi(padded_name, pad_shape)
        conv_in = padded_name
        conv_pads = [0, 0, 0, 0]
    else:
        conv_in = inp
        conv_pads = [top, left, bot, right]

    # Weight initializer (already NCHW int8-valued as float32)
    w_name = f"w_{tag}"
    add_init(w_name, w_nchw.astype(np.float32))

    # RequantShift mul/add
    mul_arr, add_arr = compute_rqs_params(
        s_in_val, s_w, s_out, zp_in_val,
        b_int32, w_nchw,
        zp_out, is_relu_out=is_relu, input_is_uint8=input_is_uint8,
    )
    mul_name = f"mul_{tag}"
    add_name = f"add_{tag}"
    # Shape for mul/add: [OC, 1, 1] (per-channel)
    add_init(mul_name, mul_arr.reshape(oc, 1, 1))
    add_init(add_name, add_arr.reshape(oc, 1, 1))

    # Conv output
    conv_out = f"conv_out_{tag}"
    # Determine output spatial size
    in_shape = pad_shape if need_pad else current_shape
    h_out = (in_shape[2] - kh + conv_pads[0] + conv_pads[2]) // strides[0] + 1
    w_out = (in_shape[3] - kw + conv_pads[1] + conv_pads[3]) // strides[1] + 1
    out_nchw = [1, oc, h_out, w_out]
    add_vi(conv_out, out_nchw)

    # Build Conv node attrs
    conv_attrs = dict(
        kernel_shape=[kh, kw],
        strides=strides,
        pads=conv_pads,
        dilations=[1, 1],
        group=oc if is_dw else 1,  # DW: group=OC (one filter per input channel)
    )
    conv_node = helper.make_node(
        'Conv', inputs=[conv_in, w_name], outputs=[conv_out],
        name=f"conv_{tag}", **conv_attrs)
    nodes.append(conv_node)

    # RequantShift output
    rqs_out = f"rqs_out_{tag}"
    signed_flag = 0.0 if is_relu else 1.0
    n_lvl = 256.0
    rqs_attrs = make_rqs_attr_tensors(DIV, n_lvl, signed_flag)
    rqs_node = helper.make_node(
        'RequantShift',
        inputs=[conv_out, mul_name, add_name],
        outputs=[rqs_out],
        name=f"rqs_{tag}",
    )
    for a in rqs_attrs:
        rqs_node.attribute.append(a)
    nodes.append(rqs_node)
    add_vi(rqs_out, out_nchw)

    return rqs_out, out_nchw


def same_pads_nchw(in_h, in_w, kh, kw, sh, sw):
    """Compute SAME padding [top, left, bottom, right] (asymmetric like TFLite)."""
    out_h = (in_h + sh - 1) // sh
    out_w = (in_w + sw - 1) // sw
    pad_h = max((out_h - 1) * sh + kh - in_h, 0)
    pad_w = max((out_w - 1) * sw + kw - in_w, 0)
    return [pad_h // 2, pad_w // 2, pad_h - pad_h // 2, pad_w - pad_w // 2]


# ── Op 0: CONV_2D 10×4, stride 2×2 ──────────────────────────────────────────
oc_idx, ins, outs = ops[0]
in_t, w_t, b_t = ins
out_t = outs[0]

w_nhwc = _tf_tensor_data(w_t)  # [64,10,4,1] OC,kH,kW,IC
w_nchw = np.transpose(w_nhwc, (0, 3, 1, 2))  # [64,1,10,4]
b_int32 = _tf_tensor_data(b_t)  # [64]
s_in_scale, _ = _tf_tensor_quant(in_t)
s_w_arr, _    = _tf_tensor_quant(w_t)   # [64] per-channel
s_out_sc, s_out_zp = _tf_tensor_quant(out_t)

pads = same_pads_nchw(49, 10, 10, 4, 2, 2)
print(f"\nOp0 pads={pads}")

prev, current_shape = add_padded_conv(
    INPUT_NAME, w_nchw, b_int32,
    float(s_in_scale[0]), s_w_arr, float(s_out_sc[0]),
    zp_in_first,   # -9 for model input
    int(s_out_zp[0]),
    strides=[2, 2],
    pads_tflite_nhwc=pads,
    is_dw=False, is_relu=True,
    tag="conv0",
    input_is_uint8=False,  # network input is int8 so Pad binding works (SignedIntegerDataTypes)
)
print(f"  → {current_shape}")

# ── Ops 1–8: 4 DS blocks (DWConv + PWConv) ───────────────────────────────────
for block in range(4):
    # DW Conv (op 2*block+1)
    op_dw_idx = 2 * block + 1
    oc_idx, ins, outs = ops[op_dw_idx]
    in_t, w_t, b_t = ins
    out_t = outs[0]

    w_nhwc = _tf_tensor_data(w_t)  # [1,3,3,C] in TFLite OHWI for DW: O=1(mult), H, W, I=C
    # TFLite DW weight layout: [1, kH, kW, C_in] — but stored as [C_in,kH,kW,1] in some versions
    # Actually TFLite DEPTHWISE_CONV_2D: weight shape = [1, kH, kW, channels*depth_multiplier]
    # Our model: [1,3,3,64] → [1,kH,kW,IC*CM] where CM=1, IC=64
    # ONNX depthwise: group=IC, weight=[IC,CM,kH,kW]=[64,1,3,3]
    # Transpose: [1,3,3,64] → [64,1,3,3]
    w_nchw = np.transpose(w_nhwc[0], (2, 0, 1))[:, np.newaxis, :, :]  # [64,1,3,3]

    b_int32 = _tf_tensor_data(b_t)       # [64]
    s_in_sc, s_in_zp = _tf_tensor_quant(in_t)
    s_w_arr, _        = _tf_tensor_quant(w_t)
    s_out_sc, s_out_zp = _tf_tensor_quant(out_t)

    pads = same_pads_nchw(current_shape[2], current_shape[3], 3, 3, 1, 1)
    prev, current_shape = add_padded_conv(
        prev, w_nchw, b_int32,
        float(s_in_sc[0]), s_w_arr, float(s_out_sc[0]),
        int(s_in_zp[0]),   # -128 (RELU)
        int(s_out_zp[0]),
        strides=[1, 1],
        pads_tflite_nhwc=pads,
        is_dw=True, is_relu=True,
        tag=f"dw{block}",
    )
    print(f"  DW block{block}: {current_shape}")

    # PW Conv (op 2*block+2)
    op_pw_idx = 2 * block + 2
    oc_idx, ins, outs = ops[op_pw_idx]
    in_t, w_t, b_t = ins
    out_t = outs[0]

    w_nhwc = _tf_tensor_data(w_t)  # [64,1,1,64] OC,1,1,IC
    w_nchw = np.transpose(w_nhwc, (0, 3, 1, 2))  # [64,64,1,1]
    b_int32 = _tf_tensor_data(b_t)
    s_in_sc, s_in_zp = _tf_tensor_quant(in_t)
    s_w_arr, _        = _tf_tensor_quant(w_t)
    s_out_sc, s_out_zp = _tf_tensor_quant(out_t)

    prev, current_shape = add_padded_conv(
        prev, w_nchw, b_int32,
        float(s_in_sc[0]), s_w_arr, float(s_out_sc[0]),
        int(s_in_zp[0]),   # -128 (RELU)
        int(s_out_zp[0]),
        strides=[1, 1],
        pads_tflite_nhwc=[0, 0, 0, 0],
        is_dw=False, is_relu=True,
        tag=f"pw{block}",
    )
    print(f"  PW block{block}: {current_shape}")

# ── Op 9: AVERAGE_POOL_2D → ReduceMean ───────────────────────────────────────
# GlobalAvgPool: reduce spatial dims [2,3], keepdims=0 → output [1,64]
# scale_in = scale_out (same tensor), zp_in = zp_out = -128 → pure averaging, no RQS needed
#
# Use axes as ONNX attribute (integer-node path, 1 input):
#   - INT8 GAP9 binding: ReduceMeanChecker([PointerClass(type)], ...) expects 1 input.
#     With 2 inputs (axes tensor), the axes constant never gets _type annotated → bind() crash.
#   - 'axis' (singular) attribute needed by RemoveOnlySingletonReduceMeanPass (it checks
#     node.attrs['axis'] not node.attrs['axes'], and falls back to node.inputs[1] otherwise).
rm_out = "avgpool_out"
rm_shape = [1, current_shape[1]]   # keepdims=0 collapses spatial dims
rm_node = helper.make_node(
    'ReduceMean',
    inputs=[prev],              # 1 input only (INT8 binding requires this)
    outputs=[rm_out],
    name="global_avgpool",
    keepdims=0,
    axes=[2, 3],                # ONNX standard — used by ReduceParser integer path
    axis=[2, 3],                # non-standard extra attr for RemoveOnlySingletonReduceMeanPass
)
nodes.append(rm_node)
add_vi(rm_out, rm_shape)
print(f"\nAvgPool (ReduceMean): {rm_shape}")

# No Reshape needed: keepdims=0 already gives [1, 64] → Gemm input directly
flat_out = rm_out

# ── Op 10: FULLY_CONNECTED → Gemm + RequantShift ────────────────────────────
oc_idx, ins, outs = ops[10]
in_t, w_t, b_t = ins
out_t = outs[0]

w_data = _tf_tensor_data(w_t)   # [12, 64]
b_int32 = _tf_tensor_data(b_t)  # [12]
s_in_sc, s_in_zp = _tf_tensor_quant(in_t)
s_w_arr_fc, _     = _tf_tensor_quant(w_t)   # shape [1] (per-tensor for FC)
s_out_sc, s_out_zp = _tf_tensor_quant(out_t)

# Broadcast per-tensor weight scale to per-output-channel [12]
s_w_fc = np.broadcast_to(s_w_arr_fc, (12,)).copy()

# FC: input is RELU uint8 → zp_in_eff = -128, (128+zp_in) = 0 → no input correction
# Weight shape [OC=12, IC=64], already in correct ONNX layout for transB=1: Gemm(A,B^T)
# or just [12,64] with transB=1 → [1,64] × [64,12] → [1,12]
w_name_fc = "w_fc"
add_init(w_name_fc, w_data.astype(np.float32))
# Zero bias: bias absorbed into RequantShift add tensor (ONNX Gemm still takes 3 inputs)
bias_fc_name = "bias_fc_zeros"
add_init(bias_fc_name, np.zeros(12, dtype=np.float32))

gemm_out = "gemm_out"
gemm_shape = [1, 12]
gemm_node = helper.make_node(
    'Gemm',
    inputs=[flat_out, w_name_fc, bias_fc_name],
    outputs=[gemm_out],
    name="fc_gemm",
    transB=1,
    alpha=1.0,
    beta=1.0,
)
nodes.append(gemm_node)
add_vi(gemm_out, gemm_shape)

# RequantShift for FC (signed int8 output)
mul_fc, add_fc = compute_rqs_params(
    float(s_in_sc[0]), s_w_fc, float(s_out_sc[0]),
    int(s_in_zp[0]),   # -128 (RELU avgpool output)
    b_int32, w_data.reshape(12, 1, 64, 1),   # fake NCHW shape for sum_w
    int(s_out_zp[0]),  # 34
    is_relu_out=False,
)
mul_name_fc = "mul_fc"
add_name_fc = "add_fc"
# For Gemm (non-spatial), shape [OC=12] (not [OC,1,1])
add_init(mul_name_fc, mul_fc.reshape(12))
add_init(add_name_fc, add_fc.reshape(12))

OUTPUT_NAME = "output_0"
rqs_attrs_fc = make_rqs_attr_tensors(DIV, 256.0, 1.0)  # signed=1
rqs_fc = helper.make_node(
    'RequantShift',
    inputs=[gemm_out, mul_name_fc, add_name_fc],
    outputs=[OUTPUT_NAME],
    name="rqs_fc",
)
for a in rqs_attrs_fc:
    rqs_fc.attribute.append(a)
nodes.append(rqs_fc)
add_vi(OUTPUT_NAME, gemm_shape)

print(f"FC: {rm_shape} → {gemm_shape} (int8 signed)")

# ── 4. Assemble ONNX model ────────────────────────────────────────────────────

graph = helper.make_graph(
    nodes,
    "DSCNNL_INT8",
    [helper.make_tensor_value_info(INPUT_NAME, TensorProto.FLOAT, input_shape)],
    [helper.make_tensor_value_info(OUTPUT_NAME, TensorProto.FLOAT, gemm_shape)],
    initializers,
    value_info=value_infos,
)

# Use same opsets as MLPerf model so Deeploy's parsers work
opset_imports = [
    helper.make_opsetid("", 10),
    helper.make_opsetid("PACTOps", 1),
]
model_onnx = helper.make_model(graph, opset_imports=opset_imports)
model_onnx.ir_version = 5

out_onnx = os.path.join(OUT_DIR, "network.onnx")
onnx.save(model_onnx, out_onnx)
print(f"\nSaved ONNX: {out_onnx}")

# ── 5. Test vectors ───────────────────────────────────────────────────────────

# Load real MFCC input from the same test header used for float32
with open(HEADER_PATH) as f:
    txt = f.read()
nums = re.findall(r'[-+]?\d*\.?\d+e[+-]?\d+f|[-+]?\d+\.\d+f', txt)
mfcc_float = np.array([float(v.rstrip('f')) for v in nums],
                       dtype=np.float32).reshape(1, 49, 10, 1)  # NHWC

# Quantize to TFLite int8: q = clip(round(real / s_in) + zp_in, -128, 127)
s_in_model = float(_tf_tensor_quant(0)[0][0])   # tensor 0 = input
zp_in_model = int(_tf_tensor_quant(0)[1][0])
q_int8 = np.clip(np.round(mfcc_float / s_in_model) + zp_in_model, -128, 127).astype(np.int8)
print(f"\nTest input (int8): min={q_int8.min()}, max={q_int8.max()}")

# Network input uses raw int8 encoding (NOT uint8+128).
# The Pad node binding only supports SignedIntegerDataTypes (int8/int16/int32/int64),
# so we must keep the input as signed int8 to allow the Pad binding to succeed.
# All intermediate activations after the first Conv are still uint8 (RELU output).

# Reshape to NCHW [1,1,49,10]
q_int8_nchw = np.transpose(q_int8, (0, 3, 1, 2))   # [1,1,49,10]

# Reference output: run TFLite to get the int8 logits
# We'll compute via numpy using TFLite quantization params (offline)
# For now, compute a simple reference by running the ONNX float values through ORT
# (We'll verify on board; the key is correct ONNX structure)

# Compute reference output by manually running quantized inference
def relu(x):
    return np.maximum(x, 0)


def requantize(acc_int32, mul_arr, add_arr, D_bits, is_relu, zp_out_uint8_offset=0):
    """Apply RequantShift: (acc * mul + add) >> D + offset, then clip."""
    out = (acc_int32.astype(np.int64) * mul_arr.astype(np.int64)
           + add_arr.astype(np.int64))
    # rounding: add 2^(D-1) and shift
    out = (out + (1 << (D_bits - 1))) >> D_bits
    out += zp_out_uint8_offset
    if is_relu:
        return np.clip(out, 0, 255).astype(np.uint8)
    else:
        return np.clip(out, -128, 127).astype(np.int8)


def int_conv2d(x_uint8, w_int8, pads, strides, is_dw=False):
    """Simple integer conv (NCHW, no bias)."""
    N, C_in, H, W = x_uint8.shape
    OC, IC, kH, kW = w_int8.shape
    sH, sW = strides
    pH = pads[0] + pads[2]
    pW = pads[1] + pads[3]
    x_pad = np.pad(x_uint8.astype(np.int32),
                   ((0,0), (0,0), (pads[0], pads[2]), (pads[1], pads[3])))
    H_out = (H + pads[0] + pads[2] - kH) // sH + 1
    W_out = (W + pads[1] + pads[3] - kW) // sW + 1
    out = np.zeros((N, OC, H_out, W_out), dtype=np.int64)
    if is_dw:
        for oc in range(OC):
            for h in range(H_out):
                for w in range(W_out):
                    patch = x_pad[0, oc, h*sH:h*sH+kH, w*sW:w*sW+kW]
                    out[0, oc, h, w] = np.sum(patch * w_int8[oc, 0])
    else:
        for oc in range(OC):
            for h in range(H_out):
                for w in range(W_out):
                    patch = x_pad[0, :, h*sH:h*sH+kH, w*sW:w*sW+kW]
                    out[0, oc, h, w] = np.sum(patch * w_int8[oc])
    return out   # int64 accumulator


print("\nRunning reference inference...")

# Collect all layer params in order
layer_params = []

# Op0: CONV_2D
oc_idx, ins, outs = ops[0]
in_t, w_t, b_t = ins
w_nhwc = _tf_tensor_data(w_t).astype(np.int8)
w_nchw0 = np.transpose(w_nhwc, (0, 3, 1, 2))
b0 = _tf_tensor_data(b_t)
s_w0, _ = _tf_tensor_quant(w_t)
s_out0, s_out_zp0 = _tf_tensor_quant(outs[0])
mul0, add0 = compute_rqs_params(s_in_model, s_w0, float(s_out0[0]),
                                  zp_in_model, b0, w_nchw0,
                                  int(s_out_zp0[0]), is_relu_out=True,
                                  input_is_uint8=False)  # int8 network input
pads0 = same_pads_nchw(49, 10, 10, 4, 2, 2)
layer_params.append(('conv', w_nchw0, mul0, add0, [2,2], pads0, False))

for block in range(4):
    # DW
    oc_idx, ins, outs = ops[2*block+1]
    in_t_dw, w_t_dw, b_t_dw = ins
    w_nhwc_dw = _tf_tensor_data(w_t_dw)  # [1,3,3,64]
    w_nchw_dw = np.transpose(w_nhwc_dw[0], (2, 0, 1))[:, np.newaxis, :, :].astype(np.int8)
    b_dw = _tf_tensor_data(b_t_dw)
    s_w_dw, _ = _tf_tensor_quant(w_t_dw)
    s_in_dw, s_in_dw_zp = _tf_tensor_quant(in_t_dw)
    s_out_dw, s_out_dw_zp = _tf_tensor_quant(outs[0])
    mul_dw, add_dw = compute_rqs_params(float(s_in_dw[0]), s_w_dw, float(s_out_dw[0]),
                                          int(s_in_dw_zp[0]), b_dw, w_nchw_dw,
                                          int(s_out_dw_zp[0]), is_relu_out=True)
    pads_dw = same_pads_nchw(25, 5, 3, 3, 1, 1)
    layer_params.append(('conv', w_nchw_dw, mul_dw, add_dw, [1,1], pads_dw, True))

    # PW
    oc_idx, ins, outs = ops[2*block+2]
    in_t_pw, w_t_pw, b_t_pw = ins
    w_nhwc_pw = _tf_tensor_data(w_t_pw)  # [64,1,1,64]
    w_nchw_pw = np.transpose(w_nhwc_pw, (0, 3, 1, 2)).astype(np.int8)  # [64,64,1,1]
    b_pw = _tf_tensor_data(b_t_pw)
    s_w_pw, _ = _tf_tensor_quant(w_t_pw)
    s_in_pw, s_in_pw_zp = _tf_tensor_quant(in_t_pw)
    s_out_pw, s_out_pw_zp = _tf_tensor_quant(outs[0])
    mul_pw, add_pw = compute_rqs_params(float(s_in_pw[0]), s_w_pw, float(s_out_pw[0]),
                                          int(s_in_pw_zp[0]), b_pw, w_nchw_pw,
                                          int(s_out_pw_zp[0]), is_relu_out=True)
    layer_params.append(('conv', w_nchw_pw, mul_pw, add_pw, [1,1], [0,0,0,0], False))

# Run inference — first layer uses int8 input, subsequent layers use uint8
x = q_int8_nchw.copy()   # int8 [1,1,49,10] (signed, TFLite native encoding)

for i, lp in enumerate(layer_params):
    kind, w_int8, mul_arr, add_arr, strides, pads, is_dw = lp
    acc = int_conv2d(x, w_int8, pads, strides, is_dw=is_dw)   # int64 [1,OC,H,W]
    oc = mul_arr.shape[0]
    mul_bc = mul_arr.reshape(1, oc, 1, 1).astype(np.int64)
    add_bc = add_arr.reshape(1, oc, 1, 1).astype(np.int64)
    x_rq = (acc * mul_bc + add_bc + (1 << (D-1))) >> D
    x = np.clip(x_rq, 0, 255).astype(np.uint8)  # uint8 after ReLU (first Conv output onward)
    print(f"  Layer {i}: {x.shape} uint8 [{x.min()},{x.max()}]")

# GlobalAvgPool over spatial [H,W]
x_avg = x.mean(axis=(2, 3), keepdims=False).round().astype(np.int64)  # [1,64]
print(f"  AvgPool: {x_avg.shape} [{x_avg.min()},{x_avg.max()}]")

# FC Gemm (uint8 input, int8 weights, int8 output)
w_fc_int8 = _tf_tensor_data(2).astype(np.int8)   # [12, 64]
b_fc = _tf_tensor_data(1)                          # [12] int32
acc_fc = x_avg.astype(np.int64) @ w_fc_int8.T.astype(np.int64)  # [1,12]
mul_fc_np = mul_fc.astype(np.int64)
add_fc_np = add_fc.astype(np.int64)
x_fc = (acc_fc * mul_fc_np + add_fc_np + (1 << (D-1))) >> D
x_fc = np.clip(x_fc, -128, 127).astype(np.int8)
print(f"  FC: {x_fc.shape} int8 [{x_fc.min()},{x_fc.max()}]")
print(f"  Predicted class: {np.argmax(x_fc[0])} (logit={x_fc[0,np.argmax(x_fc[0])]})")

# ── 6. Save NPZ test vectors ──────────────────────────────────────────────────
# Store as int64 (Deeploy convention: actual integer values)
# Input: int8 signed values (NOT uint8) — required for Pad binding (SignedIntegerDataTypes)
np.savez(os.path.join(OUT_DIR, "inputs.npz"),
         input_0=q_int8_nchw.astype(np.int64))
np.savez(os.path.join(OUT_DIR, "outputs.npz"),
         output_0=x_fc.astype(np.int64))

print(f"\nSaved inputs.npz  shape={q_int8_nchw.shape} dtype=int64 (int8 values, range [{q_int8_nchw.min()},{q_int8_nchw.max()}])")
print(f"Saved outputs.npz shape={x_fc.shape}  dtype=int64 (int8 values)")
print(f"\nTest directory ready: {OUT_DIR}")
print("Run inside container:")
print("  python deeployRunner_gap9.py -t Tests/Models/DSCNNL_INT8 -s gvsoc")
