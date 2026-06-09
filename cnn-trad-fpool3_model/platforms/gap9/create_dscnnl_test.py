#!/usr/bin/env python3
"""
Create Deeploy test directory for ds_cnn_l.onnx (float32) on GAP9.

Generates:
  Tests/Models/DSCNNL/network.onnx  -- batch dim fixed to 1
  Tests/Models/DSCNNL/inputs.npz    -- 1 synthetic MFCC sample
  Tests/Models/DSCNNL/outputs.npz   -- reference output from ONNX inference
"""

import os
import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper, TensorProto, helper

DEEPLOY_TEST_DIR = os.path.join(os.path.dirname(__file__),
                                "Deeploy/DeeployTest")
OUT_DIR = os.path.join(DEEPLOY_TEST_DIR, "Tests/Models/DSCNNL")
MODEL_PATH = os.path.join(os.path.dirname(__file__),
                          "../../models/ds_cnn_l.onnx")

os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Load and fix batch dim ─────────────────────────────────────────────────
model = onnx.load(MODEL_PATH)

# Fix dynamic batch dim (0 → 1) in input and output
for inp in model.graph.input:
    dim = inp.type.tensor_type.shape.dim[0]
    if dim.dim_value == 0:
        dim.dim_value = 1

for out in model.graph.output:
    dim = out.type.tensor_type.shape.dim[0]
    if dim.dim_value == 0:
        dim.dim_value = 1

onnx.checker.check_model(model)
model = onnx.shape_inference.infer_shapes(model)

# ── 1b. Fold Conv → Mul(BN_scale) → Add(BN_shift) into Conv ─────────────────
# Keras exports BatchNorm as separate Mul+Add ops after Conv.
# Deeploy's PULPFPConv2DParser only handles Conv+bias, not Conv+Mul+Add.
# Fold: new_W[oc] = W[oc] * scale[oc],  new_b[oc] = b[oc]*scale[oc] + shift[oc]

def fold_conv_bn(model: onnx.ModelProto) -> onnx.ModelProto:
    # Build initializer map: name → numpy array
    inits = {t.name: numpy_helper.to_array(t).copy() for t in model.graph.initializer}

    # Build tensor→consuming-node map (one consumer assumed per tensor)
    inp2node = {}
    for node in model.graph.node:
        for i in node.input:
            inp2node[i] = node

    nodes_to_remove = set()
    rewrites = {}  # old_output_name → new_output_name after folding

    for node in model.graph.node:
        if node.op_type != 'Conv':
            continue
        conv_out = node.output[0]
        if conv_out not in inp2node:
            continue
        mul_node = inp2node[conv_out]
        if mul_node.op_type != 'Mul':
            continue
        mul_out = mul_node.output[0]
        if mul_out not in inp2node:
            continue
        add_node = inp2node[mul_out]
        if add_node.op_type != 'Add':
            continue

        # Identify scale/shift initializers
        scale_name = (mul_node.input[1] if mul_node.input[0] == conv_out
                      else mul_node.input[0])
        shift_name = (add_node.input[1] if add_node.input[0] == mul_out
                      else add_node.input[0])
        if scale_name not in inits or shift_name not in inits:
            continue

        scale = inits[scale_name].squeeze()  # [C]
        shift = inits[shift_name].squeeze()  # [C]

        W_name = node.input[1]
        W = inits[W_name]  # [OC, IC, kH, kW]
        new_W = W * scale[:, None, None, None]
        inits[W_name] = new_W

        if len(node.input) >= 3 and node.input[2]:
            b_name = node.input[2]
            b = inits[b_name]  # [OC]
            inits[b_name] = b * scale + shift
        else:
            # No existing bias — add one
            b_name = node.name + '_folded_bias'
            inits[b_name] = shift.copy()
            node.input.append(b_name)

        # Redirect Add's output consumers to Conv's output
        rewrites[add_node.output[0]] = conv_out
        # Update Conv output name to carry through
        node.output[0] = conv_out  # stays the same
        # Mark Mul and Add for removal
        nodes_to_remove.add(id(mul_node))
        nodes_to_remove.add(id(add_node))

    if not nodes_to_remove:
        return model

    # Rebuild graph without folded Mul/Add nodes, applying rewrites
    new_nodes = []
    for node in model.graph.node:
        if id(node) in nodes_to_remove:
            continue
        # Rewrite inputs that point to removed Add outputs
        for i, inp in enumerate(node.input):
            if inp in rewrites:
                node.input[i] = rewrites[inp]
        new_nodes.append(node)

    # Rebuild initializers with updated weights
    new_inits = []
    init_names = {t.name for t in model.graph.initializer}
    for t in model.graph.initializer:
        new_val = inits[t.name]
        new_t = numpy_helper.from_array(new_val, name=t.name)
        new_inits.append(new_t)
    # Add any newly created bias initializers
    for name, val in inits.items():
        if name not in init_names:
            new_inits.append(numpy_helper.from_array(val, name=name))

    # Rebuild the graph outputs (rewire if needed)
    new_outputs = []
    for out in model.graph.output:
        name = rewrites.get(out.name, out.name)
        new_outputs.append(out if name == out.name else
                           helper.make_tensor_value_info(name,
                               out.type.tensor_type.elem_type,
                               [d.dim_value for d in out.type.tensor_type.shape.dim]))

    new_graph = helper.make_graph(new_nodes, model.graph.name,
                                  list(model.graph.input), new_outputs, new_inits)
    new_model = helper.make_model(new_graph, opset_imports=model.opset_import)
    new_model.ir_version = model.ir_version
    new_model = onnx.shape_inference.infer_shapes(new_model)
    print(f"  BN fold: removed {len(nodes_to_remove)} Mul/Add nodes")
    return new_model

print("Folding Conv→Mul→Add (BatchNorm) patterns...")
model = fold_conv_bn(model)

# ── 2. Load real MFCC test vector (same sample used for STM32U5 / MAX78000) ──
# Source: ds_cnn_test_input_left.h — TF-MFCC of "left_b528edb3_nohash_0.wav"
# Expected class: 2 (left), confidence >99.9%
import re

_header = os.path.join(os.path.dirname(__file__),
    "../../platforms/max78000/keyword_spotting_max78000_ds_cnn/ds_cnn_test_input_left.h")
with open(_header) as _f:
    _txt = _f.read()
_nums = re.findall(r'[-+]?\d*\.?\d+e[+-]?\d+f|[-+]?\d+\.\d+f', _txt)
test_input = np.array([float(v.rstrip('f')) for v in _nums], dtype=np.float32).reshape(1, 49, 10, 1)
assert test_input.shape == (1, 49, 10, 1), f"Unexpected shape: {test_input.shape}"
print(f"Loaded real MFCC: shape={test_input.shape} range=[{test_input.min():.2f}, {test_input.max():.2f}]")

# ── 3. Run ORT reference inference on BN-folded model ────────────────────────
# BN fold + pad fix are both math-equivalent transforms so the reference output
# from the BN-folded model equals what the Deeploy-compiled model will produce.
_tmp_bn = os.path.join(OUT_DIR, "_tmp_bn_folded.onnx")
onnx.save(model, _tmp_bn)

sess_opts = ort.SessionOptions()
sess_opts.log_severity_level = 3
sess = ort.InferenceSession(_tmp_bn, sess_opts)

input_name  = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name
print(f"Input:  {input_name} {sess.get_inputs()[0].shape}")
print(f"Output: {output_name} {sess.get_outputs()[0].shape}")

ref_output = sess.run([output_name], {input_name: test_input})[0]  # [1, 12]
print(f"Reference output (logits): {ref_output[0]}")
print(f"Predicted class: {np.argmax(ref_output[0])}")

os.remove(_tmp_bn)

# ── 1c. Convert asymmetric Conv padding to explicit Pad + Conv(pads=0) ────────
# Deeploy PadParser (Generic) requires:
#   - len(node.inputs) == 1  (old ONNX opset ≤ 10 attribute-style Pad)
#   - 'pads' in node.attrs   (pads as node attribute, NOT as a tensor input)
# Keras SAME-padding with stride>1 on odd input dims produces asymmetric pads.
# We insert an attribute-style Pad node (single input, pads/mode/value as attrs)
# and manually add the output shape to value_info since standard shape inference
# won't handle old-style Pad in an opset-17 model.

def fix_asymmetric_conv_pads(model: onnx.ModelProto) -> onnx.ModelProto:
    shapes = {}
    for v in list(model.graph.value_info) + list(model.graph.input):
        shapes[v.name] = [d.dim_value for d in v.type.tensor_type.shape.dim]

    new_nodes = []
    new_inits = list(model.graph.initializer)
    new_value_infos = list(model.graph.value_info)
    pad_counter = 0

    for node in model.graph.node:
        if node.op_type != 'Conv':
            new_nodes.append(node)
            continue

        pad_attr = next((a for a in node.attribute if a.name == 'pads'), None)
        if pad_attr is None:
            new_nodes.append(node)
            continue

        p = list(pad_attr.ints)  # [top, left, bottom, right]
        if len(p) != 4:
            new_nodes.append(node)
            continue

        top, left, bottom, right = p
        if top == bottom and left == right:  # already symmetric — Conv parser OK
            new_nodes.append(node)
            continue

        data_in_name = node.input[0]
        padded_name = f'_padded_{pad_counter}'
        pad_counter += 1

        # Attribute-style Pad (opset ≤ 10): single input, pads/mode/value as attrs.
        # onnx_graphsurgeon exposes these as node.attrs['pads'] etc., which is exactly
        # what Deeploy's PadParser and Pad2DParser check for.
        pad_node = helper.make_node(
            'Pad',
            inputs=[data_in_name],
            outputs=[padded_name],
            name=f'_auto_pad_{pad_counter}',
            mode='constant',
            pads=[0, 0, top, left, 0, 0, bottom, right],
            value=0.0,
        )
        new_nodes.append(pad_node)

        # Manually supply the output shape — onnx shape inference won't handle
        # old-style Pad in an opset-17 model, but gs reads value_info for shapes.
        in_shape = shapes.get(data_in_name)
        if in_shape and len(in_shape) == 4:
            out_shape = [in_shape[0], in_shape[1],
                         in_shape[2] + top + bottom,
                         in_shape[3] + left + right]
            new_value_infos.append(
                helper.make_tensor_value_info(padded_name, TensorProto.FLOAT, out_shape))

        # Conv with zero padding
        new_node = helper.make_node(
            'Conv',
            inputs=[padded_name] + list(node.input[1:]),
            outputs=list(node.output),
            name=node.name,
        )
        for attr in node.attribute:
            if attr.name == 'pads':
                new_node.attribute.append(onnx.helper.make_attribute('pads', [0, 0, 0, 0]))
            else:
                new_node.attribute.append(attr)
        new_nodes.append(new_node)
        print(f"  Pad fix: Conv '{node.name}' pads {p} → old-style Pad attr + Conv(pads=0)")

    new_graph = helper.make_graph(new_nodes, model.graph.name,
                                  list(model.graph.input),
                                  list(model.graph.output),
                                  new_inits,
                                  value_info=new_value_infos)
    new_model = helper.make_model(new_graph, opset_imports=model.opset_import)
    new_model.ir_version = model.ir_version
    return new_model

print("Fixing asymmetric Conv padding...")
model = fix_asymmetric_conv_pads(model)
# Note: skip onnx.checker here — old-style attribute Pad is opset ≤ 10 format;
# checker would reject it in an opset-17 model, but onnx_graphsurgeon handles it.

out_onnx = os.path.join(OUT_DIR, "network.onnx")
onnx.save(model, out_onnx)
print(f"Saved: {out_onnx}")

# ── 4. Save npz files (Deeploy expects keys matching tensor names) ────────────
inp_key = input_name.replace(":", "_").replace("/", "_")
out_key = output_name.replace(":", "_").replace("/", "_")

np.savez(os.path.join(OUT_DIR, "inputs.npz"),  **{inp_key: test_input})
np.savez(os.path.join(OUT_DIR, "outputs.npz"), **{out_key: ref_output})

print(f"Saved inputs.npz  key='{inp_key}'  shape={test_input.shape}  dtype={test_input.dtype}")
print(f"Saved outputs.npz key='{out_key}'  shape={ref_output.shape}  dtype={ref_output.dtype}")
print(f"\nTest directory ready: {OUT_DIR}")
print("Run inside container:")
print("  python deeployRunner_gap9.py -t Tests/Models/DSCNNL -s board")
