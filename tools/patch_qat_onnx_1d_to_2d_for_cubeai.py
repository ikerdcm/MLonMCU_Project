#!/usr/bin/env python3
from pathlib import Path
import onnx
from onnx import numpy_helper
import numpy as np

IN_ONNX = Path.home() / "max78000/stm32u5_kws20/kws20_v3_qat_onnx_for_stm32u5.onnx"
OUT_ONNX = Path.home() / "max78000/stm32u5_kws20/kws20_v3_qat_2d_patched_for_stm32u5.onnx"

model = onnx.load(str(IN_ONNX))
graph = model.graph

# Input von [1,128,128] auf [1,128,128,1] setzen
inp = graph.input[0]
shape = inp.type.tensor_type.shape
while len(shape.dim) > 0:
    shape.dim.pop()
for d in [1, 128, 128, 1]:
    dim = shape.dim.add()
    dim.dim_value = d

# Initializer dict
init_map = {init.name: init for init in graph.initializer}

def set_attr_ints(node, name, values):
    for attr in node.attribute:
        if attr.name == name:
            attr.ints[:] = values
            return
    attr = node.attribute.add()
    attr.name = name
    attr.ints[:] = values
    attr.type = onnx.AttributeProto.INTS

def get_attr_ints(node, name):
    for attr in node.attribute:
        if attr.name == name:
            return list(attr.ints)
    return None

patched_conv = 0
patched_pool = 0

for node in graph.node:
    if node.op_type == "Conv":
        # Weight von [out,in,k] auf [out,in,k,1]
        if len(node.input) >= 2 and node.input[1] in init_map:
            w_init = init_map[node.input[1]]
            w = numpy_helper.to_array(w_init)
            if w.ndim == 3:
                w2 = w[:, :, :, None].astype(w.dtype)
                new_init = numpy_helper.from_array(w2, name=w_init.name)
                w_init.CopyFrom(new_init)
                patched_conv += 1

        k = get_attr_ints(node, "kernel_shape")
        if k and len(k) == 1:
            set_attr_ints(node, "kernel_shape", [k[0], 1])

        s = get_attr_ints(node, "strides")
        if s and len(s) == 1:
            set_attr_ints(node, "strides", [s[0], 1])

        d = get_attr_ints(node, "dilations")
        if d and len(d) == 1:
            set_attr_ints(node, "dilations", [d[0], 1])

        p = get_attr_ints(node, "pads")
        if p and len(p) == 2:
            set_attr_ints(node, "pads", [p[0], 0, p[1], 0])

    elif node.op_type in ("MaxPool", "AveragePool"):
        k = get_attr_ints(node, "kernel_shape")
        if k and len(k) == 1:
            set_attr_ints(node, "kernel_shape", [k[0], 1])
            patched_pool += 1

        s = get_attr_ints(node, "strides")
        if s and len(s) == 1:
            set_attr_ints(node, "strides", [s[0], 1])

        d = get_attr_ints(node, "dilations")
        if d and len(d) == 1:
            set_attr_ints(node, "dilations", [d[0], 1])

        p = get_attr_ints(node, "pads")
        if p and len(p) == 2:
            set_attr_ints(node, "pads", [p[0], 0, p[1], 0])

# Alte inferred value_info Shapes löschen, damit ONNX nicht an alten 3D-Shapes hängt
del graph.value_info[:]

onnx.checker.check_model(model)

try:
    model = onnx.shape_inference.infer_shapes(model)
except Exception as e:
    print("shape inference warning:", e)

onnx.save(model, str(OUT_ONNX))

print("input :", IN_ONNX)
print("output:", OUT_ONNX)
print("patched conv weights:", patched_conv)
print("patched pool nodes:", patched_pool)
print("size KiB:", OUT_ONNX.stat().st_size / 1024)
