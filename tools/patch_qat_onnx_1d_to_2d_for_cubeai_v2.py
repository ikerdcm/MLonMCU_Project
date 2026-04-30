#!/usr/bin/env python3
from pathlib import Path
import onnx
from onnx import helper, numpy_helper
import numpy as np

IN_ONNX = Path.home() / "max78000/stm32u5_kws20/kws20_v3_qat_onnx_for_stm32u5.onnx"
OUT_ONNX = Path.home() / "max78000/stm32u5_kws20/kws20_v3_qat_2d_patched_for_stm32u5.onnx"

model = onnx.load(str(IN_ONNX))
graph = model.graph

# Input [1,128,128] -> [1,128,128,1]
inp = graph.input[0]
shape = inp.type.tensor_type.shape
while len(shape.dim) > 0:
    shape.dim.pop()
for d in [1, 128, 128, 1]:
    dim = shape.dim.add()
    dim.dim_value = d

init_map = {init.name: init for init in graph.initializer}

def set_attr_ints(node, name, values):
    for attr in node.attribute:
        if attr.name == name:
            while len(attr.ints) > 0:
                attr.ints.pop()
            attr.ints.extend(values)
            return
    attr = node.attribute.add()
    attr.name = name
    attr.ints.extend(values)
    attr.type = onnx.AttributeProto.INTS

def get_attr_ints(node, name):
    for attr in node.attribute:
        if attr.name == name:
            return list(attr.ints)
    return None

# Opset check: Unsqueeze in opset 13 needs axes as input tensor.
opset = 13
for imp in model.opset_import:
    if imp.domain == "" or imp.domain == "ai.onnx":
        opset = imp.version

new_nodes = []
patched_conv_attr = 0
patched_conv_init = 0
inserted_unsqueeze = 0
patched_pool = 0

for node in graph.node:
    if node.op_type == "Conv":
        # Patch Conv attributes from 1D to 2D
        k = get_attr_ints(node, "kernel_shape")
        if k and len(k) == 1:
            set_attr_ints(node, "kernel_shape", [k[0], 1])
            patched_conv_attr += 1

        s = get_attr_ints(node, "strides")
        if s and len(s) == 1:
            set_attr_ints(node, "strides", [s[0], 1])

        d = get_attr_ints(node, "dilations")
        if d and len(d) == 1:
            set_attr_ints(node, "dilations", [d[0], 1])

        p = get_attr_ints(node, "pads")
        if p and len(p) == 2:
            set_attr_ints(node, "pads", [p[0], 0, p[1], 0])

        # If weight is direct initializer, reshape [O,I,K] -> [O,I,K,1]
        wname = node.input[1]
        if wname in init_map:
            w_init = init_map[wname]
            w = numpy_helper.to_array(w_init)
            if w.ndim == 3:
                w2 = w[:, :, :, None].astype(w.dtype)
                w_init.CopyFrom(numpy_helper.from_array(w2, name=w_init.name))
                patched_conv_init += 1
        else:
            # Weight is produced by fake-quant subgraph. Insert Unsqueeze on axis 3.
            axes_name = f"{node.name or 'conv'}_weight_unsq_axes_{inserted_unsqueeze}"
            unsq_out = f"{node.input[1]}_unsq4d_for_{node.name or inserted_unsqueeze}"

            axes_init = numpy_helper.from_array(np.array([3], dtype=np.int64), name=axes_name)
            graph.initializer.append(axes_init)

            unsq_node = helper.make_node(
                "Unsqueeze",
                inputs=[node.input[1], axes_name],
                outputs=[unsq_out],
                name=f"{node.name or 'conv'}_weight_unsqueeze"
            )
            new_nodes.append(unsq_node)
            node.input[1] = unsq_out
            inserted_unsqueeze += 1

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

    new_nodes.append(node)

# Replace node list
while len(graph.node) > 0:
    graph.node.pop()
graph.node.extend(new_nodes)

# Clear old shape info
while len(graph.value_info) > 0:
    graph.value_info.pop()

onnx.checker.check_model(model)

try:
    model = onnx.shape_inference.infer_shapes(model)
except Exception as e:
    print("shape inference warning:", e)

onnx.save(model, str(OUT_ONNX))

print("input :", IN_ONNX)
print("output:", OUT_ONNX)
print("patched conv attrs:", patched_conv_attr)
print("patched direct conv initializers:", patched_conv_init)
print("inserted conv weight unsqueeze:", inserted_unsqueeze)
print("patched pool nodes:", patched_pool)
print("size KiB:", OUT_ONNX.stat().st_size / 1024)
