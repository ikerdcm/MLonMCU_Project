#!/usr/bin/env python3
from pathlib import Path
import onnx
from onnx import numpy_helper

IN_ONNX = Path.home() / "max78000/stm32u5_kws20/kws20_v3_qat_true2d_nchw_folded_weights_for_cubeai.onnx"
OUT_ONNX = Path.home() / "max78000/stm32u5_kws20/kws20_v3_qat_true2d_nchw_folded_opset11_for_cubeai.onnx"

model = onnx.load(str(IN_ONNX))
graph = model.graph

init_map = {i.name: i for i in graph.initializer}

# Convert Unsqueeze from opset >=13 style:
#   Unsqueeze(x, axes_tensor)
# to opset 11 style:
#   Unsqueeze(x) with axes attribute
for node in graph.node:
    if node.op_type == "Unsqueeze" and len(node.input) == 2:
        axes_name = node.input[1]
        if axes_name not in init_map:
            raise SystemExit(f"Unsqueeze axes not initializer: {node.name} {axes_name}")

        axes = numpy_helper.to_array(init_map[axes_name]).astype("int64").tolist()

        # Remove second input
        while len(node.input) > 1:
            node.input.pop()

        # Remove old axes attr if any
        keep_attrs = [a for a in node.attribute if a.name != "axes"]
        while len(node.attribute) > 0:
            node.attribute.pop()
        node.attribute.extend(keep_attrs)

        attr = node.attribute.add()
        attr.name = "axes"
        attr.ints.extend(axes)
        attr.type = onnx.AttributeProto.INTS

# Make graph input completely explicit
inp = graph.input[0]
shape = inp.type.tensor_type.shape
while len(shape.dim) > 0:
    shape.dim.pop()
for d in [1, 1, 128, 128]:
    dim = shape.dim.add()
    dim.dim_value = d
    dim.denotation = ""

# Remove all value_info. Do NOT run shape inference again.
while len(graph.value_info) > 0:
    graph.value_info.pop()

# Remove unused initializers
used = set()
for node in graph.node:
    for x in node.input:
        if x:
            used.add(x)

kept = [i for i in graph.initializer if i.name in used]
while len(graph.initializer) > 0:
    graph.initializer.pop()
graph.initializer.extend(kept)

# Force ONNX opset 11 for standard domain
for opset in model.opset_import:
    if opset.domain == "" or opset.domain == "ai.onnx":
        opset.version = 11

onnx.checker.check_model(model)
onnx.save(model, str(OUT_ONNX))

print("wrote:", OUT_ONNX)
print("graph input:", graph.input[0].name, [d.dim_value for d in graph.input[0].type.tensor_type.shape.dim])
print("nodes:", len(graph.node))
print("initializers:", len(graph.initializer))
print("opsets:", [(o.domain, o.version) for o in model.opset_import])
