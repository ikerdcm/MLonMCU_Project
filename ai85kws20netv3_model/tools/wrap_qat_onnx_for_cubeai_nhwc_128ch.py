#!/usr/bin/env python3
from pathlib import Path
import onnx
from onnx import helper

IN_ONNX = Path.home() / "max78000/stm32u5_kws20/kws20_v3_qat_2d_patched_for_stm32u5.onnx"
OUT_ONNX = Path.home() / "max78000/stm32u5_kws20/kws20_v3_qat_cubeai_nhwc_128ch_input.onnx"

model = onnx.load(str(IN_ONNX))
graph = model.graph

old_input_name = graph.input[0].name
internal_name = old_input_name + "_internal_nchw"

# External CubeAI-friendly-ish input:
# NHWC = [N,H,W,C] = [1,128,1,128]
inp = graph.input[0]
inp.name = old_input_name

shape = inp.type.tensor_type.shape
while len(shape.dim) > 0:
    shape.dim.pop()

for d in [1, 128, 1, 128]:
    dim = shape.dim.add()
    dim.dim_value = d

# Replace all uses of external input with internal tensor
for node in graph.node:
    for i, name in enumerate(node.input):
        if name == old_input_name:
            node.input[i] = internal_name

# External [1,128,1,128] NHWC -> internal [1,128,128,1] NCHW-like
transpose_node = helper.make_node(
    "Transpose",
    inputs=[old_input_name],
    outputs=[internal_name],
    perm=[0, 3, 1, 2],
    name="cubeai_input_nhwc_128ch_to_internal"
)

nodes = [transpose_node]
nodes.extend(graph.node)

while len(graph.node) > 0:
    graph.node.pop()
graph.node.extend(nodes)

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
print("external input shape: [1,128,1,128]")
print("first node: Transpose [0,3,1,2]")
print("size KiB:", OUT_ONNX.stat().st_size / 1024)
