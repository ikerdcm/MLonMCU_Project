#!/usr/bin/env python3
from pathlib import Path
import onnx
from onnx import helper, numpy_helper
import numpy as np

IN_ONNX = Path.home() / "max78000/stm32u5_kws20/kws20_v3_qat_2d_patched_for_stm32u5.onnx"
OUT_ONNX = Path.home() / "max78000/stm32u5_kws20/kws20_v3_qat_2d_flat_input_for_cubeai.onnx"

model = onnx.load(str(IN_ONNX))
graph = model.graph

old_input_name = graph.input[0].name

reshape_out = old_input_name + "_reshaped_1x128x128"
transpose_out = old_input_name + "_transposed_1x128x128"
internal_4d = old_input_name + "_internal_1x128x128x1"

# External CubeAI input: flat vector [1, 16384]
inp = graph.input[0]
inp.name = old_input_name

shape = inp.type.tensor_type.shape
while len(shape.dim) > 0:
    shape.dim.pop()

for d in [1, 16384]:
    dim = shape.dim.add()
    dim.dim_value = d

# Replace all original uses of input with internal 4D tensor
for node in graph.node:
    for i, name in enumerate(node.input):
        if name == old_input_name:
            node.input[i] = internal_4d

# Reshape [1,16384] -> [1,128,128]
shape_name = "cubeai_flat_to_1x128x128_shape"
shape_init = numpy_helper.from_array(
    np.array([1, 128, 128], dtype=np.int64),
    name=shape_name
)
graph.initializer.append(shape_init)

reshape_node = helper.make_node(
    "Reshape",
    inputs=[old_input_name, shape_name],
    outputs=[reshape_out],
    name="cubeai_flat_to_1x128x128"
)

# Bake in working transpose=1
transpose_node = helper.make_node(
    "Transpose",
    inputs=[reshape_out],
    outputs=[transpose_out],
    perm=[0, 2, 1],
    name="cubeai_internal_transpose"
)

# Unsqueeze [1,128,128] -> [1,128,128,1]
axes_name = "cubeai_unsqueeze_axis_3"
axes_init = numpy_helper.from_array(
    np.array([3], dtype=np.int64),
    name=axes_name
)
graph.initializer.append(axes_init)

unsqueeze_node = helper.make_node(
    "Unsqueeze",
    inputs=[transpose_out, axes_name],
    outputs=[internal_4d],
    name="cubeai_internal_unsqueeze_to_4d"
)

nodes = [reshape_node, transpose_node, unsqueeze_node]
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
print("external input shape: [1,16384]")
print("internal path: Reshape -> Transpose -> Unsqueeze -> [1,128,128,1]")
print("size KiB:", OUT_ONNX.stat().st_size / 1024)
