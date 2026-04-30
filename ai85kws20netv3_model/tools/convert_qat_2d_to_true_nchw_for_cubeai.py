#!/usr/bin/env python3
from pathlib import Path
import onnx
from onnx import helper, numpy_helper
import numpy as np

IN_ONNX = Path.home() / "max78000/stm32u5_kws20/kws20_v3_qat_2d_patched_for_stm32u5.onnx"
OUT_ONNX = Path.home() / "max78000/stm32u5_kws20/kws20_v3_qat_true2d_nchw_for_cubeai.onnx"

model = onnx.load(str(IN_ONNX))
graph = model.graph

old_input_name = graph.input[0].name

# External input becomes normal ONNX NCHW image:
# [N,C,H,W] = [1,1,128,128]
inp = graph.input[0]
shape = inp.type.tensor_type.shape
while len(shape.dim) > 0:
    shape.dim.pop()

for d in [1, 1, 128, 128]:
    dim = shape.dim.add()
    dim.dim_value = d

def get_attr_ints(node, name):
    for attr in node.attribute:
        if attr.name == name:
            return list(attr.ints)
    return None

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

# Map producer output -> producer node
producer = {}
for node in graph.node:
    for out in node.output:
        producer[out] = node

conv_nodes = [n for n in graph.node if n.op_type == "Conv"]
if not conv_nodes:
    raise SystemExit("No Conv nodes found")

first_conv = conv_nodes[0]

# First conv currently uses weight shape logically [O,128,K,1].
# We need [O,1,K,128].
old_w_input = first_conv.input[1]

if old_w_input not in producer:
    raise SystemExit(f"Could not find producer of first conv weight: {old_w_input}")

old_unsq = producer[old_w_input]
if old_unsq.op_type != "Unsqueeze":
    raise SystemExit(f"Expected first conv weight producer to be Unsqueeze, got {old_unsq.op_type}")

# This is the tensor before old Unsqueeze, shape logically [O,128,K]
pre_weight = old_unsq.input[0]

w_transposed = pre_weight + "_true2d_transposed"
w_true2d = pre_weight + "_true2d_weight"

# [O,128,K] -> [O,K,128]
w_transpose_node = helper.make_node(
    "Transpose",
    inputs=[pre_weight],
    outputs=[w_transposed],
    perm=[0, 2, 1],
    name="first_conv_weight_transpose_true2d"
)

# [O,K,128] -> [O,1,K,128]
axis_name = "first_conv_weight_unsqueeze_axis_1"
axis_init = numpy_helper.from_array(np.array([1], dtype=np.int64), name=axis_name)
graph.initializer.append(axis_init)

w_unsqueeze_node = helper.make_node(
    "Unsqueeze",
    inputs=[w_transposed, axis_name],
    outputs=[w_true2d],
    name="first_conv_weight_unsqueeze_true2d"
)

# Patch first Conv to use true 2D weight
first_conv.input[1] = w_true2d

# Patch first Conv kernel shape:
# old [K,1] -> new [K,128]
k = get_attr_ints(first_conv, "kernel_shape")
if not k or len(k) != 2:
    raise SystemExit(f"Unexpected first conv kernel_shape: {k}")
set_attr_ints(first_conv, "kernel_shape", [k[0], 128])

s = get_attr_ints(first_conv, "strides")
if s and len(s) == 2:
    set_attr_ints(first_conv, "strides", [s[0], 1])

d = get_attr_ints(first_conv, "dilations")
if d and len(d) == 2:
    set_attr_ints(first_conv, "dilations", [d[0], 1])

p = get_attr_ints(first_conv, "pads")
if p and len(p) == 4:
    # keep height padding, no width padding
    set_attr_ints(first_conv, "pads", [p[0], 0, p[2], 0])

# Insert new weight transform nodes right before first conv
new_nodes = []
inserted = False
for node in graph.node:
    if node is first_conv and not inserted:
        new_nodes.append(w_transpose_node)
        new_nodes.append(w_unsqueeze_node)
        inserted = True
    new_nodes.append(node)

while len(graph.node) > 0:
    graph.node.pop()
graph.node.extend(new_nodes)

# Clear old shape inference info
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
print("external input shape: [1,1,128,128]")
print("first conv patched to kernel [K,128] with weight [O,1,K,128]")
print("size KiB:", OUT_ONNX.stat().st_size / 1024)
