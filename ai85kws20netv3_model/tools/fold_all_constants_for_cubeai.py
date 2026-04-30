#!/usr/bin/env python3
from pathlib import Path
import re
import numpy as np
import onnx
from onnx import numpy_helper, TensorProto

IN_ONNX = Path.home() / "max78000/stm32u5_kws20/kws20_v3_qat_true2d_nchw_folded_opset11_for_cubeai.onnx"
OUT_ONNX = Path.home() / "max78000/stm32u5_kws20/kws20_v3_qat_true2d_nchw_allconst_folded_for_cubeai.onnx"

model = onnx.load(str(IN_ONNX))
graph = model.graph

def clean_name(s):
    return re.sub(r"[^A-Za-z0-9_]", "_", s)

def get_attr(node, name):
    for a in node.attribute:
        if a.name == name:
            return a
    return None

def tensor_from_attr(attr):
    return numpy_helper.to_array(attr.t)

values = {}

for init in graph.initializer:
    values[init.name] = numpy_helper.to_array(init)

graph_input_names = {x.name for x in graph.input}
graph_output_names = {x.name for x in graph.output}

def eval_node(node):
    op = node.op_type
    ins = []

    for name in node.input:
        if name == "":
            ins.append(None)
        elif name in values:
            ins.append(values[name])
        else:
            return None

    try:
        if op == "Constant":
            a = get_attr(node, "value")
            if a is not None:
                return [tensor_from_attr(a)]

        if op == "Identity":
            return [ins[0]]

        if op == "Add":
            return [ins[0] + ins[1]]
        if op == "Sub":
            return [ins[0] - ins[1]]
        if op == "Mul":
            return [ins[0] * ins[1]]
        if op == "Div":
            return [ins[0] / ins[1]]
        if op == "Neg":
            return [-ins[0]]
        if op == "Abs":
            return [np.abs(ins[0])]
        if op == "Pow":
            return [np.power(ins[0], ins[1])]
        if op == "Reciprocal":
            return [1.0 / ins[0]]
        if op == "Log":
            return [np.log(ins[0])]
        if op == "Floor":
            return [np.floor(ins[0])]
        if op == "Round":
            return [np.round(ins[0])]

        if op == "Clip":
            x = ins[0]
            lo = ins[1] if len(ins) > 1 and ins[1] is not None else -np.inf
            hi = ins[2] if len(ins) > 2 and ins[2] is not None else np.inf
            return [np.clip(x, lo, hi)]

        if op == "ReduceMax":
            axes_attr = get_attr(node, "axes")
            keepdims_attr = get_attr(node, "keepdims")

            axes = None
            keepdims = True

            if axes_attr is not None:
                axes = tuple(axes_attr.ints)
            if keepdims_attr is not None:
                keepdims = bool(keepdims_attr.i)

            return [np.max(ins[0], axis=axes, keepdims=keepdims)]

        if op == "Reshape":
            shape = ins[1].astype(np.int64).tolist()
            return [np.reshape(ins[0], shape)]

        if op == "Transpose":
            perm_attr = get_attr(node, "perm")
            perm = list(perm_attr.ints) if perm_attr is not None else None
            return [np.transpose(ins[0], axes=perm)]

        if op == "Unsqueeze":
            axes_attr = get_attr(node, "axes")
            if axes_attr is not None:
                axes = list(axes_attr.ints)
            else:
                axes = ins[1].astype(np.int64).tolist()

            y = ins[0]
            for ax in sorted(axes):
                y = np.expand_dims(y, axis=ax)
            return [y]

        if op == "Cast":
            to_attr = get_attr(node, "to")
            if to_attr is None:
                return None

            to = to_attr.i

            if to == TensorProto.FLOAT:
                return [ins[0].astype(np.float32)]
            if to == TensorProto.DOUBLE:
                return [ins[0].astype(np.float64)]
            if to == TensorProto.INT64:
                return [ins[0].astype(np.int64)]
            if to == TensorProto.INT32:
                return [ins[0].astype(np.int32)]
            if to == TensorProto.INT8:
                return [ins[0].astype(np.int8)]
            if to == TensorProto.UINT8:
                return [ins[0].astype(np.uint8)]

    except Exception:
        return None

    return None

foldable_outputs = set()
folded_nodes = 0

for node in graph.node:
    outs = eval_node(node)
    if outs is not None and len(outs) == len(node.output):
        for name, val in zip(node.output, outs):
            values[name] = np.asarray(val)
            foldable_outputs.add(name)
        folded_nodes += 1

kept_nodes = []

for node in graph.node:
    # Keep node if it produces a real graph output or if it cannot be folded.
    produces_graph_output = any(out in graph_output_names for out in node.output)
    all_outputs_foldable = all(out in foldable_outputs for out in node.output)

    if produces_graph_output or not all_outputs_foldable:
        kept_nodes.append(node)

# Collect needed inputs of remaining nodes.
needed_inputs = set()
for node in kept_nodes:
    for inp in node.input:
        if inp:
            needed_inputs.add(inp)

# Build new initializers for any folded constants that are now used by remaining nodes.
new_initializers = []
existing_initializer_names = set()

for name in needed_inputs:
    if name in graph_input_names:
        continue
    if name not in values:
        continue

    arr = np.asarray(values[name])

    # CubeAI often dislikes scalar tensors. Use shape [1] for scalar constants.
    if arr.shape == ():
        arr = arr.reshape(1)

    # Avoid float64 constants.
    if arr.dtype == np.float64:
        arr = arr.astype(np.float32)

    # Make integer constants int64 unless already specific.
    if arr.dtype == np.int32:
        arr = arr.astype(np.int64)

    safe_name = clean_name(name)
    if safe_name != name:
        # Rename node inputs to safe initializer name.
        for node in kept_nodes:
            for i, inp in enumerate(node.input):
                if inp == name:
                    node.input[i] = safe_name
        name = safe_name

    if name not in existing_initializer_names:
        new_initializers.append(numpy_helper.from_array(arr, name=name))
        existing_initializer_names.add(name)

# Replace graph nodes and initializers.
while len(graph.node) > 0:
    graph.node.pop()
graph.node.extend(kept_nodes)

while len(graph.initializer) > 0:
    graph.initializer.pop()
graph.initializer.extend(new_initializers)

# Clean value_info.
while len(graph.value_info) > 0:
    graph.value_info.pop()

# Explicit input shape.
inp = graph.input[0]
shape = inp.type.tensor_type.shape
while len(shape.dim) > 0:
    shape.dim.pop()

for d in [1, 1, 128, 128]:
    dim = shape.dim.add()
    dim.dim_value = d
    dim.denotation = ""

# Force opset 11.
for opset in model.opset_import:
    if opset.domain == "" or opset.domain == "ai.onnx":
        opset.version = 11

onnx.checker.check_model(model)
onnx.save(model, str(OUT_ONNX))

print("input:", IN_ONNX)
print("output:", OUT_ONNX)
print("folded nodes:", folded_nodes)
print("kept nodes:", len(graph.node))
print("initializers:", len(graph.initializer))
print("opsets:", [(o.domain, o.version) for o in model.opset_import])
print("size KiB:", OUT_ONNX.stat().st_size / 1024)
