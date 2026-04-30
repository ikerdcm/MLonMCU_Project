#!/usr/bin/env python3
from pathlib import Path
import re
import numpy as np
import onnx
from onnx import numpy_helper, TensorProto

IN_ONNX = Path.home() / "max78000/stm32u5_kws20/kws20_v3_qat_true2d_nchw_for_cubeai.onnx"
OUT_ONNX = Path.home() / "max78000/stm32u5_kws20/kws20_v3_qat_true2d_nchw_folded_weights_for_cubeai.onnx"

model = onnx.load(str(IN_ONNX))
graph = model.graph

def clean_name(s):
    return re.sub(r"[^A-Za-z0-9_]", "_", s)

def get_attr(node, name, default=None):
    for a in node.attribute:
        if a.name == name:
            return a
    return default

def tensor_from_attr(attr):
    return numpy_helper.to_array(attr.t)

values = {}
for init in graph.initializer:
    values[init.name] = numpy_helper.to_array(init)

graph_input_names = {x.name for x in graph.input}

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
            lo = ins[1] if len(ins) > 1 and ins[1] is not None else None
            hi = ins[2] if len(ins) > 2 and ins[2] is not None else None
            if lo is None:
                lo = -np.inf
            if hi is None:
                hi = np.inf
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
            if len(ins) >= 2 and ins[1] is not None:
                axes = ins[1].astype(np.int64).tolist()
            else:
                axes_attr = get_attr(node, "axes")
                axes = list(axes_attr.ints)
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
            if to == TensorProto.INT64:
                return [ins[0].astype(np.int64)]
            if to == TensorProto.INT32:
                return [ins[0].astype(np.int32)]
            if to == TensorProto.UINT8:
                return [ins[0].astype(np.uint8)]
            if to == TensorProto.INT8:
                return [ins[0].astype(np.int8)]

    except Exception:
        return None

    return None

folded_nodes = 0
for node in graph.node:
    outs = eval_node(node)
    if outs is not None and len(outs) == len(node.output):
        for name, val in zip(node.output, outs):
            values[name] = np.asarray(val)
        folded_nodes += 1

new_initializers = []
replaced = 0

for node in graph.node:
    if node.op_type in ("Conv", "Gemm"):
        for idx in [1, 2]:
            if idx < len(node.input):
                name = node.input[idx]
                if name not in graph_input_names and name in values:
                    arr = np.asarray(values[name])
                    # CubeAI expects float weights here
                    if arr.dtype == np.float64:
                        arr = arr.astype(np.float32)
                    new_name = clean_name(f"{node.name or node.op_type}_input{idx}_folded")
                    new_initializers.append(numpy_helper.from_array(arr, name=new_name))
                    node.input[idx] = new_name
                    replaced += 1

graph.initializer.extend(new_initializers)

# Prune unreachable nodes after replacing weights/biases
producer = {}
for node in graph.node:
    for out in node.output:
        producer[out] = node

needed = set(o.name for o in graph.output)
kept_rev = []

for node in reversed(graph.node):
    if any(out in needed for out in node.output):
        kept_rev.append(node)
        for inp in node.input:
            if inp:
                needed.add(inp)

kept = list(reversed(kept_rev))

while len(graph.node) > 0:
    graph.node.pop()
graph.node.extend(kept)

# Keep only initializers still needed
needed_inputs = set()
for node in graph.node:
    for inp in node.input:
        if inp:
            needed_inputs.add(inp)

kept_inits = [init for init in graph.initializer if init.name in needed_inputs]

while len(graph.initializer) > 0:
    graph.initializer.pop()
graph.initializer.extend(kept_inits)

while len(graph.value_info) > 0:
    graph.value_info.pop()

onnx.checker.check_model(model)

try:
    model = onnx.shape_inference.infer_shapes(model)
except Exception as e:
    print("shape inference warning:", e)

onnx.save(model, str(OUT_ONNX))

print("input:", IN_ONNX)
print("output:", OUT_ONNX)
print("constant-folded nodes:", folded_nodes)
print("replaced Conv/Gemm weight/bias inputs:", replaced)
print("remaining nodes:", len(graph.node))
print("remaining initializers:", len(graph.initializer))
print("size KiB:", OUT_ONNX.stat().st_size / 1024)
