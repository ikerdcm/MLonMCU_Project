#!/usr/bin/env python3
from pathlib import Path
import onnx

IN_ONNX = Path.home() / "max78000/stm32u5_kws20/kws20_v3_qat_true2d_nchw_folded_denoted_for_cubeai.onnx"
OUT_ONNX = Path.home() / "max78000/stm32u5_kws20/kws20_v3_qat_true2d_nchw_cubeai_sanitized.onnx"

model = onnx.load(str(IN_ONNX))
graph = model.graph

real_input_name = "kws_input"

# Alle Initializer-Namen sammeln
initializer_names = {init.name for init in graph.initializer}

# Nur echten Input behalten, keine Initializer/Weight/Shape Inputs
kept_inputs = []
for inp in graph.input:
    if inp.name == real_input_name:
        kept_inputs.append(inp)

if len(kept_inputs) != 1:
    raise SystemExit(f"Could not find exactly one real input named {real_input_name}")

while len(graph.input) > 0:
    graph.input.pop()

graph.input.extend(kept_inputs)

# Shape komplett eindeutig setzen: NCHW = [1,1,128,128]
inp = graph.input[0]
shape = inp.type.tensor_type.shape
while len(shape.dim) > 0:
    shape.dim.pop()

for d in [1, 1, 128, 128]:
    dim = shape.dim.add()
    dim.dim_value = d

# Denotations lieber löschen, CubeAI spinnt manchmal damit
for d in shape.dim:
    d.denotation = ""

# Alte inferred shapes entfernen
while len(graph.value_info) > 0:
    graph.value_info.pop()

onnx.checker.check_model(model)

try:
    model = onnx.shape_inference.infer_shapes(model)
except Exception as e:
    print("shape inference warning:", e)

onnx.save(model, str(OUT_ONNX))

print("wrote:", OUT_ONNX)
print("graph inputs:", len(model.graph.input))
for inp in model.graph.input:
    dims = [d.dim_value if d.dim_value else "?" for d in inp.type.tensor_type.shape.dim]
    print(inp.name, dims)
print("initializers:", len(model.graph.initializer))
