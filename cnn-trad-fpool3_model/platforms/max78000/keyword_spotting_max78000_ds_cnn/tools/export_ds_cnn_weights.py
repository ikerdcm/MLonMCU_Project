#!/usr/bin/env python3
"""
Export DS-CNN-L weights from ONNX model, BN-fold, and write ds_cnn_weights.h.

Usage:
    /home/pascal/max78000/ai8x-synthesis/.venv/bin/python tools/export_ds_cnn_weights.py

Output: ds_cnn_weights.h (in the project root, or wherever the script is invoked from).
Run from the project root: platforms/max78000/keyword_spotting_max78000_ds_cnn/
"""

import sys
import os
import numpy as np

try:
    import onnx
    from onnx import numpy_helper
except ImportError:
    print("ERROR: onnx not available in this venv", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
ONNX_PATH   = os.path.join(PROJECT_DIR, "..", "..", "..", "models", "ds_cnn_l.onnx")
ONNX_PATH   = os.path.normpath(ONNX_PATH)
OUT_PATH    = os.path.join(PROJECT_DIR, "ds_cnn_weights.h")

print(f"Loading ONNX from: {ONNX_PATH}")

model = onnx.load(ONNX_PATH)
graph = model.graph

# ---------------------------------------------------------------------------
# Build initializer lookup (name -> numpy array)
# ---------------------------------------------------------------------------
inits = {}
for init in graph.initializer:
    inits[init.name] = numpy_helper.to_array(init).astype(np.float32)

# ---------------------------------------------------------------------------
# Walk nodes and collect layers
# ---------------------------------------------------------------------------
# We expect a repeating pattern:
#   Conv -> Mul(BN scale) -> Add(BN shift) -> Relu
# plus AvgPool, Flatten, Gemm(Dense), Softmax at the end.
#
# BN fold: w_folded[c] = w[c] * scale[c]
#          b_folded[c] = bias[c] * scale[c] + shift[c]

nodes = list(graph.node)

def get_init(name):
    """Return initializer array or None if it is a graph input (not a weight)."""
    return inits.get(name, None)

# Layer naming plan (from the task spec):
#   CONV0  -> first conv (standard)
#   DW1/PW2, DW3/PW4, DW5/PW6, DW7/PW8, DW9/PW10, DW11/PW12
# We collect them in order and assign names after parsing.

class ConvLayer:
    def __init__(self, name, w, b, is_dw, strides, pads):
        self.name    = name   # e.g. "CONV0", "DW1", "PW2", ...
        self.w       = w      # float32 ndarray, folded
        self.b       = b      # float32 ndarray, folded
        self.is_dw   = is_dw
        self.strides = strides
        self.pads    = pads   # [pt, pl, pb, pr]  (ONNX order: top, left, bottom, right)

layers = []  # ConvLayer objects, in order

# Dense weights
dense_w = None
dense_b = None

i = 0
while i < len(nodes):
    node = nodes[i]
    op   = node.op_type

    if op == "Reshape":
        i += 1
        continue

    if op == "Conv":
        w_name = node.input[1]
        b_name = node.input[2] if len(node.input) > 2 else None
        w = inits[w_name].copy()   # (out_ch, in_ch/groups, kH, kW)
        b = inits[b_name].copy() if b_name and b_name in inits else np.zeros(w.shape[0], dtype=np.float32)

        # Read conv attributes
        strides = [1, 1]
        pads    = [0, 0, 0, 0]
        group   = 1
        for attr in node.attribute:
            if attr.name == "strides":
                strides = list(attr.ints)
            elif attr.name == "pads":
                pads = list(attr.ints)
            elif attr.name == "group":
                group = attr.i
        is_dw = (group > 1)

        # Expect next: Mul (BN scale), then Add (BN shift), then Relu
        mul_node = nodes[i+1] if (i+1) < len(nodes) else None
        add_node = nodes[i+2] if (i+2) < len(nodes) else None
        # Relu is optional at position i+3

        scale = np.ones(w.shape[0], dtype=np.float32)
        shift = np.zeros(w.shape[0], dtype=np.float32)

        if mul_node is not None and mul_node.op_type == "Mul":
            # input[1] should be the scale tensor of shape (1, out_ch, 1, 1)
            s_name = mul_node.input[1] if mul_node.input[0] != w_name else mul_node.input[0]
            # pick whichever input is an initializer
            for inp in mul_node.input:
                if inp in inits:
                    scale = inits[inp].flatten().astype(np.float32)
                    break
            i_inc = 1
        else:
            i_inc = 0

        if add_node is not None and add_node.op_type == "Add":
            for inp in add_node.input:
                if inp in inits:
                    shift = inits[inp].flatten().astype(np.float32)
                    break
            i_inc += 1

        # Skip Relu
        relu_idx = i + 1 + i_inc
        if relu_idx < len(nodes) and nodes[relu_idx].op_type == "Relu":
            i_inc += 1

        # BN fold
        # w shape: (out_ch, in_ch/groups, kH, kW)
        out_ch = w.shape[0]
        for c in range(out_ch):
            w[c] = w[c] * scale[c]
            b[c] = b[c] * scale[c] + shift[c]

        layers.append((w, b, is_dw, strides, pads))
        i += 1 + i_inc
        continue

    if op == "AveragePool":
        # Nothing to store — handled directly in inference engine
        i += 1
        continue

    if op == "Flatten":
        i += 1
        continue

    if op == "Gemm":
        # Dense: input[1] = W (transposed in Gemm: (out, in)), input[2] = B
        w_name = node.input[1]
        b_name = node.input[2] if len(node.input) > 2 else None
        dense_w = inits[w_name].copy().astype(np.float32)
        dense_b = inits[b_name].copy().astype(np.float32) if b_name and b_name in inits else np.zeros(dense_w.shape[0], dtype=np.float32)
        # Gemm stores W as (out, in) — that's what we want for our dense layer
        i += 1
        continue

    if op == "MatMul":
        # Dense layer in models exported as MatMul+Add
        # input[0] = activations, input[1] = W (in, out) — transpose needed -> (out, in)
        w_name = node.input[1]
        w_raw = inits[w_name].copy().astype(np.float32)
        # w_raw shape is (in_features, out_features) for MatMul — transpose to (out, in)
        dense_w = w_raw.T
        # Bias comes from the following Add node
        add_node = nodes[i+1] if (i+1) < len(nodes) else None
        dense_b = np.zeros(dense_w.shape[0], dtype=np.float32)
        i_inc = 0
        if add_node is not None and add_node.op_type == "Add":
            for inp in add_node.input:
                if inp in inits:
                    dense_b = inits[inp].flatten().astype(np.float32)
                    break
            i_inc = 1
        i += 1 + i_inc
        continue

    if op in ("Softmax", "Relu"):
        i += 1
        continue

    # unknown — skip
    i += 1

print(f"Parsed {len(layers)} conv layers + dense")

# Assign names
name_plan = ["CONV0", "DW1", "PW2", "DW3", "PW4", "DW5", "PW6",
             "DW7", "PW8", "DW9", "PW10", "DW11", "PW12"]
assert len(layers) == len(name_plan), f"Expected {len(name_plan)} conv layers, got {len(layers)}"

named_layers = []
for idx, (w, b, is_dw, strides, pads) in enumerate(layers):
    named_layers.append(ConvLayer(name_plan[idx], w, b, is_dw, strides, pads))

# ---------------------------------------------------------------------------
# Generate header
# ---------------------------------------------------------------------------

def arr_to_c(name, data, dtype="float"):
    """Emit a static const float array."""
    flat = data.flatten()
    lines = [f"static const {dtype} {name}[{flat.size}] = {{"]
    for row_start in range(0, flat.size, 8):
        chunk = flat[row_start:row_start+8]
        vals  = ", ".join(f"{v:.8e}f" for v in chunk)
        lines.append(f"    {vals},")
    lines.append("};")
    return "\n".join(lines)

with open(OUT_PATH, "w") as f:
    f.write("/* Auto-generated by tools/export_ds_cnn_weights.py — do not edit. */\n")
    f.write("#ifndef DS_CNN_WEIGHTS_H\n#define DS_CNN_WEIGHTS_H\n\n")

    for lyr in named_layers:
        n   = lyr.name
        w   = lyr.w
        b   = lyr.b
        sh  = w.shape   # (out_ch, in_ch/groups, kH, kW)
        out_ch, in_ch_g, kH, kW = sh

        f.write(f"/* Layer {n}: shape ({out_ch},{in_ch_g},{kH},{kW}) strides={lyr.strides} pads={lyr.pads} dw={lyr.is_dw} */\n")
        f.write(f"#define DS_CNN_{n}_OUT_CH  {out_ch}\n")
        f.write(f"#define DS_CNN_{n}_IN_CH_G {in_ch_g}\n")
        f.write(f"#define DS_CNN_{n}_KH      {kH}\n")
        f.write(f"#define DS_CNN_{n}_KW      {kW}\n")
        f.write(f"#define DS_CNN_{n}_SH      {lyr.strides[0]}\n")
        f.write(f"#define DS_CNN_{n}_SW      {lyr.strides[1]}\n")
        f.write(f"#define DS_CNN_{n}_PT      {lyr.pads[0]}\n")
        f.write(f"#define DS_CNN_{n}_PL      {lyr.pads[1]}\n")
        f.write(f"#define DS_CNN_{n}_PB      {lyr.pads[2]}\n")
        f.write(f"#define DS_CNN_{n}_PR      {lyr.pads[3]}\n")
        f.write(f"\n")
        f.write(arr_to_c(f"DS_CNN_{n}_W", w))
        f.write("\n\n")
        f.write(arr_to_c(f"DS_CNN_{n}_B", b))
        f.write("\n\n")

    # Dense
    if dense_w is not None:
        out_n, in_n = dense_w.shape
        f.write(f"/* Dense: ({out_n},{in_n}) */\n")
        f.write(f"#define DS_CNN_DENSE_OUT {out_n}\n")
        f.write(f"#define DS_CNN_DENSE_IN  {in_n}\n\n")
        f.write(arr_to_c("DS_CNN_DENSE_W", dense_w))
        f.write("\n\n")
        f.write(arr_to_c("DS_CNN_DENSE_B", dense_b))
        f.write("\n\n")

    f.write("/* Scratch buffer size: max activation tensor = 64*25*5 = 8000 floats */\n")
    f.write("#define DS_CNN_MAX_ACT_SIZE 8000\n\n")

    f.write("#endif /* DS_CNN_WEIGHTS_H */\n")

print(f"Wrote: {OUT_PATH}")
