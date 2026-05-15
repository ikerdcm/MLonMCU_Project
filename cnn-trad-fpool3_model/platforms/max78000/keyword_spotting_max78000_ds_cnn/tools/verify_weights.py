#!/usr/bin/env python3
"""
Host-side verification: run DS-CNN-L forward pass in numpy using the
folded weights from ds_cnn_weights.h, and compare against ONNX runtime.

Usage:
    /home/pascal/max78000/ai8x-synthesis/.venv/bin/python tools/verify_weights.py
"""

import sys
import os
import re
import numpy as np

# ---------------------------------------------------------------------------
# Parse ds_cnn_weights.h
# ---------------------------------------------------------------------------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
WEIGHTS_H   = os.path.join(PROJECT_DIR, "ds_cnn_weights.h")
ONNX_PATH   = os.path.normpath(os.path.join(PROJECT_DIR, "..", "..", "..", "models", "ds_cnn_l.onnx"))
TEST_INPUT_H = os.path.join(PROJECT_DIR, "ds_cnn_test_input_left.h")

print(f"Parsing {WEIGHTS_H}")

header = open(WEIGHTS_H).read()

def parse_array(name, header):
    """Parse 'static const float NAME[N] = { ... };' from header text."""
    # Find the array declaration
    pat = rf'static const float {re.escape(name)}\[(\d+)\] = \{{([^}}]*)\}};'
    m = re.search(pat, header, re.DOTALL)
    if m is None:
        raise ValueError(f"Array {name} not found in header")
    size = int(m.group(1))
    vals_str = m.group(2)
    vals = [float(v.strip().rstrip('f')) for v in vals_str.split(',') if v.strip()]
    assert len(vals) == size, f"{name}: expected {size} values, got {len(vals)}"
    return np.array(vals, dtype=np.float32)

def parse_define(name, header):
    pat = rf'#define\s+{re.escape(name)}\s+(\d+)'
    m = re.search(pat, header)
    if m is None:
        raise ValueError(f"#define {name} not found")
    return int(m.group(1))

# Layer config
layer_names = ["CONV0", "DW1", "PW2", "DW3", "PW4", "DW5", "PW6",
               "DW7", "PW8", "DW9", "PW10", "DW11", "PW12"]

layers = []
for n in layer_names:
    out_ch  = parse_define(f"DS_CNN_{n}_OUT_CH",  header)
    in_ch_g = parse_define(f"DS_CNN_{n}_IN_CH_G", header)
    kh      = parse_define(f"DS_CNN_{n}_KH",       header)
    kw      = parse_define(f"DS_CNN_{n}_KW",       header)
    sh      = parse_define(f"DS_CNN_{n}_SH",       header)
    sw      = parse_define(f"DS_CNN_{n}_SW",       header)
    pt      = parse_define(f"DS_CNN_{n}_PT",       header)
    pb      = parse_define(f"DS_CNN_{n}_PB",       header)
    pl      = parse_define(f"DS_CNN_{n}_PL",       header)
    pr      = parse_define(f"DS_CNN_{n}_PR",       header)
    w = parse_array(f"DS_CNN_{n}_W", header).reshape(out_ch, in_ch_g, kh, kw)
    b = parse_array(f"DS_CNN_{n}_B", header)
    layers.append(dict(name=n, w=w, b=b, out_ch=out_ch, in_ch_g=in_ch_g,
                       kh=kh, kw=kw, sh=sh, sw=sw,
                       pt=pt, pb=pb, pl=pl, pr=pr))

dense_out = parse_define("DS_CNN_DENSE_OUT", header)
dense_in  = parse_define("DS_CNN_DENSE_IN",  header)
dense_w   = parse_array("DS_CNN_DENSE_W", header).reshape(dense_out, dense_in)
dense_b   = parse_array("DS_CNN_DENSE_B", header)

print(f"Loaded {len(layers)} conv layers + dense ({dense_out}x{dense_in})")

# ---------------------------------------------------------------------------
# Parse test input
# ---------------------------------------------------------------------------
test_h = open(TEST_INPUT_H).read()
pat = r'static const float ds_cnn_test_input_left\[490\] = \{([^}]*)\};'
m = re.search(pat, test_h, re.DOTALL)
if m:
    vals = [float(v.strip().rstrip('f')) for v in m.group(1).split(',') if v.strip()]
    test_input = np.array(vals, dtype=np.float32)
else:
    test_input = np.zeros(490, dtype=np.float32)
print(f"Test input shape: {test_input.shape}")

# ---------------------------------------------------------------------------
# NumPy forward pass
# ---------------------------------------------------------------------------

def relu(x):
    return np.maximum(x, 0.0)

def conv2d_relu(x, w, b, sh, sw, pt, pb, pl, pr):
    """x: (in_c, H, W), w: (out_c, in_c, kH, kW)"""
    in_c, in_h, in_w = x.shape
    out_c, _, kh, kw = w.shape
    # Pad
    x_pad = np.pad(x, ((0,0),(pt,pb),(pl,pr)), mode='constant')
    out_h = (in_h + pt + pb - kh) // sh + 1
    out_w = (in_w + pl + pr - kw) // sw + 1
    out = np.zeros((out_c, out_h, out_w), dtype=np.float32)
    for oc in range(out_c):
        for oh in range(out_h):
            for ow in range(out_w):
                patch = x_pad[:, oh*sh:oh*sh+kh, ow*sw:ow*sw+kw]
                out[oc, oh, ow] = np.sum(patch * w[oc]) + b[oc]
    return relu(out)

def dwconv2d_relu(x, w, b, sh, sw, pt, pb, pl, pr):
    """x: (ch, H, W), w: (ch, 1, kH, kW)"""
    ch, in_h, in_w = x.shape
    kh, kw = w.shape[2], w.shape[3]
    x_pad = np.pad(x, ((0,0),(pt,pb),(pl,pr)), mode='constant')
    out_h = (in_h + pt + pb - kh) // sh + 1
    out_w = (in_w + pl + pr - kw) // sw + 1
    out = np.zeros((ch, out_h, out_w), dtype=np.float32)
    for c in range(ch):
        for oh in range(out_h):
            for ow in range(out_w):
                patch = x_pad[c, oh*sh:oh*sh+kh, ow*sw:ow*sw+kw]
                out[c, oh, ow] = np.sum(patch * w[c, 0]) + b[c]
    return relu(out)

def avgpool(x, kh, kw, sh, sw):
    ch, in_h, in_w = x.shape
    out_h = (in_h - kh) // sh + 1
    out_w = (in_w - kw) // sw + 1
    out = np.zeros((ch, out_h, out_w), dtype=np.float32)
    for c in range(ch):
        for oh in range(out_h):
            for ow in range(out_w):
                out[c, oh, ow] = np.mean(x[c, oh*sh:oh*sh+kh, ow*sw:ow*sw+kw])
    return out

def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)

def forward(inp490):
    x = inp490.reshape(1, 49, 10)  # (1, H, W)

    for i, lyr in enumerate(layers):
        n  = lyr['name']
        w  = lyr['w']
        b  = lyr['b']
        sh = lyr['sh']; sw = lyr['sw']
        pt = lyr['pt']; pb = lyr['pb']
        pl = lyr['pl']; pr = lyr['pr']
        is_dw = (lyr['in_ch_g'] == 1 and lyr['out_ch'] == x.shape[0])
        if is_dw:
            x = dwconv2d_relu(x, w, b, sh, sw, pt, pb, pl, pr)
        else:
            x = conv2d_relu(x, w, b, sh, sw, pt, pb, pl, pr)

    # AvgPool (24,5) stride (24,5)
    x = avgpool(x, 24, 5, 24, 5)
    # Flatten
    x = x.flatten()
    # Dense
    logits = dense_w @ x + dense_b
    scores = softmax(logits)
    return scores

labels = ["down","go","left","no","off","on",
          "right","stop","up","yes","silence","unknown"]

print("Running numpy forward pass on test input...")
scores = forward(test_input)
pred = int(np.argmax(scores))
print(f"Prediction: {pred} ({labels[pred]})  score={scores[pred]:.6f}")
print(f"Expected:   2 (left)")
print(f"Result: {'PASS' if pred == 2 else 'FAIL'}")
print()

# Top 5
top5 = np.argsort(scores)[::-1][:5]
print("Top-5 scores:")
for idx in top5:
    print(f"  {idx:2d} {labels[idx]:<10s} {scores[idx]:.6f}")

# ---------------------------------------------------------------------------
# Compare against ONNX runtime
# ---------------------------------------------------------------------------
try:
    import onnxruntime as ort
    print("\nComparing against ONNX runtime...")
    sess = ort.InferenceSession(ONNX_PATH)
    inp_name = sess.get_inputs()[0].name
    # ONNX model expects NHWC (N,49,10,1)
    onnx_inp = test_input.reshape(1, 49, 10, 1)
    onnx_out = sess.run(None, {inp_name: onnx_inp})[0]
    onnx_pred = int(np.argmax(onnx_out))
    print(f"ONNX prediction: {onnx_pred} ({labels[onnx_pred]})")
    print(f"Numpy prediction: {pred} ({labels[pred]})")
    max_diff = np.max(np.abs(onnx_out[0] - scores))
    print(f"Max score diff vs ONNX: {max_diff:.6f}")
    if pred == onnx_pred:
        print("Predictions match: PASS")
    else:
        print("WARNING: predictions differ!")
except ImportError:
    print("\nonnxruntime not available — skipping ONNX comparison")
