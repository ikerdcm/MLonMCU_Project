#!/usr/bin/env python3
"""
Generate PTQ INT8 weights for the kws20_int8_cpu software backend.

This backend keeps activations in int8 and rescales each convolution with an
integer right shift. The important invariant is:

    w_i8 = round(w_float * 2^RSHIFT)
    out_i8 = clip((acc + 2^(RSHIFT-1)) >> RSHIFT, 0, 127)

Using abs_max/127 scales without matching right shifts makes the integer
pipeline numerically inconsistent and kills the activations in later layers.
"""

import argparse
import math
import re
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE = SCRIPT_DIR.parent / "kws20_32bit" / "kws20_32bit_weights.h"
DEFAULT_OUT = SCRIPT_DIR / "kws20_int8_cpu_weights.h"

LAYERS = [
    ("voice_conv1_weight", "voice_conv1", "VOICE_CONV1", (100, 128, 1)),
    ("voice_conv2_weight", "voice_conv2", "VOICE_CONV2", (96, 100, 3)),
    ("voice_conv3_weight", "voice_conv3", "VOICE_CONV3", (64, 96, 3)),
    ("voice_conv4_weight", "voice_conv4", "VOICE_CONV4", (48, 64, 3)),
    ("kws_conv1_weight", "kws_conv1", "KWS_CONV1", (64, 48, 3)),
    ("kws_conv2_weight", "kws_conv2", "KWS_CONV2", (96, 64, 3)),
    ("kws_conv3_weight", "kws_conv3", "KWS_CONV3", (100, 96, 3)),
    ("kws_conv4_weight", "kws_conv4", "KWS_CONV4", (64, 100, 6)),
    ("fc_weight", "fc", "FC", (21, 256)),
]


def array_size(shape):
    n = 1
    for value in shape:
        n *= value
    return n


def load_float_array(text, name, size):
    pattern = rf"static const float {re.escape(name)}\[{size}\]\s*=\s*\{{([^}}]+)\}}"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        raise RuntimeError(f"Could not find float array {name}[{size}]")

    values = []
    for raw in match.group(1).split(","):
        raw = raw.strip()
        if raw:
            values.append(float(raw.replace("f", "")))

    if len(values) != size:
        raise RuntimeError(f"{name}: expected {size} values, found {len(values)}")
    return np.array(values, dtype=np.float32)


def quantize_power_of_two(weights):
    abs_max = float(np.abs(weights).max())
    if abs_max == 0.0:
        rshift = 0
    else:
        rshift = int(math.floor(math.log2(127.0 / abs_max)))
        rshift = max(rshift, 0)

    factor = 1 << rshift
    weights_i8 = np.clip(np.round(weights * factor), -128, 127).astype(np.int8)
    return weights_i8, rshift, abs_max


def format_int8_array(name, values):
    flat = values.flatten()
    lines = [f"static const int8_t {name}[{len(flat)}] = {{"]
    for start in range(0, len(flat), 16):
        row = ", ".join(str(int(v)) for v in flat[start:start + 16])
        lines.append(f"    {row},")
    lines.append("};")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default=str(DEFAULT_SOURCE),
        help="float32 weights header to quantize",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help="output INT8 weights header",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="deprecated; accepted for old commands but not used",
    )
    args = parser.parse_args()

    if args.checkpoint:
        print("NOTE: --checkpoint is ignored. This backend quantizes the float32 header with PTQ.")

    source = Path(args.source).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    text = source.read_text()

    out_lines = [
        "/*",
        " * Auto-generated PTQ INT8 weights for kws20_int8_cpu.",
        f" * Source: {source.name}",
        " * Per-layer scaling: w_i8 = round(w_float * 2^RSHIFT), fits in int8.",
        " * Output quantization: out_i8 = clip((acc + 2^(RS-1)) >> RS, 0, 127)",
        " */",
        "",
        "#ifndef KWS20_INT8_CPU_WEIGHTS_H",
        "#define KWS20_INT8_CPU_WEIGHTS_H",
        "#include <stdint.h>",
        "",
    ]

    for float_name, var_name, macro_prefix, shape in LAYERS:
        size = array_size(shape)
        weights = load_float_array(text, float_name, size).reshape(shape)
        weights_i8, rshift, abs_max = quantize_power_of_two(weights)

        print(
            f"{float_name}: shape={list(shape)}, abs_max={abs_max:.4f}, "
            f"factor=2^{rshift}, int8_abs_max={int(np.abs(weights_i8).max())}"
        )

        out_lines.append(
            f"/* {float_name}: shape={list(shape)}, factor=2^{rshift}, "
            f"abs_max_f32={abs_max:.4f} */"
        )
        out_lines.append(f"#define {macro_prefix}_RSHIFT {rshift}")
        out_lines.append(format_int8_array(f"{var_name}_weight", weights_i8))
        out_lines.append("")

    out_lines.append("#endif /* KWS20_INT8_CPU_WEIGHTS_H */")
    out_lines.append("")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(out_lines))
    print(f"Written: {out}")


if __name__ == "__main__":
    main()
