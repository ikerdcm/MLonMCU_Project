#!/usr/bin/env python3
"""
Patch all shape_signature vectors in a TFLite flatbuffer to replace dynamic
batch dimension (-1) with 1, making every tensor statically shaped.

The Edge TPU compiler requires fully static shapes. TFLite models converted
from Keras store shape_signature with -1 in the batch position for every
tensor.  This script finds every int32 vector whose first element is -1 and
whose remaining elements are all valid positive dimensions, and replaces -1
with 1.  It is safe for models with batch_size=1.

Usage:
    python3 fix_static_shape.py [input.tflite [output.tflite]]

Defaults:
    input : ../../../models/ds_cnn_l.tflite
    output: ../../../models/ds_cnn_l_static.tflite
"""

import sys
import struct
import os


def patch(in_path: str, out_path: str, batch: int = 1) -> bool:
    with open(in_path, "rb") as f:
        data = bytearray(f.read())

    patched = 0
    i = 0
    # Walk the entire file looking for int32 vectors (count in 1..6)
    # whose first element is -1 and whose remaining elements are all > 0.
    # These are shape_signature vectors: [-1, dim1, dim2, ...] → [1, dim1, ...]
    while i < len(data) - 4:
        count_raw = struct.unpack_from("<I", data, i)[0]
        if 1 <= count_raw <= 6:
            vec_size = 4 + count_raw * 4
            if i + vec_size <= len(data):
                vals = list(struct.unpack_from(f"<{count_raw}i", data, i + 4))
                if vals[0] == -1 and all(v > 0 for v in vals[1:]):
                    # Patch: replace -1 with batch (1)
                    struct.pack_into("<i", data, i + 4, batch)
                    patched += 1
                    i += vec_size
                    continue
        i += 1

    if patched == 0:
        print("No dynamic shape_signature vectors found — model may already be static.")
        if in_path != out_path:
            with open(out_path, "wb") as f:
                f.write(data)
        return True

    with open(out_path, "wb") as f:
        f.write(data)
    print(f"Patched {patched} shape_signature vector(s): batch -1 → {batch}")
    print(f"Output: {out_path}")
    return True


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir  = os.path.normpath(os.path.join(script_dir, "../../../models"))

    in_path  = sys.argv[1] if len(sys.argv) > 1 else os.path.join(model_dir, "ds_cnn_l.tflite")
    out_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(model_dir, "ds_cnn_l_static.tflite")

    print(f"Input : {in_path}")
    print(f"Output: {out_path}")
    ok = patch(in_path, out_path)
    sys.exit(0 if ok else 1)
