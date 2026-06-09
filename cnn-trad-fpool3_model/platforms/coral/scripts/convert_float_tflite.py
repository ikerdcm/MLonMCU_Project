#!/usr/bin/env python3
"""
Build the FP32 (float32) TFLite model for the Coral `fp32-cpu` experiment.

Story A baseline / Story B bottom rung: run the *same* 6-block DS-CNN-L on the
Coral M7 CPU (plain TFLite-Micro, no Edge-TPU op) so we get the TPU-vs-CPU
latency/energy curve. Latency/energy are architecture-determined, so any faithful
6-block float model gives the right numbers; using the finetuned Keras model also
yields a meaningful FP32 *accuracy* (eval_quantized_model.py works on float .tflite).

Outputs:
    models/ds_cnn_l_float.tflite                         (float32 I/O, no quant)
    platforms/coral/apps/common/ds_cnn_test_input_left_float.h
        (the "left" MFCC vector dequantized to float, so kws_bench_cpu can mirror
         the int8 bench's fixed-input design)

The float test vector is produced by dequantizing the existing int8 header with
the int8 model's input quantization (scale/zero-point), so the two benches feed
the same physical MFCC. The script then runs the float model on it and reports
the prediction as a sanity check (expected: index 2 = "left").

Usage:
    cd platforms/coral/scripts
    python3 convert_float_tflite.py
    # then build/flash on the board host:
    #   ./build_and_flash_bench_cpu.sh   (offline)  /  ./build_and_flash_live_cpu.sh (online)
"""

import os
import re
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.normpath(os.path.join(SCRIPT_DIR, "../../../models"))
TRAIN_DIR  = os.path.normpath(os.path.join(SCRIPT_DIR, "../../../training"))
COMMON_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "../apps/common"))

# Faithful 6-block source (real trained weights). Falls back to the
# architecture-only "restored" model (placeholder upper blocks) if absent.
KERAS_PRIMARY  = os.path.join(TRAIN_DIR, "trained_models/ds_cnn_l_finetuned.keras")
KERAS_FALLBACK = os.path.join(TRAIN_DIR, "trained_models/ds_cnn_l_restored.keras")

# int8 model whose input quant maps the int8 test vector back to MFCC floats.
# This is the model behind the measured int8-accel (v1) result.
INT8_TFLITE = os.path.join(MODEL_DIR, "ds_cnn_l_static.tflite")
INT8_HEADER = os.path.join(COMMON_DIR, "ds_cnn_test_input_left_int8.h")

OUT_TFLITE = os.path.join(MODEL_DIR, "ds_cnn_l_float.tflite")
OUT_HEADER = os.path.join(COMMON_DIR, "ds_cnn_test_input_left_float.h")

INPUT_SHAPE = [1, 49, 10, 1]


def convert_keras_to_float_tflite() -> bytes:
    keras_path = KERAS_PRIMARY if os.path.exists(KERAS_PRIMARY) else KERAS_FALLBACK
    faithful = keras_path == KERAS_PRIMARY
    print(f"Loading Keras model: {keras_path}")
    if not faithful:
        print("  WARNING: finetuned model not found — using 'restored' (placeholder upper")
        print("           blocks). Latency/energy are valid; FP32 accuracy is NOT.")
    model = tf.keras.models.load_model(keras_path, compile=False)

    # Pin batch=1 so the exported graph is fully static — otherwise the None batch
    # dim makes TFLite emit SHAPE/STRIDED_SLICE/PACK to compute the final reshape
    # at runtime (ops we don't want in TFLite-Micro). Re-wrapping the loaded model
    # on a fixed Input keeps from_keras_model's BatchNorm folding intact (a bare
    # concrete function instead unfolds BN into ADD/MUL/RSQRT and freezes weights
    # as resource variables — wrong and unsupported on micro).
    fixed_in = tf.keras.Input(batch_shape=tuple(INPUT_SHAPE), name="input")
    fixed_model = tf.keras.Model(fixed_in, model(fixed_in))
    converter = tf.lite.TFLiteConverter.from_keras_model(fixed_model)
    # Default optimizations OFF → keep weights & activations float32 (no quant).
    tflite = converter.convert()
    with open(OUT_TFLITE, "wb") as f:
        f.write(tflite)
    print(f"Wrote {OUT_TFLITE}  ({len(tflite)/1024:.1f} KiB)")
    return tflite


def inspect_model(tflite_bytes: bytes):
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    di = interp.get_input_details()[0]
    do = interp.get_output_details()[0]
    print("Float model I/O:")
    print(f"  input : {di['dtype'].__name__} shape={list(di['shape'])}")
    print(f"  output: {do['dtype'].__name__} shape={list(do['shape'])}")
    try:
        ops = sorted({o['op_name'] for o in interp._get_ops_details()})
        print(f"  builtin ops ({len(ops)}): {', '.join(ops)}")
        print("  -> MicroMutableOpResolver must register exactly these.")
    except Exception as e:
        print(f"  (could not enumerate ops: {e})")
    return interp, di, do


def int8_input_quant() -> tuple[float, int]:
    it = tf.lite.Interpreter(model_path=INT8_TFLITE)
    it.allocate_tensors()
    scale, zp = it.get_input_details()[0]['quantization']
    print(f"int8 input quant (from {os.path.basename(INT8_TFLITE)}): scale={scale}, zp={zp}")
    return float(scale), int(zp)


def parse_int8_header() -> tuple[np.ndarray, int, int]:
    txt = open(INT8_HEADER).read()
    size = int(re.search(r"KWS_TEST_INPUT_SIZE\s+(\d+)", txt).group(1))
    expected = int(re.search(r"KWS_TEST_EXPECTED_IDX\s+(\d+)", txt).group(1))
    body = re.search(r"kws_test_input_int8\[\d+\]\s*=\s*\{(.*?)\}", txt, re.S).group(1)
    vals = np.array([int(v) for v in re.findall(r"-?\d+", body)], dtype=np.int32)
    assert vals.size == size, f"parsed {vals.size} values, expected {size}"
    return vals, size, expected


def write_float_header(floats: np.ndarray, expected_idx: int, scale: float, zp: int):
    n = floats.size
    lines = [
        "#ifndef DS_CNN_TEST_INPUT_LEFT_FLOAT_H",
        "#define DS_CNN_TEST_INPUT_LEFT_FLOAT_H",
        "",
        '/* FP32 MFCC of "left_b528edb3_nohash_0.wav" for ds_cnn_l_float.tflite.',
        f" * Dequantized from the int8 vector: f = (q - ({zp})) * {scale!r}",
        " * Generated by platforms/coral/scripts/convert_float_tflite.py — do not edit by hand.",
        f" * Expected prediction: index {expected_idx} (left) */",
        f"#define KWS_TEST_INPUT_FLOAT_COUNT {n}",
        f"#define KWS_TEST_INPUT_FLOAT_SIZE  ({n} * (int)sizeof(float))",
        f"#define KWS_TEST_EXPECTED_IDX {expected_idx}",
        "",
        f"static const float kws_test_input_float[{n}] = {{",
    ]
    for i in range(0, n, 10):
        chunk = ", ".join(f"{v:.8f}f" for v in floats[i:i + 10])
        lines.append(f"    {chunk},")
    lines += ["};", "", "#endif  // DS_CNN_TEST_INPUT_LEFT_FLOAT_H", ""]
    with open(OUT_HEADER, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {OUT_HEADER}  ({n} floats)")


def main() -> int:
    tflite_bytes = convert_keras_to_float_tflite()
    interp, di, _ = inspect_model(tflite_bytes)

    scale, zp = int8_input_quant()
    q, size, expected = parse_int8_header()
    floats = ((q - zp).astype(np.float32) * scale)
    write_float_header(floats, expected, scale, zp)

    # Sanity: run the float model on the dequantized vector.
    x = floats.reshape(INPUT_SHAPE).astype(di['dtype'])
    interp.set_tensor(di['index'], x)
    interp.invoke()
    out = interp.get_tensor(interp.get_output_details()[0]['index']).ravel()
    pred = int(np.argmax(out))
    labels = ["down", "go", "left", "no", "off", "on",
              "right", "stop", "up", "yes", "silence", "unknown"]
    ok = "OK" if pred == expected else "MISMATCH"
    print(f"Float-model prediction on 'left' vector: idx={pred} ({labels[pred]})  "
          f"[expected {expected} ({labels[expected]})]  -> {ok}")
    if pred != expected:
        print("  NOTE: latency/energy are still valid (value-independent). A mismatch")
        print("        only means the finetuned model's input domain differs from the")
        print("        original int8 calibration; regenerate the vector from a fresh MFCC")
        print("        if a correct live prediction matters.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
