#!/usr/bin/env python3
"""
§7.1/§7.2 — Convert DS-CNN-L to a fully static INT8 TFLite model (v2).

v1 bug: convert_static_int8.py loaded training/trained_models/kws_ref_model
(the 4-block MLCommons reference, ~2.6M MACs) instead of the 6-block DS-CNN-L
(~3.94M MACs) that is actually deployed as ds_cnn_l_static_edgetpu.tflite.
It also defaulted to random calibration.

v2 fix: use models/ds_cnn_l.tflite as the authoritative 6-block source.
That file was produced by the original quantize.py run with real MFCC
calibration data (quant_cal_idxs.txt).  Its only flaw is a dynamic batch
dimension in every shape_signature vector, which the Edge TPU compiler
rejects.  fix_static_shape.patch() byte-swaps those -1 → 1 without touching
weights or scale/zero-point parameters, so calibration quality is preserved.

If the 6-block SavedModel is ever re-saved (finetune.py → --saved_model_path),
use --saved_model with --data_dir to re-quantize from scratch; that path
produces a freshly calibrated model rather than patching the existing one.

Outputs:
    models/ds_cnn_l_static_v2.tflite      (patch mode, default)
    models/ds_cnn_l_static_v2.tflite      (re-quantize mode, when flags given)

Next steps:
    ./compile_edgetpu_v2.sh
    cd ../../training
    python3 eval_quantized_model.py \\
        --tfl_file_name ../models/ds_cnn_l_static_v2.tflite \\
        --target_set test --data_dir ~/data/speech_commands_v002
"""

import os
import sys
import argparse
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.normpath(os.path.join(SCRIPT_DIR, "../../../models"))
TRAIN_DIR   = os.path.normpath(os.path.join(SCRIPT_DIR, "../../../training"))

SRC_TFLITE  = os.path.join(MODEL_DIR, "ds_cnn_l.tflite")
OUT_STATIC  = os.path.join(MODEL_DIR, "ds_cnn_l_static_v2.tflite")

INPUT_SHAPE    = [1, 49, 10, 1]
NUM_CAL_STEPS  = 200


# ---------------------------------------------------------------------------
# Calibration dataset helpers
# ---------------------------------------------------------------------------

def random_representative_dataset():
    rng = np.random.default_rng(42)
    for _ in range(NUM_CAL_STEPS):
        yield [rng.standard_normal(INPUT_SHAPE).astype(np.float32)]


# ---------------------------------------------------------------------------
# Patch mode (default): byte-patch the existing well-calibrated 6-block tflite
# ---------------------------------------------------------------------------

def patch_mode():
    sys.path.insert(0, SCRIPT_DIR)
    import fix_static_shape

    if not os.path.isfile(SRC_TFLITE):
        print(f"ERROR: source model not found: {SRC_TFLITE}")
        sys.exit(1)

    src_size = os.path.getsize(SRC_TFLITE)
    print(f"Source : {SRC_TFLITE} ({src_size} bytes, 6-block DS-CNN-L)")
    fix_static_shape.patch(SRC_TFLITE, OUT_STATIC)

    _verify(OUT_STATIC)
    print(f"\nNext step: ./compile_edgetpu_v2.sh")


# ---------------------------------------------------------------------------
# Re-quantize mode: rebuild from SavedModel with real MFCC calibration
# ---------------------------------------------------------------------------

def mfcc_representative_dataset(data_dir: str):
    sys.path.insert(0, TRAIN_DIR)
    import get_dataset as kws_data
    import kws_util

    flags = kws_util.parse_command()[0]
    flags.data_dir = data_dir
    flags.batch_size = 1

    _, _, ds_val = kws_data.get_training_data(flags, val_cal_subset=True)
    ds_val = ds_val.unbatch().batch(1)

    with open(os.path.join(TRAIN_DIR, "quant_cal_idxs.txt")) as f:
        cal_idxs = sorted(int(l) for l in f)

    it = iter(ds_val)
    for _ in cal_idxs:
        sample = next(it)[0].numpy()
        yield [sample]


def requantize_mode(saved_model: str, data_dir: str | None):
    print(f"Loading SavedModel: {saved_model}")
    model = tf.keras.models.load_model(saved_model)

    input_spec = tf.TensorSpec(shape=INPUT_SHAPE, dtype=tf.float32)
    serving_fn = tf.function(
        lambda x: model(x, training=False),
        input_signature=[input_spec],
    )
    concrete_func = serving_fn.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.int8
    converter.inference_output_type = tf.int8

    if data_dir:
        print(f"Using MFCC calibration data from: {data_dir}")
        converter.representative_dataset = lambda: mfcc_representative_dataset(data_dir)
    else:
        print("WARNING: no --data_dir given — falling back to random calibration.")
        print("         Re-run with --data_dir /path/to/speech_commands for best accuracy.")
        converter.representative_dataset = random_representative_dataset

    print("Converting...")
    tflite_model = converter.convert()

    with open(OUT_STATIC, "wb") as f:
        f.write(tflite_model)
    print(f"Written {len(tflite_model)} bytes to {OUT_STATIC}")

    _verify(OUT_STATIC)
    print(f"\nNext step: ./compile_edgetpu_v2.sh")


# ---------------------------------------------------------------------------

def _verify(path: str):
    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    sig = inp.get('shape_signature')
    tensors = interp.get_tensor_details()

    if sig is None or -1 not in sig:
        print("OK: shape_signature is fully static")
    else:
        print(f"WARNING: shape_signature still contains -1: {sig}")

    print(f"Input  shape={inp['shape']} dtype={inp['dtype'].__name__}")
    print(f"Output shape={out['shape']} dtype={out['dtype'].__name__}")
    print(f"Tensors: {len(tensors)}  (expect ~52 for 6-block DS-CNN-L)")
    print(f"Output : {path}  ({os.path.getsize(path)} bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--saved_model", default=None,
                        help="Path to 6-block DS-CNN-L SavedModel for re-quantization "
                             "(requires --data_dir). Omit to use patch mode (default).")
    parser.add_argument("--data_dir", default=None,
                        help="Path to speech_commands dataset (required with --saved_model).")
    args = parser.parse_args()

    if args.saved_model:
        requantize_mode(args.saved_model, args.data_dir)
    else:
        if args.data_dir:
            print("NOTE: --data_dir is ignored without --saved_model (patch mode uses "
                  "the existing well-calibrated ds_cnn_l.tflite)")
        patch_mode()
