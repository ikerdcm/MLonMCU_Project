#!/usr/bin/env python3
"""
Convert the DS-CNN-L Keras SavedModel to a fully static INT8 TFLite model
suitable for Edge TPU compilation.

The critical difference from the original quantize.py: this script uses a
concrete function with a fixed [1, 49, 10, 1] input shape so every tensor
in the graph gets a static shape_signature.  The original model had batch=-1
which the Edge TPU compiler rejects.

Usage:
    python3 convert_static_int8.py [--data_dir /path/to/speech_commands]

With real calibration data (recommended for best accuracy):
    python3 convert_static_int8.py --data_dir ~/data/speech_commands_v002

Without data (uses random calibration - structure correct, accuracy may degrade):
    python3 convert_static_int8.py
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

SAVED_MODEL = os.path.join(TRAIN_DIR, "trained_models/kws_ref_model")
OUT_STATIC  = os.path.join(MODEL_DIR, "ds_cnn_l_static.tflite")

INPUT_SHAPE = [1, 49, 10, 1]
NUM_CAL_STEPS = 200


def random_representative_dataset():
    rng = np.random.default_rng(42)
    for _ in range(NUM_CAL_STEPS):
        yield [rng.standard_normal(INPUT_SHAPE).astype(np.float32)]


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


def convert(data_dir=None):
    print(f"Loading SavedModel: {SAVED_MODEL}")
    model = tf.keras.models.load_model(SAVED_MODEL)

    @tf.autograph.experimental.do_not_convert
    @tf.function(input_signature=[tf.TensorSpec(shape=INPUT_SHAPE, dtype=tf.float32)])
    def predict(x):
        return model(x, training=False)

    concrete_func = predict.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func], model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.int8
    converter.inference_output_type = tf.int8

    if data_dir:
        print(f"Using MFCC calibration data from: {data_dir}")
        converter.representative_dataset = lambda: mfcc_representative_dataset(data_dir)
    else:
        print("WARNING: using random calibration data — accuracy may be reduced.")
        print("         Re-run with --data_dir /path/to/speech_commands for best accuracy.")
        converter.representative_dataset = random_representative_dataset

    print("Converting...")
    tflite_model = converter.convert()

    with open(OUT_STATIC, "wb") as f:
        f.write(tflite_model)
    print(f"Written {len(tflite_model)} bytes to {OUT_STATIC}")

    # Verify shapes are fully static
    interp = tf.lite.Interpreter(model_content=tflite_model)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    sig = inp.get('shape_signature')
    if sig is None or -1 not in sig:
        print("OK: shape_signature is fully static")
    else:
        print(f"WARNING: shape_signature still contains -1: {sig}")
    print(f"Input  shape={inp['shape']}")
    print(f"Output shape={out['shape']}")
    print(f"\nNext step: run ./compile_edgetpu.sh {OUT_STATIC}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None,
                        help="Path to speech_commands dataset for MFCC calibration")
    args = parser.parse_args()
    convert(args.data_dir)
