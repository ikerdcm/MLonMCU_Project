#!/usr/bin/env python3
"""
Reconstruct the 6-block DS-CNN-L float Keras model so it can be re-quantized
with a static input shape (required by the Edge TPU compiler).

Background
----------
The only 6-block trained artifact in the repo is models/ds_cnn_l.tflite
(INT8, dynamic batch dim). The Edge TPU compiler requires a model traced with
a *fixed* input shape so that every intermediate tensor shape is statically
known at compile time.  Byte-patching shape_signature metadata is not
sufficient — the compiler runs its own shape-inference pass on the graph.

This script reconstructs a float Keras model (.keras) via weight transfer:
  - Layers 0-17 (initial Conv + 4 DS blocks): directly from kws_ref_model
    checkpoint (name-based, exact match)
  - Layers 18-25 (blocks 5-6): weights duplicated from block 4
    as a structural placeholder — not fine-tuned, accuracy not final
  - Layer 26 (Dense): left with random weights (feature distribution differs
    between 4-block and 6-block outputs, so 4-block Dense is wrong here)

After running this script, call convert_static_int8_v2.py to produce the
static INT8 tflite, then compile_edgetpu_v2.sh to produce the edgetpu model.

For production accuracy run finetune.py with the real dataset, then:
    python3 convert_static_int8_v2.py \\
        --saved_model training/trained_models/ds_cnn_l_finetuned.keras \\
        --data_dir ~/data/speech_commands_v002

Usage:
    cd platforms/coral/scripts
    python3 restore_ds_cnn_l_float.py
    python3 convert_static_int8_v2.py          # → models/ds_cnn_l_static_v2.tflite
    ./compile_edgetpu_v2.sh                    # → coral/model/*_edgetpu.tflite
"""

import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR       = os.path.normpath(os.path.join(SCRIPT_DIR, "../../../training"))
MODEL_DIR       = os.path.normpath(os.path.join(SCRIPT_DIR, "../../../models"))

REF_MODEL_PATH  = os.path.join(TRAIN_DIR, "trained_models/kws_ref_model")
OUT_KERAS_MODEL = os.path.join(TRAIN_DIR, "trained_models/ds_cnn_l_restored.keras")

sys.path.insert(0, TRAIN_DIR)
import kws_util
import keras_model as models


# kws_ref_model checkpoint: what each layer_with_weights-N contains
# Layer 0:  Conv2D         → kernel [10,4,1,64], bias [64]
# Layer 1:  BN             → gamma, beta, moving_mean, moving_variance [64]
# Layers 2k,2k+1 (k=1..4): DW + BN
# Layers 2k+2,2k+3: Conv1x1 + BN
# Layer 18: Dense          → kernel [64,12], bias [12]
#
# DS-CNN-L layers_with_weights[] mapping:
# [0..17] → checkpoint [0..17] (exact)
# [18..25] → duplicated from [14..17] (blocks 5 and 6 ← block 4)
# [26]    → Dense (left random; 4-block Dense doesn't match 6-block features)

def _layer_weights_from_ckpt(reader, ckpt_layer_idx, keras_layer):
    """Read the appropriate checkpoint tensors for one Keras layer."""
    p = f'layer_with_weights-{ckpt_layer_idx}/'
    ltype = type(keras_layer).__name__

    if ltype == 'DepthwiseConv2D':
        return [
            reader.get_tensor(p + 'depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE'),
            reader.get_tensor(p + 'bias/.ATTRIBUTES/VARIABLE_VALUE'),
        ]
    elif ltype == 'Conv2D':
        return [
            reader.get_tensor(p + 'kernel/.ATTRIBUTES/VARIABLE_VALUE'),
            reader.get_tensor(p + 'bias/.ATTRIBUTES/VARIABLE_VALUE'),
        ]
    elif ltype == 'BatchNormalization':
        return [
            reader.get_tensor(p + 'gamma/.ATTRIBUTES/VARIABLE_VALUE'),
            reader.get_tensor(p + 'beta/.ATTRIBUTES/VARIABLE_VALUE'),
            reader.get_tensor(p + 'moving_mean/.ATTRIBUTES/VARIABLE_VALUE'),
            reader.get_tensor(p + 'moving_variance/.ATTRIBUTES/VARIABLE_VALUE'),
        ]
    elif ltype == 'Dense':
        return [
            reader.get_tensor(p + 'kernel/.ATTRIBUTES/VARIABLE_VALUE'),
            reader.get_tensor(p + 'bias/.ATTRIBUTES/VARIABLE_VALUE'),
        ]
    else:
        return None


def transfer_weights(ref_path: str, model: tf.keras.Model) -> int:
    ckpt_prefix = os.path.join(ref_path, 'variables', 'variables')
    reader = tf.train.load_checkpoint(ckpt_prefix)

    layers_ww = [l for l in model.layers if l.weights]
    assert len(layers_ww) == 27, f"Expected 27 layers-with-weights, got {len(layers_ww)}"

    transferred = 0

    # ── Transfer checkpoint layers 0-17 (initial block + 4 DS blocks) ──────
    for ckpt_idx in range(18):
        layer = layers_ww[ckpt_idx]
        weights = _layer_weights_from_ckpt(reader, ckpt_idx, layer)
        if weights is not None:
            layer.set_weights(weights)
            transferred += 1
        else:
            print(f"  WARNING: unhandled layer type {type(layer).__name__} at index {ckpt_idx}")

    # ── Duplicate block 4 (layers 14-17) → blocks 5 (18-21) and 6 (22-25) ─
    block4 = layers_ww[14:18]
    for blk_start, blk_name in [(18, 'block 5'), (22, 'block 6')]:
        for offset, src_layer in enumerate(block4):
            dst_layer = layers_ww[blk_start + offset]
            dst_layer.set_weights(src_layer.get_weights())
        print(f"  {blk_name}: weights duplicated from block 4")

    print(f"Weight transfer: {transferred}/18 checkpoint layers transferred to DS-CNN-L[0-17]")
    print( "  Layers 18-25 (blocks 5-6): placeholder weights (block 4 copy)")
    print( "  Layer 26 (Dense): random initialisation")
    return transferred


def main():
    print("Building DS-CNN-L (6-block) model …")
    flags, _ = kws_util.parse_command()
    flags.model_architecture = 'ds_cnn_l'
    ds_cnn_l = models.get_model(args=flags)

    layers_ww = [l for l in ds_cnn_l.layers if l.weights]
    print(f"  Layers with weights: {len(layers_ww)}")

    print(f"Transferring weights from kws_ref_model: {REF_MODEL_PATH}")
    transfer_weights(REF_MODEL_PATH, ds_cnn_l)

    print(f"Saving to {OUT_KERAS_MODEL} …")
    ds_cnn_l.save(OUT_KERAS_MODEL)
    print(f"  Saved ({os.path.getsize(OUT_KERAS_MODEL) // 1024} KB)")

    print()
    print("Next steps:")
    print(f"  python3 convert_static_int8_v2.py --saved_model {OUT_KERAS_MODEL}")
    print( "  ./compile_edgetpu_v2.sh")
    print()
    print("NOTE: blocks 5-6 weights are a structural placeholder (block 4 copy).")
    print("      Accuracy is not representative of the deployed model.")
    print("      For proper accuracy: finetune.py + dataset → re-run convert_static_int8_v2.py")


if __name__ == '__main__':
    main()
