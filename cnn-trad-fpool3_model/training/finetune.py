#!/usr/bin/env python
"""Fine-tune DS-CNN-L starting from the mlcommons kws_ref_model checkpoint.

The reference model (ds_cnn, 64 filters, 4 DS blocks) and DS-CNN-L (64 filters,
6 DS blocks) share the same layer shapes for the first 4 DS blocks. This script
transfers those weights and trains only the 2 new blocks + classifier from scratch
for the first few epochs, then fine-tunes the full network.
"""

import os
import re
from tensorflow import keras
import tensorflow as tf

import keras_model as models
import get_dataset as kws_data
import kws_util

# ds_cnn reference (4 DS blocks) has 18 layers with weights before Dense:
# Conv1 + BN + 4x(DW + BN + Conv1x1 + BN) = 2 + 4*4 = 18
# Layers beyond index 18 (2 new DS blocks + Dense) are trained from scratch.
REF_LAYERS_WITH_WEIGHTS = 18


def transfer_weights(ref_path, new_model):
    """Read variables directly from checkpoint files, bypassing optimizer state."""
    ckpt_prefix = os.path.join(ref_path, 'variables', 'variables')
    reader = tf.train.load_checkpoint(ckpt_prefix)
    shape_map = reader.get_variable_to_shape_map()

    def sort_key(name):
        m = re.search(r'layer_with_weights-(\d+)', name)
        return (int(m.group(1)) if m else 999, name)

    weight_names = sorted(
        [k for k in shape_map if '.ATTRIBUTES/VARIABLE_VALUE' in k],
        key=sort_key
    )
    ref_tensors = [reader.get_tensor(name) for name in weight_names]

    new_vars = list(new_model.variables)
    transferred, j = 0, 0
    for new_var in new_vars:
        while j < len(ref_tensors) and ref_tensors[j].shape != new_var.shape:
            j += 1
        if j < len(ref_tensors):
            new_var.assign(ref_tensors[j])
            transferred += 1
            j += 1

    print(f"Weight transfer: {transferred}/{len(new_vars)} variables transferred")
    return transferred


if __name__ == '__main__':
    Flags, unparsed = kws_util.parse_command()
    Flags.model_architecture = 'ds_cnn_l'

    print(f"Building DS-CNN-L and loading weights from {Flags.model_init_path}")
    new_model = models.get_model(args=Flags)
    new_model.summary()

    n_transferred = transfer_weights(Flags.model_init_path, new_model)
    if n_transferred == 0:
        print("WARNING: No weights transferred — check model_init_path")

    ds_train, ds_test, ds_val = kws_data.get_training_data(Flags)
    ds_train = ds_train.shuffle(85511)
    ds_val   = ds_val.shuffle(10102)

    # Phase 1: freeze pre-trained layers, warm up new blocks only (5 epochs)
    layers_with_weights = [l for l in new_model.layers if l.weights]
    for i, layer in enumerate(layers_with_weights):
        layer.trainable = (i >= REF_LAYERS_WITH_WEIGHTS)

    new_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    print("\n--- Phase 1: warming up new layers (5 epochs) ---")
    new_model.fit(ds_train, validation_data=ds_val, epochs=5)

    # Phase 2: unfreeze everything and fine-tune
    for layer in new_model.layers:
        layer.trainable = True

    callbacks = kws_util.get_callbacks(args=Flags)
    new_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    print("\n--- Phase 2: full fine-tuning ---")
    new_model.fit(ds_train, validation_data=ds_val,
                  epochs=Flags.epochs, callbacks=callbacks)

    new_model.save(Flags.saved_model_path)
    print(f"Saved to {Flags.saved_model_path}")

    if Flags.run_test_set:
        scores = new_model.evaluate(ds_test)
        print(f"Test loss: {scores[0]:.4f} | Test accuracy: {scores[1]:.4f}")
