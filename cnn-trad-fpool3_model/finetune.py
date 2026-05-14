#!/usr/bin/env python
"""Fine-tune DS-CNN-L starting from the mlcommons kws_ref_model checkpoint.

The reference model (ds_cnn, 64 filters, 4 DS blocks) and DS-CNN-L (64 filters,
6 DS blocks) share the same layer shapes for the first 4 DS blocks. This script
transfers those weights and trains only the 2 new blocks + classifier from scratch
for the first few epochs, then fine-tunes the full network.
"""

import argparse
import numpy as np
from tensorflow import keras
import tensorflow as tf

import keras_model as models
import get_dataset as kws_data
import kws_util


def transfer_weights(ref_model, new_model):
    """Copy weights layer-by-layer where shapes match."""
    ref_layers = [l for l in ref_model.layers if l.get_weights()]
    new_layers = [l for l in new_model.layers if l.get_weights()]

    transferred, skipped = 0, 0
    for ref_l, new_l in zip(ref_layers, new_layers):
        ref_w = ref_l.get_weights()
        new_w = new_l.get_weights()
        if all(r.shape == n.shape for r, n in zip(ref_w, new_w)):
            new_l.set_weights(ref_w)
            transferred += 1
        else:
            skipped += 1

    print(f"Weight transfer: {transferred} layers transferred, {skipped} skipped (new layers)")
    return transferred


if __name__ == '__main__':
    Flags, unparsed = kws_util.parse_command()

    # Override architecture to ds_cnn_l
    Flags.model_architecture = 'ds_cnn_l'

    print(f"Loading reference checkpoint from {Flags.model_init_path}")
    ref_model = keras.models.load_model(Flags.model_init_path)

    print("Building DS-CNN-L model")
    new_model = models.get_model(args=Flags)
    new_model.summary()

    n_transferred = transfer_weights(ref_model, new_model)
    if n_transferred == 0:
        print("WARNING: No weights transferred — check that model_init_path points to kws_ref_model")

    ds_train, ds_test, ds_val = kws_data.get_training_data(Flags)
    ds_train = ds_train.shuffle(85511)
    ds_val   = ds_val.shuffle(10102)

    # Phase 1: freeze transferred layers, train only new blocks (~5 epochs)
    ref_layers_with_weights = [l for l in ref_model.layers if l.get_weights()]
    new_layers_with_weights = [l for l in new_model.layers if l.get_weights()]
    for i, layer in enumerate(new_layers_with_weights):
        layer.trainable = (i >= n_transferred)

    new_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    print("\n--- Phase 1: training new layers only (5 epochs) ---")
    new_model.fit(ds_train, validation_data=ds_val, epochs=5)

    # Phase 2: unfreeze all layers and fine-tune
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
    print(f"Model saved to {Flags.saved_model_path}")

    if Flags.run_test_set:
        scores = new_model.evaluate(ds_test)
        print(f"Test loss: {scores[0]:.4f} | Test accuracy: {scores[1]:.4f}")
