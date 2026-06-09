#!/usr/bin/env python3
"""
Fine-tune ds_cnn_l_restored.keras (6-block DS-CNN-L, properly initialized)
with the full speech_commands dataset.

Why not finetune.py?
  finetune.py does a shape-based weight transfer from kws_ref_model that only
  copies 6/82 variables.  With a mostly-random backbone frozen in Phase 1,
  gradients explode → NaN weights.

This script loads the already-initialized model from restore_ds_cnn_l_float.py
(blocks 1-4 from kws_ref_model, blocks 5-6 duplicated from block 4) and runs a
single full-fine-tune phase with all layers trainable.

Usage (from cnn-trad-fpool3_model/training):
    ~/.pyenv/versions/3.11.8/bin/python3 train_ds_cnn_l.py \
        --data_dir ~/data --epochs 36
"""

import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras

import get_dataset as kws_data
import kws_util

INIT_MODEL = 'trained_models/ds_cnn_l_restored.keras'
OUT_MODEL  = 'trained_models/ds_cnn_l_finetuned.keras'


def main():
    Flags, _ = kws_util.parse_command()
    Flags.model_architecture = 'ds_cnn_l'

    print(f"Loading initialised model: {INIT_MODEL}")
    model = tf.keras.models.load_model(INIT_MODEL)
    model.summary(line_length=80, print_fn=lambda x: None)

    for layer in model.layers:
        layer.trainable = True

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=Flags.learning_rate or 1e-4),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

    print(f"Loading dataset from {Flags.data_dir} …")
    ds_train, ds_test, ds_val = kws_data.get_training_data(Flags)
    ds_train = ds_train.shuffle(85511)

    callbacks = []
    ckpt = keras.callbacks.ModelCheckpoint(
        OUT_MODEL, monitor='val_sparse_categorical_accuracy',
        save_best_only=True, verbose=1,
    )
    callbacks.append(ckpt)
    callbacks.append(keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1))

    print(f"Training for up to {Flags.epochs} epochs …")
    model.fit(ds_train, validation_data=ds_val,
              epochs=Flags.epochs, callbacks=callbacks)

    print(f"\nEvaluating on test set …")
    scores = model.evaluate(ds_test)
    print(f"Test loss: {scores[0]:.4f}  |  Test accuracy: {scores[1]:.4f}")
    print(f"Best model saved to: {OUT_MODEL}")


if __name__ == '__main__':
    main()
