#!/usr/bin/env python3
"""
Structured-pruning branch trainer for DS-CNN-L (Coral v3 campaign).

Builds a smaller DS-CNN (fewer filters and/or fewer DS blocks), warm-starts it
from the trained 6-block model (ds_cnn_l_finetuned.keras) by transferring the
matching layers — slicing channels (first-N) when narrower, dropping trailing
blocks when shallower — then fine-tunes. Each branch is saved under its own
name (nothing overwritten).

Examples (run from training/):
    python3 prune_branch.py --filters 32 --blocks 6 --branch f32b6 --epochs 15 --data_dir ~/data
    python3 prune_branch.py --filters 64 --blocks 4 --branch f64b4 --epochs 15 --data_dir ~/data
    python3 prune_branch.py --filters 32 --blocks 4 --branch f32b4 --epochs 15 --data_dir ~/data
    # smoke test (time one capped epoch):
    python3 prune_branch.py --filters 64 --blocks 4 --branch smoke --epochs 1 --max-steps 100 --data_dir ~/data
"""
import os, sys, argparse, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout, AveragePooling2D, Flatten, Dense
from tensorflow.keras.models import Model

# --- our args first, then hand the rest to kws_util.parse_command() ---
pre = argparse.ArgumentParser(add_help=False)
pre.add_argument('--filters', type=int, default=64)
pre.add_argument('--blocks',  type=int, default=6)
pre.add_argument('--branch',  default='prune')
pre.add_argument('--max-steps', type=int, default=None, help='cap steps/epoch (smoke test)')
pre.add_argument('--teacher', default='trained_models/ds_cnn_l_finetuned.keras')
myargs, rest = pre.parse_known_args()
sys.argv = [sys.argv[0]] + rest

import kws_util
import get_dataset as kws_data
from keras_model import _ds_cnn_body

INPUT_SHAPE = (49, 10, 1)
NUM_CLASSES = 12
OUT = f"trained_models/ds_cnn_l_pruned_{myargs.branch}.keras"


def build_student(filters, blocks):
    inp = Input(shape=INPUT_SHAPE)
    x = _ds_cnn_body(inp, filters=filters, num_ds_blocks=blocks)
    x = Dropout(0.4)(x)
    x = AveragePooling2D(pool_size=(INPUT_SHAPE[0] // 2, INPUT_SHAPE[1] // 2))(x)
    x = Flatten()(x)
    out = Dense(NUM_CLASSES, activation='softmax')(x)
    return Model(inp, out)


def warm_start(student, teacher_path):
    """Transfer matching layers from the 6-block teacher; slice channels when
    narrower (first-N), skip the final Dense (feature dims differ)."""
    teacher = tf.keras.models.load_model(teacher_path, compile=False)
    t = [l for l in teacher.layers if l.get_weights()]
    s = [l for l in student.layers if l.get_weights()]
    transferred = sliced = 0
    # align from the front; last student layer is the Dense head -> leave random
    for i, sl in enumerate(s[:-1]):
        if i >= len(t):
            break
        tw, sw = t[i].get_weights(), sl.get_weights()
        if len(tw) != len(sw):
            continue
        new = []
        ok = True
        for a, b in zip(sw, tw):       # a = student-shaped, b = teacher weights
            if a.shape == b.shape:
                new.append(b)
            elif a.ndim == b.ndim:
                new.append(b[tuple(slice(0, d) for d in a.shape)]); sliced += 1
            else:
                ok = False; break
        if ok:
            sl.set_weights(new); transferred += 1
    print(f"warm-start: transferred {transferred}/{len(s)-1} layers ({sliced} sliced); Dense reinit")
    return student


class EpochTimer(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, e, logs=None): self.t = time.time()
    def on_epoch_end(self, e, logs=None):
        print(f"[timing] epoch {e} took {time.time()-self.t:.1f}s  (val_acc={logs.get('val_accuracy')})")


def main():
    Flags, _ = kws_util.parse_command()
    ds_train, ds_test, ds_val = kws_data.get_training_data(Flags)

    model = build_student(myargs.filters, myargs.blocks)
    n_macs_note = f"filters={myargs.filters}, blocks={myargs.blocks}, params={model.count_params():,}"
    print(f"student: {n_macs_note}")
    warm_start(model, myargs.teacher)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=Flags.learning_rate or 1e-4),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    callbacks = [EpochTimer()]
    if not myargs.max_steps:
        callbacks += [
            tf.keras.callbacks.ModelCheckpoint(OUT, monitor='val_accuracy', mode='max',
                                               save_best_only=True, verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.5,
                                                 patience=4, min_lr=1e-5, verbose=1),
        ]
    fit_kw = dict(validation_data=ds_val, epochs=Flags.epochs, callbacks=callbacks, verbose=2)
    if myargs.max_steps:
        fit_kw['steps_per_epoch'] = myargs.max_steps
        fit_kw['validation_steps'] = 20
        print(f"SMOKE TEST: {myargs.max_steps} steps/epoch (timing only)")
    t0 = time.time()
    model.fit(ds_train, **fit_kw)
    print(f"total fit: {time.time()-t0:.1f}s")

    if myargs.max_steps:
        print("(smoke test — model not saved)")
    else:
        print(f"best model saved to {OUT}")


if __name__ == '__main__':
    main()
