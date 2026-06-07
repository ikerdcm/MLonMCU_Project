#!/usr/bin/env python3
"""
Knowledge distillation for the pruned DS-CNN branches (int8-prune-distill / v4).

Teacher = the 6-block FP32 model (ds_cnn_l_finetuned.keras). Student = a pruned
branch (warm-started from its trained weights). Loss = alpha*CE(hard labels) +
(1-alpha)*KL(temperature-softened teacher vs student). Recovers the accuracy
that pruning + int8 cost — most for the heavily-pruned 32-filter students.

Examples (run from training/):
  python3 distill_branch.py --student f64b4 --epochs 20 --data_dir ~/data
  python3 distill_branch.py --student f32b4 --epochs 25 --data_dir ~/data
  python3 distill_branch.py --student f64b4 --epochs 1 --max-steps 80 --data_dir ~/data   # smoke
"""
import os, sys, argparse, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import keras

pre = argparse.ArgumentParser(add_help=False)
pre.add_argument('--student', required=True, help='branch stem, e.g. f64b4 (loads ds_cnn_l_pruned_<stem>.keras)')
pre.add_argument('--teacher', default='trained_models/ds_cnn_l_finetuned.keras')
pre.add_argument('--alpha', type=float, default=0.3, help='weight on hard CE (1-alpha on soft KL)')
pre.add_argument('--temp', type=float, default=4.0)
pre.add_argument('--max-steps', type=int, default=None)
myargs, rest = pre.parse_known_args()
sys.argv = [sys.argv[0]] + rest

import kws_util
import get_dataset as kws_data

STUDENT_IN = f"trained_models/ds_cnn_l_pruned_{myargs.student}.keras"
OUT        = f"trained_models/ds_cnn_l_distilled_{myargs.student}.keras"


class Distiller(keras.Model):
    def __init__(self, student, teacher, alpha, temp):
        super().__init__()
        self.student, self.teacher = student, teacher
        self.alpha, self.temp = alpha, temp
        self.ce = keras.losses.SparseCategoricalCrossentropy()
        self.kld = keras.losses.KLDivergence()
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.acc = keras.metrics.SparseCategoricalAccuracy(name="accuracy")

    @property
    def metrics(self):
        return [self.loss_tracker, self.acc]

    def call(self, x, training=False):
        return self.student(x, training=training)

    def _soften(self, p):
        p = tf.pow(tf.maximum(p, 1e-9), 1.0 / self.temp)
        return p / tf.reduce_sum(p, axis=-1, keepdims=True)

    def train_step(self, data):
        x, y = data
        t = self.teacher(x, training=False)
        with tf.GradientTape() as tape:
            s = self.student(x, training=True)
            hard = self.ce(y, s)
            soft = self.kld(self._soften(t), self._soften(s))
            loss = self.alpha * hard + (1.0 - self.alpha) * soft
        grads = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.student.trainable_variables))
        self.loss_tracker.update_state(loss); self.acc.update_state(y, s)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        s = self.student(x, training=False)
        self.loss_tracker.update_state(self.ce(y, s)); self.acc.update_state(y, s)
        return {m.name: m.result() for m in self.metrics}


class SaveBestStudent(keras.callbacks.Callback):
    def __init__(self, path): super().__init__(); self.best = -1.0; self.path = path
    def on_epoch_end(self, e, logs=None):
        v = (logs or {}).get('val_accuracy')
        print(f"[epoch {e}] val_acc={v}")
        if v is not None and v > self.best:
            self.best = v; self.model.student.save(self.path)
            print(f"  saved student (best val_acc={v:.4f}) -> {self.path}")


def main():
    Flags, _ = kws_util.parse_command()
    ds_train, ds_test, ds_val = kws_data.get_training_data(Flags)

    teacher = keras.models.load_model(myargs.teacher, compile=False); teacher.trainable = False
    student = keras.models.load_model(STUDENT_IN, compile=False)
    print(f"teacher={myargs.teacher} ({teacher.count_params():,} p)  student={STUDENT_IN} ({student.count_params():,} p)  alpha={myargs.alpha} T={myargs.temp}")

    d = Distiller(student, teacher, myargs.alpha, myargs.temp)
    d.compile(optimizer=keras.optimizers.Adam(learning_rate=Flags.learning_rate or 5e-5))

    cbs = [] if myargs.max_steps else [SaveBestStudent(OUT)]
    fit_kw = dict(validation_data=ds_val, epochs=Flags.epochs, callbacks=cbs, verbose=2)
    if myargs.max_steps:
        fit_kw['steps_per_epoch'] = myargs.max_steps; fit_kw['validation_steps'] = 20
        print(f"SMOKE TEST: {myargs.max_steps} steps")
    t0 = time.time()
    d.fit(ds_train, **fit_kw)
    print(f"done in {time.time()-t0:.1f}s" + ("" if myargs.max_steps else f"; best student -> {OUT}"))


if __name__ == '__main__':
    main()
