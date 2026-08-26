import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score

TARGET_ACC = 0.98
PATIENCE = 5


class PerClassF1EarlyStopping(tf.keras.callbacks.Callback):
    """Stop when the *worst-performing* class's F1 >= target, with patience on that minimum."""

    def __init__(self, val_ds, class_names, target=TARGET_ACC, patience=PATIENCE):
        super().__init__()
        self.val_ds = val_ds
        self.class_names = class_names
        self.target = target
        self.patience = patience
        self.best_min_f1 = 0.0
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        y_true = np.concatenate([y for _, y in self.val_ds])
        logits = self.model.predict(self.val_ds, verbose=0)
        y_pred = np.argmax(tf.nn.softmax(logits), axis=1)

        per_class_f1 = f1_score(
            y_true,
            y_pred,
            average=None,
            labels=range(len(self.class_names)),
            zero_division=0,
        )

        # expose per-class scores in logs/history for later inspection
        logs = logs or {}
        for name, f1 in zip(self.class_names, per_class_f1):
            logs[f"val_f1_{name}"] = f1

        min_f1 = per_class_f1.min()
        worst = self.class_names[per_class_f1.argmin()]
        print(
            "\nPer-class F1: "
            + ", ".join(f"{n}={f:.3f}" for n, f in zip(self.class_names, per_class_f1))
        )
        print(f"Worst class: {worst} ({min_f1:.4f})")

        if min_f1 >= self.target:
            print(f"min per-class F1 {min_f1:.4f} >= {self.target} — stopping")
            self.model.stop_training = True
            return

        if min_f1 > self.best_min_f1:
            self.best_min_f1 = min_f1
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(
                    f"No improvement in min per-class F1 for {self.patience} epochs — early stopping"
                )
                self.model.stop_training = True
