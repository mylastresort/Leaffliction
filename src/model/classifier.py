import hashlib
import os
import re
import zipfile
from pathlib import Path

import cv2
import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import layers
from sklearn.metrics import classification_report, confusion_matrix

from src.data import LeafLoader
from src.model.callbacks import PATIENCE, TARGET_ACC, PerClassF1EarlyStopping

OPTIMIZERS = {
    "adam": lambda lr: keras.optimizers.Adam(learning_rate=lr),
    "sgd": lambda lr: keras.optimizers.SGD(learning_rate=lr, momentum=0.9),
    "rmsprop": lambda lr: keras.optimizers.RMSprop(learning_rate=lr),
}


class LeafClassifier:
    def __init__(self, loader=None, max_images=None):
        self.loader = loader or LeafLoader(max_images=max_images)
        self.model = None
        self.base = None

    def build_model(self, num_classes, dropout=0.2, dense_units=128, lr=0.001, optimizer="adam"):
        input_shape = (*self.loader.img_size, 3)
        inputs = keras.Input(shape=input_shape)
        x = tf.keras.applications.resnet50.preprocess_input(inputs)

        base = tf.keras.applications.ResNet50(
            include_top=False, weights="imagenet", input_tensor=inputs,
        )
        base.trainable = False

        x = base.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(dropout)(x)
        outputs = layers.Dense(dense_units, activation="relu")(x)
        outputs = layers.Dense(num_classes)(outputs)

        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer=OPTIMIZERS[optimizer](lr),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )
        self.model = model
        self.base = base
        return model, base

    def train(self, train_dir, val_dir=None, epochs=10, target_acc=TARGET_ACC,
              patience=PATIENCE, batch_size=32, dropout=0.2, dense_units=128,
              lr=0.001, optimizer="adam"):
        train_ds, val_ds, test_ds, classnames = self.loader.load_datasets(train_dir, val_dir, batch_size=batch_size)
        self.loader.classes = {name: name for name in classnames}
        num_classes = len(classnames)

        self.build_model(num_classes, dropout=dropout, dense_units=dense_units, lr=lr, optimizer=optimizer)

        print(f"Classes ({num_classes}): {classnames}")
        history = self.model.fit(
            train_ds, epochs=epochs, validation_data=val_ds,
            callbacks=[PerClassF1EarlyStopping(val_ds, classnames, target=target_acc, patience=patience)],
        )
        return self.model, history, train_ds, val_ds, test_ds, classnames

    def evaluate(self, val_ds, class_names=None):
        class_names = class_names or self.loader.class_names
        y_true = np.concatenate([y for _, y in val_ds])
        logits = self.model.predict(val_ds)
        y_pred = np.argmax(tf.nn.softmax(logits), axis=1)
        acc = (y_true == y_pred).mean()
        print(f"\nAccuracy: {acc:.4f}  (target: >95%)")
        print(f"\n{classification_report(y_true, y_pred, target_names=class_names)}")
        cm = confusion_matrix(y_true, y_pred)
        return acc, y_true, y_pred, cm

    def predict_image(self, img_path):
        """Returns (class_name, confidence, all_probs)."""
        img_resized = self.loader.load_resized(img_path)
        logits = self.model.predict(img_resized[np.newaxis, ...], verbose=0)
        probs = tf.nn.softmax(logits).numpy()[0]
        pred_idx = np.argmax(probs)
        return self.loader.class_names[pred_idx], probs[pred_idx], probs

    def test_on_images(self, test_dirs):
        """Run prediction on test directories. Returns (accuracy, details_list)."""
        test_files = sorted(f for d in test_dirs if d.exists() for f in d.glob("*.JPG"))
        if not test_files:
            print("\nNo test images found.")
            return 0.0, []

        print(f"\nFound {len(test_files)} test images\n")
        details = []
        for img_path in test_files:
            true = re.sub(r"(\d+)?\.JPG$", "", img_path.name)
            pred, _, probs = self.predict_image(img_path)
            details.append({"file": img_path.name, "true": true, "pred": pred, "probs": probs})

        correct = sum(d["true"] == d["pred"] for d in details)
        acc = correct / len(details)
        print(f"Test accuracy: {acc:.2%}  ({correct}/{len(details)})")
        print(f"\n{classification_report([d['true'] for d in details], [d['pred'] for d in details], zero_division=0)}")
        print("Details:")
        for d in details:
            mark = "OK" if d["true"] == d["pred"] else "WRONG"
            print(f"  [{mark:5s}] {d['file']:30s}  true={d['true']:25s}  pred={d['pred']}")
        return acc, details

    def show_predictions(self, test_dirs, details, acc):
        """Grid of test images with true/predicted labels colored green/red."""
        test_files = sorted(f for d in test_dirs for f in d.glob("*.JPG"))
        cols = 5
        rows = (len(test_files) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 4))
        axes = axes.flatten()
        for i, d in enumerate(details):
            for d_dir in test_dirs:
                img_path = d_dir / d["file"]
                if img_path.exists():
                    break
            img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
            color = "green" if d["true"] == d["pred"] else "red"
            axes[i].imshow(img)
            axes[i].set_title(f"{d['pred']}\n(true: {d['true']})", fontsize=8, color=color)
            axes[i].axis("off")
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")
        plt.suptitle(f"Test predictions — {acc:.0%} accuracy", fontsize=14)
        plt.tight_layout()
        plt.show()

    def show_prediction(self, rgb, masked, pred_class, confidence, probs):
        """Original image, masked image, and class probability bar chart."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(rgb)
        axes[0].set_title("Original")
        axes[0].axis("off")
        axes[1].imshow(masked)
        axes[1].set_title("Transformed (masked)")
        axes[1].axis("off")
        bars = axes[2].barh(self.loader.class_names, probs, color="steelblue")
        axes[2].set_xlabel("Probability")
        axes[2].set_title("Class Probabilities")
        axes[2].set_xlim(0, 1)
        bars[np.argmax(probs)].set_color("green")
        plt.suptitle(f"Prediction: {pred_class} ({confidence:.1%})", fontsize=14, fontweight="bold",
                     color="green" if confidence > 0.5 else "orange")
        plt.tight_layout()
        plt.show()

    def save(self, out_dir="model"):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.model.save(out / "model.keras")
        with open(out / "class_names.txt", "w") as f:
            f.write("\n".join(self.loader.class_names))
        print(f"Model saved to {out}")

    def load(self, model_dir):
        model_path = Path(model_dir) / "model.keras"
        assert model_path.exists(), f"Model not found at {model_path}."
        self.model = tf.keras.models.load_model(model_path)
        with open(Path(model_dir) / "class_names.txt") as f:
            self.loader.classes = {name: name for name in [line.strip() for line in f if line.strip()]}
        print(f"Loaded model with {len(self.loader.class_names)} classes: {self.loader.class_names}")
        return self

    def zip_artifacts(self, train_dir, model_dir, out_zip="model.zip"):
        def walk_and_add(base_dir):
            base = Path(base_dir)
            for root, _, files in os.walk(base):
                for f in files:
                    full = os.path.join(root, f)
                    arcname = Path(full).relative_to(base.parent)
                    zf.write(full, arcname)

        with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            walk_and_add(model_dir)
            walk_and_add(train_dir)
        print(f"Zipped → {out_zip}")
        return out_zip

    def sign(self, zip_path, sig_file="signature.txt"):
        sha1 = hashlib.sha1()
        with open(zip_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha1.update(chunk)
        sig = f"{sha1.hexdigest()}  {Path(zip_path).name}"
        Path(sig_file).write_text(sig)
        print(f"Signed → {sig_file}")
        print(f"  {sig}")
        return sig
