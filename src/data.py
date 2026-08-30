import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import image_dataset_from_directory
from PIL import Image
from tensorflow.data.experimental import cardinality

DATA_PATH = Path(__file__).resolve().parent.parent / "data/leaves/images"


class LeafLoader:
    def __init__(
        self, data_path=None, classes=None, img_size=(96, 96), seed=42, max_images=None
    ):
        self.data_path = Path(data_path) if data_path else DATA_PATH
        if classes is None:
            # discover classes
            classes = {d.name: d.name for d in self.data_path.iterdir() if d.is_dir()}
        if not isinstance(classes, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in classes.items()
        ):
            raise Exception("Classes are not well structured")
        self.classes = classes
        self.img_size = img_size
        self.seed = seed
        self.max_images = max_images
        self._images = None

    def load_rgb(self, path):
        return np.asarray(Image.open(path).convert("RGB"))

    def load_pil(self, path):
        return Image.open(path).convert("RGB")

    def load_resized(self, path, size=None):
        size = size or self.img_size
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return cv2.resize(img, size)

    def load_datasets(self, train_dir, val_dir=None, batch_size=32, test_split=0.1):
        """Split train_dir into train/val/test datasets.

        val is 20% of train_dir (validation_split); test_split (default 0.1)
        of the remaining train pool is held out and returned as test_ds, so it
        is never seen by training.
        """
        common = dict(seed=self.seed, image_size=self.img_size, batch_size=batch_size)
        if val_dir:
            train_ds = image_dataset_from_directory(train_dir, **common)
            val_ds = image_dataset_from_directory(val_dir, **common)
        else:
            train_ds, val_ds = image_dataset_from_directory(
                train_dir, validation_split=0.2, subset="both", **common
            )

        classnames = train_ds.class_names
        n = cardinality(train_ds).numpy()

        n_test = int(n * test_split)
        test_ds = train_ds.take(n_test)
        train_ds = train_ds.skip(n_test)
        return train_ds, val_ds, test_ds, classnames

    @property
    def images(self):
        if self._images is None:
            self._images = self._resolve()
        return self._images

    def _resolve(self):
        rng = random.Random(self.seed)
        resolved = {}
        for class_name, subdir in self.classes.items():
            class_path = self.data_path / subdir
            if class_path.exists() and class_path.is_dir():
                imgs = [str(img) for img in class_path.glob("*") if img.is_file()]
                if self.max_images and len(imgs) > self.max_images:
                    imgs = sorted(rng.sample(imgs, self.max_images))
                resolved[class_name] = imgs
            else:
                resolved[class_name] = []
        return resolved

    def refresh(self):
        self._images = None
        return self.images

    @property
    def counts(self):
        return {k: len(v) for k, v in self.images.items()}

    @property
    def majority_count(self):
        return max(self.counts.values())

    @property
    def total(self):
        return sum(self.counts.values())

    @property
    def class_names(self):
        return list(self.classes.keys())

    def show_distribution(self, title="Distribution"):
        counts = self.counts
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].pie(
            list(counts.values()),
            labels=list(counts.keys()),
            autopct="%1.1f%%",
            startangle=90,
        )
        axes[0].set_title("Pie chart")
        categories = list(self.images.keys())
        my_colors = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99", "#c2c2f0"]
        axes[1].bar(categories, list(counts.values()), color=my_colors)
        axes[1].set_xticks(range(len(categories)))
        axes[1].set_xticklabels(categories, rotation=45, ha="right", fontsize=10)
        axes[1].set_title("Bar chart")
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        plt.show()

    def compare_distribution(
        self, other, title="Distribution — before vs after augmentation"
    ):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].pie(
            list(self.counts.values()),
            labels=list(self.images.keys()),
            autopct="%1.1f%%",
            startangle=90,
        )
        axes[0].set_title("Original dataset")
        axes[1].pie(
            list(other.counts.values()),
            labels=list(other.images.keys()),
            autopct="%1.1f%%",
            startangle=90,
        )
        axes[1].set_title("augmented_directory (balanced)")
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        plt.show()
