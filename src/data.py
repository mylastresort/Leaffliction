import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import image_dataset_from_directory
from PIL import Image

DATA_PATH = Path(__file__).resolve().parent.parent / "data/leaves/images"
CLASSES = {
    "Apple_Black_rot": "Apple_Black_rot",
    "Apple_healthy": "Apple_healthy",
    "Apple_rust": "Apple_rust",
    "Apple_scab": "Apple_scab",
    "Grape_Black_rot": "Grape_Black_rot",
    "Grape_Esca": "Grape_Esca",
    "Grape_healthy": "Grape_healthy",
    "Grape_spot": "Grape_spot",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


class LeafLoader:
    def __init__(self, data_path=None, classes=None, img_size=(96, 96), seed=42, max_images=None):
        self.data_path = Path(data_path) if data_path else DATA_PATH
        self.classes = classes or CLASSES
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

    def load_datasets(self, train_dir, val_dir=None, batch_size=32):
        common = dict(seed=self.seed, image_size=self.img_size, batch_size=batch_size)
        if val_dir:
            train_ds = image_dataset_from_directory(train_dir, **common)
            val_ds = image_dataset_from_directory(val_dir, **common)
        else:
            train_ds, val_ds = image_dataset_from_directory(
                train_dir, validation_split=0.2, subset="both", **common
            )
        if train_ds.class_names != val_ds.class_names:
            raise ValueError("Class mismatch between train and validation")
        return train_ds, val_ds

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
                imgs = [
                    str(img)
                    for img in class_path.glob("*")
                    if img.is_file() and img.suffix in IMAGE_EXTENSIONS
                ]
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
        axes[0].pie(list(counts.values()), labels=list(counts.keys()), autopct="%1.1f%%", startangle=90)
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

    def compare_distribution(self, other, title="Distribution — before vs after augmentation"):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].pie(list(self.counts.values()), labels=list(self.images.keys()), autopct="%1.1f%%", startangle=90)
        axes[0].set_title("Original dataset")
        axes[1].pie(list(other.counts.values()), labels=list(other.images.keys()), autopct="%1.1f%%", startangle=90)
        axes[1].set_title("augmented_directory (balanced)")
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        plt.show()
