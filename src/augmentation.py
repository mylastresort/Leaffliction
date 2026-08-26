from pathlib import Path

import albumentations as A
import matplotlib.pyplot as plt
from PIL import Image

from src.data import LeafLoader
import shutil


class LeafAugmenter:
    TRANSFORMS = {
        "Flip": A.HorizontalFlip(p=1.0),
        "Rotate": A.Rotate(limit=45, p=1.0),
        "Skew": A.Perspective(scale=(0.1, 0.15), p=1.0),
        "Shear": A.Affine(shear=(-20, 20), p=1.0),
        "Crop": A.RandomResizedCrop(size=(224, 224), scale=(0.7, 0.9), p=1.0),
        "Distortion": A.OpticalDistortion(distort_limit=0.15, p=1.0),
    }

    def __init__(self, loader=None):
        self.loader = loader or LeafLoader()

    def augment_one(self, rgb, transform):
        return transform(image=rgb)["image"]

    def save_augmentations(self, img_path, out_dir=None):
        """Save 6 augmentations alongside original, named <stem>_<Type>.<ext>"""
        rgb = self.loader.load_rgb(img_path)
        stem = Path(img_path).stem
        ext = Path(img_path).suffix
        out = Path(out_dir) if out_dir else Path(img_path).parent
        out.mkdir(parents=True, exist_ok=True)
        saved = []
        for name, tfm in self.TRANSFORMS.items():
            aug = self.augment_one(rgb, tfm)
            fname = out / f"{stem}_{name}{ext}"
            Image.fromarray(aug).save(fname)
            saved.append(fname)
        return saved

    def balance_dataset(self, images, max_len=None, out_dir="augmented_directory"):
        """For each class, keep all originals + augment until count == majority."""
        out_dir = Path(out_dir)
        if out_dir.exists():
            answer = input(f"{out_dir} already exists. Empty it? [y/N]: ")
            if answer.lower() == "y":
                shutil.rmtree(out_dir)
                out_dir.mkdir(parents=True)
            else:
                print("Keeping existing directory.")
        else:
            out_dir.mkdir(parents=True)
        t_list = list(self.TRANSFORMS.values())
        for cls, img_list in images.items():
            cls_dir = out_dir / cls
            cls_dir.mkdir(parents=True, exist_ok=True)
            for img_path in img_list:
                dest = cls_dir / Path(img_path).name
                if not dest.exists():
                    Image.open(img_path).convert("RGB").save(dest)
            n = len(img_list)
            idx = 0
            while n < max_len:
                rgb = self.loader.load_rgb(img_list[idx % len(img_list)])
                tfm = t_list[n % len(t_list)]
                aug = self.augment_one(rgb, tfm)
                fname = (
                    cls_dir
                    / f"{Path(img_list[idx % len(img_list)]).stem}_aug_{n}{Path(img_list[0]).suffix}"
                )
                Image.fromarray(aug).save(fname)
                n += 1
                idx += 1
        return out_dir

    def preview_sample(self, sample_path):
        """Display original + 6 augmentations for one image."""
        rgb = self.loader.load_rgb(sample_path)
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes[0, 0].imshow(rgb)
        axes[0, 0].set_title("Original")
        axes[0, 0].axis("off")
        for ax, (name, tfm) in zip(axes.flat[1:], self.TRANSFORMS.items()):
            ax.imshow(self.augment_one(rgb, tfm))
            ax.set_title(name)
            ax.axis("off")
        axes.flat[-1].axis("off")
        plt.suptitle("6 Augmentations — one technique per output")
        plt.tight_layout()
        plt.show()

    def show_balance_comparison(self, original_loader, balanced_loader):
        """Pie charts: original vs balanced dataset."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].pie(
            list(original_loader.counts.values()),
            labels=list(original_loader.images.keys()),
            autopct="%1.1f%%",
        )
        axes[0].set_title("Before augmentation")
        axes[1].pie(
            list(balanced_loader.counts.values()),
            labels=list(balanced_loader.images.keys()),
            autopct="%1.1f%%",
        )
        axes[1].set_title("After augmentation (augmented_directory)")
        plt.suptitle("Distribution — before vs after augmentation", fontsize=14)
        plt.tight_layout()
        plt.show()
