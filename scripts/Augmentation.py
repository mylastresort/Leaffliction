import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

from src.augmentation import LeafAugmenter
from src.data import LeafLoader

HELP_TEXT = """Usage:
  Augmentation.py <image_path>       Display and save 6 augmentations for one image
  Augmentation.py -h                 Show this help message

When run without arguments, balances the full dataset into augmented_directory/."""


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-h", "--help", action="store_true")
    parser.add_argument("image", nargs="?", default=None)
    args = parser.parse_args()
    if args.help:
        print(HELP_TEXT)
        return

    root = Path(__file__).resolve().parent.parent
    loader = LeafLoader()
    augmenter = LeafAugmenter(loader=loader)

    if args.image:
        img_path = Path(args.image)
        assert img_path.exists(), f"Image not found: {img_path}"
        print(f"Augmenting: {img_path}")
        for f in augmenter.save_augmentations(img_path):
            print(f"  {f}")
        return

    max_len = loader.majority_count
    sample_path = loader.images["Apple_healthy"][0]
    augmenter.preview_sample(sample_path)

    out = augmenter.balance_dataset(
        loader.images, max_len=max_len, out_dir=str(root / "augmented_directory")
    )
    print(f"Saved to {out}")

    balanced = LeafLoader(data_path=root / "augmented_directory")
    augmenter.show_balance_comparison(loader, balanced)


if __name__ == "__main__":
    main()
