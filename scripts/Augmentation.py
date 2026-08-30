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
    else:
        exit("No image provided. Use -h for help.")


if __name__ == "__main__":
    main()
