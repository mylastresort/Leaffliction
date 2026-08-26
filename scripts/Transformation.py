import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

from src.data import LeafLoader
from src.features import LeafTransformer

HELP_TEXT = """Usage:
  Transformation.py <image_path>               Show 6 transformations for a single image
  Transformation.py -h                         Show this help message"""


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-h", "--help", action="store_true")
    parser.add_argument("image", nargs="?", default=None)
    args = parser.parse_args()
    if args.help:
        print(HELP_TEXT)
        return

    loader = LeafLoader()
    transformer = LeafTransformer(loader=loader)

    if args.image:
        img_path = Path(args.image)
        assert img_path.exists(), f"Image not found: {img_path}"
        print(f"Processing: {img_path}")
        rgb = loader.load_rgb(img_path)
        print(f"image shape: {rgb.shape}")
        transformer.display_results(transformer.transform(rgb))
        return

    exit("Script must take image as input")


if __name__ == "__main__":
    main()
