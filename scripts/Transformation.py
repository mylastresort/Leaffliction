#!/usr/bin/env python3
"""Part 3 — Image Transformation (plantCV)
1. Gaussian Blur  2. Mask  3. ROI Objects  4. Analyze Object  5. Pseudolandmarks  6. Color Histogram

Usage:
  ./Transformation.py <image_path>                  Show 6 transformations for a single image
  ./Transformation.py -src <dir> -dst <dir> [-mask] Save transformations for all images
  ./Transformation.py -h                            Show this help message
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

from src.data import LeafLoader
from src.features import LeafTransformer

HELP_TEXT = """Usage:
  Transformation.py <image_path>               Show 6 transformations for a single image
  Transformation.py -src <dir> -dst <dir>      Save transformations for all images
  Transformation.py -h                         Show this help message"""


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-h", "--help", action="store_true")
    parser.add_argument("-src", type=str, default=None)
    parser.add_argument("-dst", type=str, default=None)
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

    if args.src:
        dst_dir = Path(args.dst) if args.dst else Path("transformed_output")
        transformer.process_directory(args.src, dst_dir)
        return

    sample_path = loader.images["Apple_healthy"][0]
    rgb = loader.load_rgb(sample_path)
    print(f"image shape: {rgb.shape}")
    transformer.display_results(transformer.transform(rgb))


if __name__ == "__main__":
    main()
