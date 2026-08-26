#!/usr/bin/env python3
"""Part 4 — Predict
Load a trained model, display the original and transformed image, and classify the leaf disease.

Usage:
  ./Predict.py <image_path>   Classify a leaf image
  ./Predict.py -h             Show this help message
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

import plantcv.plantcv as pcv

from src.data import LeafLoader
from src.features import LeafTransformer
from src.model import LeafClassifier

root = Path(__file__).resolve().parent.parent
MODEL_DIR = root / "model"

HELP_TEXT = """Usage:
  Predict.py <image_path>   Classify a leaf image
  Predict.py -h             Show this help message"""


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
    classifier = LeafClassifier(loader=loader)
    classifier.load(MODEL_DIR)

    if args.image:
        img_path = Path(args.image)
    else:
        test_image = root / "test" / "Unit_test1" / "Apple_healthy1.JPG"
        img_path = (
            test_image if test_image.exists() else loader.images["Apple_healthy"][0]
        )
        print(f"Using sample image: {img_path}")

    assert Path(img_path).exists(), f"Image not found: {img_path}"

    rgb = loader.load_rgb(img_path)
    mask = transformer.get_mask(rgb)
    masked = pcv.apply_mask(img=rgb, mask=mask, mask_color="white")

    pred_class, confidence, probs = classifier.predict_image(img_path)

    classifier.show_prediction(rgb, masked, pred_class, confidence, probs)

    print(f"\nResult: {pred_class} (confidence: {confidence:.2%})")
    print(f"\nAll probabilities:")
    for name, prob in sorted(zip(loader.class_names, probs), key=lambda x: -x[1]):
        print(f"  {name:25s}  {prob:.4f}")


if __name__ == "__main__":
    main()
