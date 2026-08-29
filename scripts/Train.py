#!/usr/bin/env python3
"""Part 4 — Train
Balance (Part 2) + train ResNet50 CNN, evaluate, save artifacts.
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

from src.augmentation import LeafAugmenter
from src.data import LeafLoader
from src.model import LeafClassifier

warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")

HELP_TEXT = """Usage:
  Train.py                          Run full pipeline (balance, train, evaluate, save)
  Train.py -h                       Show this help message
  Train.py --train-dir <dir>        Use pre-balanced directory for training
  Train.py --epochs <n>             Number of epochs (default: 100)
  Train.py --target-acc <float>     Target accuracy (default: 0.90)
  Train.py --patience <n>           Early stopping patience (default: 10)"""


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-h", "--help", action="store_true")
    parser.add_argument("--train-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--target-acc", type=float, default=0.90)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()
    if args.help:
        print(HELP_TEXT)
        return

    root = Path(__file__).resolve().parent.parent

    if args.train_dir:
        train_dir = args.train_dir
    else:
        train_dir = str(root / "augmented_directory")
        augmented_dir = Path(train_dir)
        if not augmented_dir.exists():
            print("Balancing dataset...")
            loader = LeafLoader()
            augmenter = LeafAugmenter(loader=loader)
            print("Before:", loader.counts)
            augmenter.balance_dataset(
                loader.images, max_len=loader.majority_count, out_dir=str(augmented_dir)
            )
            balanced = LeafLoader(data_path=augmented_dir)
            print("After:", balanced.counts)
        else:
            print(f"Using existing balanced dataset at {train_dir}")

    print(f"\nTraining CNN...")
    classifier = LeafClassifier()
    _, _, _, val_ds, test_ds, _ = classifier.train(
        train_dir=train_dir,
        epochs=args.epochs,
        target_acc=args.target_acc,
        patience=args.patience,
    )

    print("\nEvaluating validation split...")
    classifier.evaluate(val_ds)

    print("\nEvaluating held-out test split (returned from load_datasets, unseen from training)...")
    classifier.evaluate(test_ds)

    model_dir = str(root / "model")
    classifier.save(out_dir=model_dir)

    zip_path = classifier.zip_artifacts(
        train_dir=train_dir, model_dir=model_dir, out_zip=str(root / "model.zip")
    )
    classifier.sign(zip_path=zip_path, sig_file=str(root / "signature.txt"))

    acc, details = classifier.test_on_images(
        [root / "test" / d for d in ["Unit_test1", "Unit_test2"]]
    )

    if details:
        classifier.show_predictions(
            [root / "test" / d for d in ["Unit_test1", "Unit_test2"]], details, acc
        )


if __name__ == "__main__":
    main()
