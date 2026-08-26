#!/usr/bin/env python3
"""Part 1 — Distribution
Analyze the dataset and display pie charts and bar charts for each plant type.
Also verifies the augmented_directory balance after Part 2.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

from src.data import LeafLoader

HELP_TEXT = """Usage:
  Distribution.py    Show dataset distribution and augmented_directory verification
  Distribution.py -h Show this help message"""


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-h", "--help", action="store_true")
    args = parser.parse_args()
    if args.help:
        print(HELP_TEXT)
        return

    root = Path(__file__).resolve().parent.parent
    dataset = LeafLoader()

    print("Class counts:")
    for c, n in dataset.counts.items():
        print(f"  {c:20s}  {n}")
    print(f"\nMajority = {dataset.majority_count}, total = {dataset.total}")

    dataset.show_distribution(title="Original dataset")

    augmented_dir = root / "augmented_directory"
    if augmented_dir.exists():
        aug = LeafLoader(data_path=augmented_dir)
        print("augmented_directory class counts:")
        for c, n in aug.counts.items():
            print(f"  {c:20s}  {n}")
        print(f"\nAll equal: {len(set(aug.counts.values())) == 1}")
        dataset.compare_distribution(aug)
    else:
        print(f"\naugmented_directory not found at {augmented_dir}")
        print("Run Augmentation.py first to generate it.")


if __name__ == "__main__":
    main()
