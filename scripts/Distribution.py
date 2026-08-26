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

    dataset = LeafLoader()

    print("Class counts:")
    for c, n in dataset.counts.items():
        print(f"  {c:20s}  {n}")
    print(f"\nMajority = {dataset.majority_count}, total = {dataset.total}")

    dataset.show_distribution(title="Original dataset")


if __name__ == "__main__":
    main()
