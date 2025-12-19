import argparse
from pathlib import Path
from dataset_utils import find_missing_labels, remove_empty_labels

parser = argparse.ArgumentParser()
parser.add_argument("--images", type=Path)
parser.add_argument("--labels", type=Path)
parser.add_argument("--remove-empty", action="store_true")

args = parser.parse_args()

missing = find_missing_labels(args.images, args.labels)
print(f"Missing labels: {len(missing)}")

if args.remove_empty:
    removed = remove_empty_labels(args.images, args.labels)
    print(f"Removed empty labels: {removed}")
