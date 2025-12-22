from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

from data_tools.dataset_utils import (
    audit_labels,
    find_empty_labels,
    find_missing_labels,
    remove_empty_labels,
)


def format_paths(paths: Iterable[Path], max_items: int = 10) -> str:
    paths = list(paths)
    if not paths:
        return ""
    preview = paths[:max_items]
    suffix = "" if len(paths) <= max_items else f" ... (+{len(paths) - max_items} more)"
    return "\n".join(f"- {p}" for p in preview) + suffix


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit YOLO-format datasets for missing or empty labels."
    )
    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Directory containing images.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Directory containing YOLO label txt files.",
    )
    parser.add_argument(
        "--remove-empty",
        action="store_true",
        help="Remove empty label files (images are kept).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the list of affected files for visibility.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.images.exists() or not args.images.is_dir():
        print(f"[ERROR] Images directory not found: {args.images}", file=sys.stderr)
        return 1
    if not args.labels.exists() or not args.labels.is_dir():
        print(f"[ERROR] Labels directory not found: {args.labels}", file=sys.stderr)
        return 1

    missing = find_missing_labels(args.images, args.labels)
    empty = find_empty_labels(args.labels)

    print(f"Missing labels: {len(missing)}")
    print(f"Empty labels: {len(empty)}")

    if args.list:
        if missing:
            print("\nImages missing labels:")
            print(format_paths(missing))
        if empty:
            print("\nEmpty label files:")
            print(format_paths(empty))

    if args.remove_empty and empty:
        removed = remove_empty_labels(empty)
        print(f"Removed empty labels: {removed}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

