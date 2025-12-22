from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
LABEL_EXT = ".txt"


@dataclass(frozen=True)
class LabelCheckResult:
    """Structured result for label audits."""

    missing: List[Path]
    empty: List[Path]


def iter_images(images_dir: Path) -> Iterable[Path]:
    """Yield image files recursively under images_dir."""
    for img in images_dir.rglob("*"):
        if img.is_file() and img.suffix.lower() in IMAGE_EXTS:
            yield img


def iter_labels(labels_dir: Path) -> Iterable[Path]:
    """Yield label files recursively under labels_dir."""
    for label in labels_dir.rglob(f"*{LABEL_EXT}"):
        if label.is_file():
            yield label


def find_missing_labels(images_dir: Path, labels_dir: Path) -> List[Path]:
    """Return images that do not have a corresponding label file."""
    missing: List[Path] = []
    for img in iter_images(images_dir):
        relative = img.relative_to(images_dir)
        label = labels_dir / relative.with_suffix(LABEL_EXT)
        if not label.exists():
            missing.append(img)
    return missing


def find_empty_labels(labels_dir: Path) -> List[Path]:
    """Return label files that exist but are empty."""
    empty: List[Path] = []
    for label in iter_labels(labels_dir):
        if label.stat().st_size == 0:
            empty.append(label)
    return empty


def remove_empty_labels(empty_labels: Sequence[Path]) -> int:
    """Remove provided empty label files. Returns count removed."""
    removed = 0
    for label in empty_labels:
        label.unlink()
        removed += 1
    return removed


def audit_labels(images_dir: Path, labels_dir: Path) -> LabelCheckResult:
    """Audit labels, returning missing images and empty label files."""
    missing = find_missing_labels(images_dir, labels_dir)
    empty = find_empty_labels(labels_dir)
    return LabelCheckResult(missing=missing, empty=empty)
