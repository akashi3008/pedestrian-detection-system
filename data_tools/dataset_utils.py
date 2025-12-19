from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
LABEL_EXT = ".txt"


def find_missing_labels(images_dir: Path, labels_dir: Path):
    missing = []
    for img in images_dir.rglob("*"):
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        label = labels_dir / (img.stem + LABEL_EXT)
        if not label.exists():
            missing.append(img)
    return missing


def remove_empty_labels(images_dir: Path, labels_dir: Path):
    removed = 0
    for img in images_dir.iterdir():
        label = labels_dir / (img.stem + LABEL_EXT)
        if label.exists() and label.stat().st_size == 0:
            img.unlink()
            label.unlink()
            removed += 1
    return removed
