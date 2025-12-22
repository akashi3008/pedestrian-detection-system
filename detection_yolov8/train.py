from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a YOLOv8 pedestrian detector.")
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8s.pt",
        help="Base model weights or model name (e.g., yolov8s.pt).",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("dataset.yaml"),
        help="Path to YOLO dataset yaml.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument(
        "--name",
        type=str,
        default="pedestrian_yolov8",
        help="Training run name.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device to train on (e.g., '0', '0,1', 'cpu').",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
