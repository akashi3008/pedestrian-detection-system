from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run YOLOv8 inference for pedestrian detection.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("runs/detect/pedestrian_yolov8/weights/best.pt"),
        help="Path to trained weights file.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="images/",
        help="Inference source (file, directory, or glob).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device for inference (e.g., '0', 'cpu').",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size.",
    )
    parser.add_argument(
        "--save-txt",
        action="store_true",
        help="Save predictions to .txt files (YOLO format).",
    )
    parser.add_argument(
        "--save-conf",
        action="store_true",
        help="Save confidence scores to .txt labels.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/detect",
        help="Project directory for saves.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="pedestrian_infer",
        help="Name for this inference run.",
    )
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Allow existing project/name directory.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not args.weights.exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    model = YOLO(str(args.weights))
    model.predict(
        source=args.source,
        save=True,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        device=args.device,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
    )


if __name__ == "__main__":
    main()
