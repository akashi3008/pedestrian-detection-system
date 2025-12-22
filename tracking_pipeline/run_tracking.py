from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np
import yaml
from tqdm import tqdm
from yolox.tracker.byte_tracker import BYTETracker, STrack

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


@dataclass
class TrackerConfig:
    track_thresh: float = 0.5
    match_thresh: float = 0.8
    track_buffer: int = 30
    frame_rate: int = 30
    mot20: bool = False
    min_box_area: float = 10.0
    aspect_ratio_thresh: float = 1.6

    @classmethod
    def from_file(cls, path: Path) -> "TrackerConfig":
        if not path.exists():
            raise FileNotFoundError(f"Tracker config not found: {path}")
        data = yaml.safe_load(path.read_text()) or {}
        defaults = asdict(cls())

        unexpected = set(data) - set(defaults)
        if unexpected:
            raise ValueError(f"Unexpected tracker config keys: {sorted(unexpected)}")

        merged = {**defaults, **data}
        return cls(**merged)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ByteTrack on YOLO detections.")
    parser.add_argument("--frames-dir", type=Path, required=True, help="Directory of input frames.")
    parser.add_argument(
        "--detections-dir",
        type=Path,
        required=True,
        help="Directory containing YOLO txt detections matching frame stems.",
    )
    parser.add_argument(
        "--tracker-config",
        type=Path,
        default=Path("tracker.yaml"),
        help="YAML file with tracker parameters.",
    )
    parser.add_argument(
        "--output-video",
        type=Path,
        default=Path("tracks.mp4"),
        help="Path to save visualization video.",
    )
    parser.add_argument(
        "--save-tracks",
        type=Path,
        default=Path("tracks.csv"),
        help="CSV file to save track results (frame,id,x1,y1,x2,y2,score).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional frame limit for quick runs (0 = all frames).",
    )
    return parser


def image_files(frames_dir: Path) -> List[Path]:
    return sorted(
        p for p in frames_dir.glob("*") if p.suffix.lower() in IMAGE_EXTS and p.is_file()
    )


def load_yolo_detections(det_path: Path, img_shape: Tuple[int, int]) -> np.ndarray:
    """Load YOLO-format detections (class x y w h conf) -> xyxy+score array."""
    if not det_path.exists():
        return np.empty((0, 5), dtype=float)
    h, w = img_shape
    detections = []
    for line in det_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 6:
            continue
        _, x_c, y_c, bw, bh, conf = map(float, parts[:6])
        x1 = max((x_c - bw / 2) * w, 0)
        y1 = max((y_c - bh / 2) * h, 0)
        x2 = min((x_c + bw / 2) * w, w)
        y2 = min((y_c + bh / 2) * h, h)
        detections.append([x1, y1, x2, y2, conf])
    return np.asarray(detections, dtype=float)


def tracker_from_config(cfg: TrackerConfig) -> BYTETracker:
    class Args:
        pass

    args = Args()
    args.track_thresh = cfg.track_thresh
    args.match_thresh = cfg.match_thresh
    args.track_buffer = cfg.track_buffer
    args.mot20 = cfg.mot20
    args.min_box_area = cfg.min_box_area
    args.aspect_ratio_thresh = cfg.aspect_ratio_thresh
    return BYTETracker(args, frame_rate=cfg.frame_rate)


def draw_tracks(frame: np.ndarray, tracks: Iterable[STrack]) -> np.ndarray:
    vis = frame.copy()
    for track in tracks:
        tlwh = track.tlwh
        x1, y1, w, h = map(int, tlwh)
        x2, y2 = x1 + w, y1 + h
        track_id = track.track_id
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            vis,
            f"ID {track_id}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
    return vis


def write_tracks_csv(rows: List[Tuple[int, int, float, float, float, float, float]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "track_id", "x1", "y1", "x2", "y2", "score"])
        writer.writerows(rows)


def run_tracking(args: argparse.Namespace) -> None:
    if not args.frames_dir.exists():
        raise FileNotFoundError(f"Frames directory not found: {args.frames_dir}")
    if not args.detections_dir.exists():
        raise FileNotFoundError(f"Detections directory not found: {args.detections_dir}")

    frames = image_files(args.frames_dir)
    if not frames:
        raise FileNotFoundError(f"No frames found in {args.frames_dir}")

    cfg = TrackerConfig.from_file(args.tracker_config)
    tracker = tracker_from_config(cfg)

    first_frame = cv2.imread(str(frames[0]))
    if first_frame is None:
        raise RuntimeError(f"Unable to read first frame: {frames[0]}")
    height, width = first_frame.shape[:2]

    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    args.save_tracks.parent.mkdir(parents=True, exist_ok=True)

    video_writer = cv2.VideoWriter(
        str(args.output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        cfg.frame_rate,
        (width, height),
    )

    track_rows: List[Tuple[int, int, float, float, float, float, float]] = []

    limit = args.limit if args.limit and args.limit > 0 else len(frames)
    missing_detections = 0
    for frame_idx, frame_path in enumerate(tqdm(frames[:limit], desc="Tracking")):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        det_path = args.detections_dir / f"{frame_path.stem}.txt"
        detections = load_yolo_detections(det_path, (height, width))
        if detections.size == 0:
            missing_detections += 1
        online_targets = tracker.update(detections, (height, width), (height, width))

        for track in online_targets:
            if track.state != track.Tracked:
                continue
            x1, y1, w, h = track.tlwh
            x2, y2 = x1 + w, y1 + h
            track_rows.append(
                (frame_idx, int(track.track_id), float(x1), float(y1), float(x2), float(y2), float(track.score))
            )

        vis_frame = draw_tracks(frame, online_targets)
        video_writer.write(vis_frame)

    video_writer.release()
    if track_rows:
        write_tracks_csv(track_rows, args.save_tracks)
    print(f"Saved video to {args.output_video}")
    print(f"Saved tracks to {args.save_tracks}")
    if missing_detections:
        print(f"[WARN] Frames without detections: {missing_detections}")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_tracking(args)


if __name__ == "__main__":
    main()
