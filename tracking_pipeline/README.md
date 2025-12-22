# Pedestrian Tracking Pipeline

Multi-object pedestrian tracking using YOLO detections and ByteTrack.

This module demonstrates how detection outputs are integrated into
a tracking system similar to ADAS perception pipelines.

## Usage
1. Export detections from YOLOv8 with `--save-txt --save-conf` so each frame has a matching `.txt` file containing `class x y w h conf` in YOLO format.
2. Ensure frames are available as images in a directory.
3. Run tracking:
   ```bash
   python tracking_pipeline/run_tracking.py \
     --frames-dir frames/ \
     --detections-dir runs/detect/pedestrian_infer/labels \
     --tracker-config tracking_pipeline/tracker.yaml \
     --output-video outputs/tracks.mp4 \
     --save-tracks outputs/tracks.csv
   ```

The script visualizes tracks to a video and saves per-frame track data to CSV.

Notes:
- The tracker validates the config file keys and fails fast on unknown settings.
- Frames and detection txt files are matched by stem (e.g., `0001.jpg` ↔ `0001.txt`); missing detections are reported after the run.
