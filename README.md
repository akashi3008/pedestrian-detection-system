# Pedestrian Perception System

End-to-end pedestrian perception project covering:
1. Dataset preparation and validation
2. Pedestrian detection using YOLOv8
3. Multi-object pedestrian tracking using ByteTrack

Each folder represents a distinct stage of a real-world perception pipeline,
similar to those used in ADAS and autonomous driving systems.

## Getting started

1. **Create environment & install deps**
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Validate datasets**
   ```bash
   python -m data_tools.cli --images datasets/pedestrian/images/train --labels datasets/pedestrian/labels/train --list --remove-empty
   ```

3. **Train YOLOv8**
   ```bash
   python detection_yolov8/train.py --data detection_yolov8/dataset.yaml --epochs 50 --imgsz 640
   ```

4. **Run inference**
   ```bash
   python detection_yolov8/infer.py --weights runs/detect/pedestrian_yolov8/weights/best.pt --source images/
   ```

5. **Run tracking**
   ```bash
   python tracking_pipeline/run_tracking.py --frames-dir frames/ --detections-dir runs/detect/pedestrian_infer/labels --output-video outputs/tracks.mp4
   ```

Each stage is modular so you can swap datasets, models, or trackers as needed.
