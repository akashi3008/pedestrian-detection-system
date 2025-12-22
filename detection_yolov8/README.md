# Pedestrian Detection (YOLOv8)

YOLOv8-based pedestrian detection using image datasets.

Scope:
- Training YOLOv8 on pedestrian data
- Running inference on images
- Exporting detection results for tracking

Dataset and weights are not included.

## Training
```bash
python detection_yolov8/train.py \
  --data detection_yolov8/dataset.yaml \
  --model yolov8s.pt \
  --epochs 50 \
  --imgsz 640 \
  --batch 16
```

## Inference
```bash
python detection_yolov8/infer.py \
  --weights runs/detect/pedestrian_yolov8/weights/best.pt \
  --source images/ \
  --save-txt \
  --save-conf
```

`dataset.yaml` defines a single class (`pedestrian`) to align with YOLOv8 training conventions.
