# Dataset Tools

Utilities for preparing pedestrian detection datasets in YOLO format.

Includes:
- Checking missing labels
- Removing or moving empty annotations
- Converting YOLO detections to ByteTrack format

No model training or inference is done here.

## Usage

Audit a dataset split and optionally remove empty labels:
```bash
python -m data_tools.cli \
  --images datasets/pedestrian/images/train \
  --labels datasets/pedestrian/labels/train \
  --list \
  --remove-empty
```

The tool:
- Reports images missing labels and empty label files.
- Does **not** delete images, only optionally removes empty `.txt` labels.
