from ultralytics import YOLO

model = YOLO("yolov8s.pt")

model.train(
    data="dataset.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    name="pedestrian_yolov8"
)
