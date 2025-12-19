from ultralytics import YOLO

model = YOLO("runs/detect/pedestrian_yolov8/weights/best.pt")

model.predict(
    source="images/",
    save=True,
    save_txt=True,
    save_conf=True
)
