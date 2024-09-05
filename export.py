from ultralytics import YOLO

model = YOLO("yolov8m.pt")
model.export(format="engine", simplify=True, dynamic=True, half=True)
