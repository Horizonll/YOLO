from ultralytics import YOLO

model = YOLO("yolov8x.pt")
model.export(format="engine", simplify=True, imgsz=(1080, 1920))
