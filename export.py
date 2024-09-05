from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.export(format="engine", simplify=True, int8=True)
