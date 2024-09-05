from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.export(format="engine", int8=True, dynamic=True)
