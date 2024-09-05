from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")
model.export(format="engine", half=True)
