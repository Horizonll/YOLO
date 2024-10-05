from ultralytics import YOLO

model = YOLO("yolo11x.pt")
model.export(format="engine", dynamic=True, simplify=True)
