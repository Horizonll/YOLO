from ultralytics import YOLO

model = YOLO("yolo11n.onnx")
results = model(0, stream=True, show=True)
for result in results:
    pass
