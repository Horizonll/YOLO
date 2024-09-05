from ultralytics import YOLO

model = YOLO("yolov8m.engine")
results = model("256s.mp4", save=True, stream=True, classes=[32])
for result in results:
    pass
