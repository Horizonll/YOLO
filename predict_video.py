from ultralytics import YOLO
import time

start_time = time.time()
model = YOLO("yolov8n.pt")
results = model.predict("256s.mp4", save=True, stream=True)
for result in results:
    pass
end_time = time.time()
print(f"推理时间：{(end_time - start_time):.2f}s")


"""
pt 71s
engine 67s
engine int8 48s
"""
