from ultralytics import YOLO
import time

start_time = time.time()
model = YOLO("yolo11n.pt")
results = model.predict("256s.mp4", save=True, stream=True)
for result in results:
    pass
end_time = time.time()
print(f"推理时间：{(end_time - start_time):.2f}s")


"""
pt 71s 110s
engine 67s 82s
engine int8 48s
"""
