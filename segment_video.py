from ultralytics import YOLO
import time

model = YOLO("yolov8n-seg.pt")
start_time = time.time()
results = model("256s.mp4", stream=True)
for result in results:
    pass
end_time = time.time()
print(f"推理时间：{(end_time - start_time):.2f}s")
