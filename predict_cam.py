from ultralytics import YOLO
import time, os

model = YOLO("yolov8n.engine")

start_time = time.time()
frame_count = 0

while True:
    results = model(0, stream=True, show=True, visualize=True)
    for r in results:
        frame_count += 1
        if frame_count % 10 == 0:
            end_time = time.time()
            elapsed_time = end_time - start_time
            fps = frame_count / elapsed_time
            os.system("clear")
            print(f"FPS: {fps:.2f}")
            start_time = end_time
            frame_count = 0
