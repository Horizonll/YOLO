import cv2
from ultralytics import YOLO
import time

model = YOLO("yolo11x.onnx")
video_path = "256s.mp4"
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
prev_time = time.time()
start_time = time.time()

while cap.isOpened():
    success, frame = cap.read()
    if success:
        current_time = time.time()
        elapsed_time = current_time - prev_time
        prev_time = current_time
        fps = 1 / elapsed_time
        results = model(frame)
        annotated_frame = results[0].plot()
        cv2.putText(
            annotated_frame,
            f"FPS: {int(fps)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.imshow("YOLOv8 Inference", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break
end_time = time.time()
cap.release()
cv2.destroyAllWindows()
