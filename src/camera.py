import cv2
from ultralytics import YOLO
import time

# Load model (ncnn model gives like 10 more frames XD)
model = YOLO("yolo11n_ncnn_model/") # if this breaks use "yolo11n.pt" (nano)

# Open USB camera
cap = cv2.VideoCapture("/dev/video0")

if not cap.isOpened():
    print("Camera failed to open")
    exit()

# Optional resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

fps = 0
tStart = time.time()

frame_skip = 2
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    count += 1
    if count % frame_skip != 0:
        continue
    # Run inference
    results = model(frame, imgsz=480, conf=0.4, verbose=False)

    # Draw detections
    annotated = results[0].plot()

    # FPS calculation
    deltaT = time.time() - tStart
    tStart = time.time()
    fps = 0.9 * fps + 0.1 / deltaT

    cv2.putText(
        annotated,
        f"FPS: {round(fps,1)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    cv2.imshow("YOLO USB Detection", annotated)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()