from KalmanFilter import KalmanFilter
import cv2
import numpy as np
from ultralytics import YOLO


# Function to draw the predicted point on the frame
def draw_predicted_point(frame, x, y):
    cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)


# Function to draw bounding box
def draw_bounding_box(frame, box):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)


# Function to process the frame and predict future positions
def process_frame(frame, model, kf):
    results = model(frame)
    if len(results) > 0 and len(results[0].boxes) > 0:
        box = results[0].boxes[0]
        draw_bounding_box(frame, box)

        cX = int((box.xyxy[0][0] + box.xyxy[0][2]) / 2)
        cY = int((box.xyxy[0][1] + box.xyxy[0][3]) / 2)

        # Predict using Kalman Filter
        predX, predY = kf.predict(cX, cY)

        # Draw the predicted future point
        draw_predicted_point(frame, predX, predY)

    return frame


# Input setup
print("Choose input method:")
print("1: Webcam")
print("2: Video file")
print("3: Image file")
choice = int(input("Enter your choice (1, 2, or 3): "))
if choice == 1:
    cap = cv2.VideoCapture(0)  # Use 0 for webcam
elif choice == 2:
    video_path = input("Enter the path to the video file: ")
    cap = cv2.VideoCapture(video_path)
elif choice == 3:
    image_path = input("Enter the path to the image file: ")
    frame = cv2.imread(image_path)

# Load YOLO model
model = YOLO("yolov8n.pt")

# Initialize Kalman filter
kf = KalmanFilter()

if choice in [1, 2]:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_frame(frame, model, kf)
        cv2.imshow('Kalman Filter Prediction', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
else:
    frame = process_frame(frame, model, kf)
    cv2.imshow('Kalman Filter Prediction', frame)
    cv2.waitKey(0)

cv2.destroyAllWindows()
