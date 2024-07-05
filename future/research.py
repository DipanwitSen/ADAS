import cv2
import numpy as np
from ultralytics import YOLO
import random

# Initialize Kalman Filter
def initialize_kalman():
    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], np.float32)
    kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                       [0, 1, 0, 0],
                                       [0, 0, 5, 0],
                                       [0, 0, 0, 5]], np.float32) * 0.03
    return kalman

# Function to draw the predicted future path on the frame
def draw_future_path(frame, path):
    for (x, y) in path:
        cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)

# Function to draw parabolic and hyperbolic paths
def draw_parabolic_hyperbolic_paths(frame, a, b):
    h, k = 100, 100  # Assumed distances for demonstration purposes

    # Hyperbola with horizontal transverse axis
    for x in range(a, frame.shape[1], 10):
        try:
            value = (x - a) ** 2 * k ** 2 / h ** 2 - k ** 2
            if value >= 0:
                y1 = b + np.sqrt(value)
                y2 = b - np.sqrt(value)
                cv2.circle(frame, (int(x), int(y1)), 2, (0, 255, 0), -1)
                cv2.circle(frame, (int(x), int(y2)), 2, (0, 255, 0), -1)
        except ValueError:
            continue

    # Hyperbola with vertical transverse axis
    for y in range(b, frame.shape[0], 10):
        try:
            value = (y - b) ** 2 * h ** 2 / k ** 2 - h ** 2
            if value >= 0:
                x1 = a + np.sqrt(value)
                x2 = a - np.sqrt(value)
                cv2.circle(frame, (int(x1), int(y)), 2, (0, 255, 0), -1)
                cv2.circle(frame, (int(x2), int(y)), 2, (0, 255, 0), -1)
        except ValueError:
            continue

    # Parabola with horizontal axis
    p = 50  # Assumed distance for demonstration purposes
    for x in range(a, frame.shape[1], 10):
        try:
            value = 4 * p * (x - a)
            if value >= 0:
                y1 = b + np.sqrt(value)
                y2 = b - np.sqrt(value)
                cv2.circle(frame, (int(x), int(y1)), 2, (0, 255, 0), -1)
                cv2.circle(frame, (int(x), int(y2)), 2, (0, 255, 0), -1)
        except ValueError:
            continue

    # Parabola with vertical axis
    for y in range(b, frame.shape[0], 10):
        try:
            value = 4 * p * (y - b)
            if value >= 0:
                x1 = a + np.sqrt(value)
                x2 = a - np.sqrt(value)
                cv2.circle(frame, (int(x1), int(y)), 2, (0, 255, 0), -1)
                cv2.circle(frame, (int(x2), int(y)), 2, (0, 255, 0), -1)
        except ValueError:
            continue

# Function to perform object detection using YOLOv8
def detect_object_yolo(frame, model):
    results = model.predict(source=frame, save=False, imgsz=640, conf=0.25)
    if len(results) > 0 and len(results[0].boxes) > 0:
        box = results[0].boxes[0]
        x1, y1, x2, y2 = int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])
        cX = int((x1 + x2) / 2)
        cY = int((y1 + y2) / 2)
        return np.array([cX, cY]), (x1, y1, x2, y2)
    return None, None

# Function to add uncertainty to the future predictions
def add_uncertainty(value, uncertainty):
    return value + random.uniform(-uncertainty, uncertainty)

# Generate multiple future paths based on environmental factors
def generate_future_paths(a, b, num_paths=10):
    paths = []
    for _ in range(num_paths):
        path = []
        x, y = a, b
        for _ in range(100):  # Assume 100 future points
            x = add_uncertainty(x + 10, 5)  # 10 units ahead with 5 units uncertainty
            y = add_uncertainty(y, 5)  # 5 units uncertainty in y-axis
            # Add environmental factors here (e.g., gravity, friction)
            y += 1  # Simulating gravity effect
            path.append((x, y))
        paths.append(path)
    return paths

# Function to merge paths into a single path
def merge_paths(paths):
    merged_path = []
    for i in range(len(paths[0])):
        avg_x = np.mean([path[i][0] for path in paths])
        avg_y = np.mean([path[i][1] for path in paths])
        merged_path.append((avg_x, avg_y))
    return merged_path

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

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Initialize Kalman filter
kalman = initialize_kalman()

def process_frame(frame):
    measured, bbox = detect_object_yolo(frame, model)
    if measured is not None:
        # Draw bounding box
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Kalman filter prediction and correction
        predicted = kalman.predict()
        kalman.correct(np.array([[np.float32(measured[0])], [np.float32(measured[1])]]))

        # Get the predicted position (a, b) from the Kalman filter
        a, b = int(predicted[0, 0]), int(predicted[1, 0])

        # Generate multiple future paths
        paths = generate_future_paths(a, b)

        # Merge the paths into a single predicted path
        merged_path = merge_paths(paths)

        # Draw the merged path
        draw_future_path(frame, merged_path)

        # Draw parabolic and hyperbolic paths for visual reference
        draw_parabolic_hyperbolic_paths(frame, a, b)

    return frame

if choice in [1, 2]:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_frame(frame)
        cv2.imshow('Future Path Prediction', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
else:
    frame = process_frame(frame)
    cv2.imshow('Future Path Prediction', frame)
    cv2.waitKey(0)

cv2.destroyAllWindows()
