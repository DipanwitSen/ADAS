import cv2
import numpy as np
import time
from ultralytics import YOLO

# Function to draw the predicted future path on the frame
def draw_future_path(frame, a, b):
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
        cX = int((box.xyxy[0][0] + box.xyxy[0][2]) / 2)
        cY = int((box.xyxy[0][1] + box.xyxy[0][3]) / 2)
        return np.array([cX, cY])
    return None

# Lagrange Interpolation function
def lagrange_interpolation(x, y, x_pred):
    n = len(x)
    y_pred = 0
    for i in range(n):
        L = 1
        for j in range(n):
            if i != j:
                denominator = (x[i] - x[j])
                if denominator != 0:
                    L *= (x_pred - x[j]) / denominator
        y_pred += y[i] * L
    return y_pred

# Newton-Raphson method
def newton_raphson(f, df, x0, tol=1e-6, max_iter=100):
    x = x0
    for i in range(max_iter):
        f_value = f(x)
        df_value = df(x)
        if df_value == 0:
            break
        x_new = x - f_value / df_value
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    return x

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

# Lists to store detected positions
x_positions = []
y_positions = []

def process_frame(frame):
    measured = detect_object_yolo(frame, model)
    if measured is not None:
        # Get the center (a, b) of the detected object
        a, b = measured
        x_positions.append(a)
        y_positions.append(b)
        # Draw the detected object center
        cv2.circle(frame, (a, b), 5, (0, 0, 255), -1)

        # Lagrange interpolation for future points
        if len(x_positions) > 2:
            x_pred = a + 10  # Predict next point
            y_pred = lagrange_interpolation(x_positions, y_positions, x_pred)
            # Draw the Lagrange interpolated point
            cv2.circle(frame, (int(x_pred), int(y_pred)), 5, (255, 0, 0), -1)

            # Refine prediction with Newton-Raphson method
            f = lambda x: lagrange_interpolation(x_positions, y_positions, x) - y_pred
            df = lambda x: (f(x + 1e-5) - f(x)) / 1e-5  # Numerical derivative
            refined_x_pred = newton_raphson(f, df, x_pred)
            refined_y_pred = lagrange_interpolation(x_positions, y_positions, refined_x_pred)
            # Draw the Newton-Raphson refined point
            cv2.circle(frame, (int(refined_x_pred), int(refined_y_pred)), 5, (0, 255, 255), -1)

        # Draw the predicted future path using all equations
        draw_future_path(frame, a, b)

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
