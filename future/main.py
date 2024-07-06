from KalmanFilter import KalmanFilter
import cv2
import numpy as np

# Function to draw the predicted point on the frame
def draw_predicted_point(frame, x, y):
    cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

def process_frame(frame, kf):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    ret, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # Predict using Kalman Filter
            predX, predY = kf.predict(cX, cY)

            # Draw the predicted future point
            draw_predicted_point(frame, predX, predY)

            # Draw the current detected point
            cv2.circle(frame, (cX, cY), 5, (255, 0, 0), -1)

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

# Initialize Kalman filter
kf = KalmanFilter()

if choice in [1, 2]:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_frame(frame, kf)
        cv2.imshow('Kalman Filter Prediction', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
else:
    frame = process_frame(frame, kf)
    cv2.imshow('Kalman Filter Prediction', frame)
    cv2.waitKey(0)

cv2.destroyAllWindows()
