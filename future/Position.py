%pip install Opencv-python
%pip install numpy
%pip install ultralytics
import cv2
import numpy as np
from ultralytics import YOLO

class KalmanFilter:
    def __init__(self, dt, u_x, u_y, std_acc, x_std_meas, y_std_meas):
        # Define sampling time
        self.dt = dt
        
        # Define the control input variables
        self.u = np.matrix([[u_x],[u_y]])
        
        # Initial State
        self.x = np.matrix([[0], [0], [0], [0]])
        
        # Define the State Transition Matrix A
        self.A = np.matrix([[1, 0, self.dt, 0],
                            [0, 1, 0, self.dt],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
        
        # Define the Control Input Matrix B
        self.B = np.matrix([[(self.dt**2)/2, 0],
                            [0, (self.dt**2)/2],
                            [self.dt, 0],
                            [0, self.dt]])
        
        # Define Measurement Mapping Matrix
        self.H = np.matrix([[1, 0, 0, 0],
                            [0, 1, 0, 0]])
        
        # Initial Process Noise Covariance
        self.Q = np.matrix([[(self.dt**4)/4, 0, (self.dt**3)/2, 0],
                            [0, (self.dt**4)/4, 0, (self.dt**3)/2],
                            [(self.dt**3)/2, 0, self.dt**2, 0],
                            [0, (self.dt**3)/2, 0, self.dt**2]]) * std_acc**2
        
        # Initial Measurement Noise Covariance
        self.R = np.matrix([[x_std_meas**2, 0],
                            [0, y_std_meas**2]])
        
        # Initial Covariance Matrix
        self.P = np.eye(self.A.shape[1])
    
    def predict(self):
        # Update time state
        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)
        
        # Update Process Covariance Matrix
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x[0:2]
    
    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.H.shape[1])
        self.P = (I - np.dot(K, self.H)) * self.P

# Function to draw the tracked points on the frame
def draw_prediction(frame, pt):
    x, y = int(pt[0]), int(pt[1])
    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
    cv2.putText(frame, f'({x}, {y})', (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

# Function to perform object detection using YOLOv8
def detect_object_yolo(frame, model):
    results = model.predict(source=frame, save=False, imgsz=640, conf=0.25)
    if len(results) > 0 and len(results[0].boxes) > 0:
        box = results[0].boxes[0]
        cX = int((box.xyxy[0][0] + box.xyxy[0][2]) / 2)
        cY = int((box.xyxy[0][1] + box.xyxy[0][3]) / 2)
        return np.array([[cX], [cY]])
    return None

# Video input setup
print("Choose input method:")
print("1: Webcam")
print("2: Video file")
print("3: Image file")
choice = int(input("Enter your choice (1, 2, or 3): "))

if choice == 1:
    cap = cv2.VideoCapture(0)
elif choice == 2:
    video_path = input("Enter the path to the video file: ")
    cap = cv2.VideoCapture(video_path)
elif choice == 3:
    image_path = input("Enter the path to the image file: ")
    frame = cv2.imread(image_path)
    cap = None
else:
    print("Invalid choice")
    exit()

# Kalman Filter parameters
dt = 1/30  # Assuming 30fps
u_x = 1  # Acceleration in x-direction
u_y = 1  # Acceleration in y-direction
std_acc = 1  # Process noise magnitude
x_std_meas = 0.1  # Measurement noise in x-direction
y_std_meas = 0.1  # Measurement noise in y-direction

kf = KalmanFilter(dt, u_x, u_y, std_acc, x_std_meas, y_std_meas)

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

if cap:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        measured = detect_object_yolo(frame, model)

        if measured is not None:
            # Kalman prediction
            kf.predict()
            
            # Kalman update
            kf.update(measured)

            # Get the current state
            predicted = kf.x[0:2]

            # Draw the detection
            draw_prediction(frame, measured)
            draw_prediction(frame, predicted)
        
        cv2.imshow('Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
else:
    measured = detect_object_yolo(frame, model)
    
    if measured is not None:
        # Kalman prediction
        kf.predict()
        
        # Kalman update
        kf.update(measured)

        # Get the current state
        predicted = kf.x[0:2]

        # Draw the detection
        draw_prediction(frame, measured)
        draw_prediction(frame, predicted)
    
    cv2.imshow('Tracking', frame)
    cv2.waitKey(0)

cv2.destroyAllWindows()

