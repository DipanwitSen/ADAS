!pip install Opencv-python
!pip install ultralytics
!pip install PIL
import cv2
import numpy as np
import time
from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image

# Load YOLO model
model = YOLO("yolov8x.pt")

# Function to calculate the distance between two points (Euclidean distance)
def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# Function to calculate the time to collision (TTC)
def calculate_ttc(prev_distance, curr_distance, time_elapsed):
    if prev_distance - curr_distance <= 0:
        return float('inf')  # No collision if object is moving away or staying at the same distance
    return curr_distance / ((prev_distance - curr_distance) / time_elapsed)

def process_frame(frame, prev_frame_time, prev_distance, save_path=None):
    results = model.predict(source=frame, conf=0.2, iou=0.5)
    frame = results[0].plot()
    current_frame_time = time.time()

    if prev_frame_time is not None:
        time_elapsed = current_frame_time - prev_frame_time
        for result in results:
            for box in result.boxes:
                cords = box.xyxy[0].tolist()
                class_id = result.names[box.cls[0].item()]
                conf = round(box.conf[0].item(), 2)
                
                curr_distance = calculate_distance([cords[0], cords[1]], [cords[2], cords[3]])

                if prev_distance is not None:
                    ttc = calculate_ttc(prev_distance, curr_distance, time_elapsed)
                    if ttc < 5:
                        cv2.putText(frame, f"Warning! Time to Collision: {ttc:.2f} seconds", (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                
                prev_distance = curr_distance

                print("Object type:", class_id)
                print("Coordinates:", cords)
                print("Probability:", conf)
                print("---")
                
    prev_frame_time = current_frame_time

    if save_path:
        cv2.imwrite(save_path, frame)

    return frame, prev_frame_time, prev_distance

# User choice for input type
choice = input("Select input type (image/video/webcam): ").lower()

if choice == "image":
    image_path = input("Enter the path to the image: ")
    frame = cv2.imread(image_path)
    processed_frame, _, _ = process_frame(frame, None, None)
    plt.imshow(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
    plt.show()

elif choice == "video":
    video_path = input("Enter the path to the video: ")
    cap = cv2.VideoCapture(video_path)
    out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (int(cap.get(3)), int(cap.get(4))))
    
    prev_frame_time = None
    prev_distance = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame, prev_frame_time, prev_distance = process_frame(frame, prev_frame_time, prev_distance)
        out.write(processed_frame)

    cap.release()
    out.release()

elif choice == "webcam":
    cap = cv2.VideoCapture(0)
    prev_frame_time = None
    prev_distance = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame, prev_frame_time, prev_distance = process_frame(frame, prev_frame_time, prev_distance)
        
        cv2.imshow('Webcam Feed', processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

else:
    print("Invalid choice! Please select 'image', 'video', or 'webcam'.")
