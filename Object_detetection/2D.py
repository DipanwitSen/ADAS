import cv2
import math
import time
import os
from ultralytics import YOLO

# Parameters
objHeight = 1.0  # Actual height of the object in meters
objWidth = 0.5  # Actual width of the object in meters
focalLength = 480  # Focal length of the camera in pixels
classNames = ["person", "bicycle", "car", "motorbike", "bus", "truck"]
confThreshold = 0.5
nmsThreshold = 0.4

# Initialize YOLO model
model = YOLO("yolo-Weights/yolov8x.pt")

# Output directory
output_dir = "C:\\Users\\KIIT\\Desktop\\intel\\webapp\\output"
os.makedirs(output_dir, exist_ok=True)

# Initialize global variables for distance, time, and speed
prevDistance = 0.0
prevTime = time.time()
speed = 0.0

def calculate_time(distance, speed):
    return distance / speed if speed != 0 else float('inf')

def process_frame(img):
    global prevDistance, prevTime, speed
    results = model(img, stream=True)
    boxes, confidences, classIDs = [], [], []

    for r in results:
        detections = r.boxes
        for detection in detections:
            score = detection.conf[0].item()  # Convert tensor to float
            if score > confThreshold:
                x1, y1, x2, y2 = map(int, detection.xyxy[0])
                confidence = round(score, 2)
                cls = int(detection.cls[0].item())  # Convert tensor to int

                if cls < len(classNames):
                    if classNames[cls] in classNames:
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                        cv2.putText(img, classNames[cls], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                        boxes.append([x1, y1, x2, y2])
                        confidences.append(confidence)
                        classIDs.append(cls)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0] if isinstance(i, (list, tuple)) else i
        x1, y1, x2, y2 = boxes[i]
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

        objectWidth = x2 - x1
        objectHeight = y2 - y1
        distance = (objWidth * focalLength) / objectWidth
        distance *= objHeight / objectHeight

        currTime = time.time()
        timeDiff = currTime - prevTime
        if timeDiff > 0:  # Ensure timeDiff is not zero
            speed = round((distance - prevDistance) / timeDiff, 2)
            time_to_collision = calculate_time(distance, speed)

            if time_to_collision < 10:
                cv2.putText(img, "Warning: Object is too close!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            prevDistance = distance
            prevTime = currTime

            cv2.putText(img, f"Speed: {speed} m/s", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, f"Distance: {distance} m", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, f"Time to Collision: {time_to_collision} s", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return img

def process_webcam():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    
    fourcc_avi = cv2.VideoWriter_fourcc(*'XVID')
    fourcc_mp4 = cv2.VideoWriter_fourcc(*'mp4v')
    output_path_avi = os.path.join(output_dir, "webcam_output.avi")
    output_path_mp4 = os.path.join(output_dir, "webcam_output.mp4")
    out_avi = cv2.VideoWriter(output_path_avi, fourcc_avi, 20.0, (640, 480))
    out_mp4 = cv2.VideoWriter(output_path_mp4, fourcc_mp4, 20.0, (640, 480))
    
    while True:
        success, img = cap.read()
        if not success:
            break
        img = process_frame(img)
        out_avi.write(img)
        out_mp4.write(img)
        cv2.imshow('Webcam', img)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    out_avi.release()
    out_mp4.release()
    cv2.destroyAllWindows()
    print(f"Processed webcam video saved to: {output_path_avi} and {output_path_mp4}")

def process_image(image_path):
    img = cv2.imread(image_path)
    if img is not None:
        img = process_frame(img)
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(output_path, img)
        print(f"Processed image saved to: {output_path}")
    else:
        print("Failed to load image.")

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fourcc_avi = cv2.VideoWriter_fourcc(*'XVID')
    fourcc_mp4 = cv2.VideoWriter_fourcc(*'mp4v')
    output_path_avi = os.path.join(output_dir, os.path.basename(video_path).replace('.mp4', '_output.avi'))
    output_path_mp4 = os.path.join(output_dir, os.path.basename(video_path).replace('.avi', '_output.mp4'))
    out_avi = cv2.VideoWriter(output_path_avi, fourcc_avi, 20.0, (640, 480))
    out_mp4 = cv2.VideoWriter(output_path_mp4, fourcc_mp4, 20.0, (640, 480))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = process_frame(frame)
        out_avi.write(frame)
        out_mp4.write(frame)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    out_avi.release()
    out_mp4.release()
    cv2.destroyAllWindows()
    print(f"Processed video saved to: {output_path_avi} and {output_path_mp4}")

def main():
    choice = input("Choose input type (webcam, image, video): ").strip().lower()
    if choice == "webcam":
        process_webcam()
    elif choice == "image":
        image_path = input("Enter image path: ").strip()
        process_image(image_path)
    elif choice == "video":
        video_path = input("Enter video path: ").strip()
        process_video(video_path)
    else:
        print("Invalid choice. Please choose either 'webcam', 'image', or 'video'.")

if __name__ == "__main__":
    prevDistance = 0.0  # Initial distance
    prevTime = time.time()  # Initial time
    speed = 0.0  # Speed of the object in meters per second
    main()
