import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list al
# files under the input directory

import os
for dirname, _,filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
from pathlib import Path
import os.path
import matplotlib.pyplot as plt
from IPython.display import Image, display
import matplotlib.cm as cm
from sklearn.model_selection import train_test_split
import tensorflow as tf 
image_dir = Path('/kaggle/input/cleaned-vehicle-dataset/Vehicle_5_classes_sample')

# Get filepaths and labels
filepaths = list(image_dir.glob(r'*/*'))
labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))

filepaths = pd.Series(filepaths, name='Filepath').astype(str)
labels = pd.Series(labels, name='Label')

# Concatenate filepaths and labels
image_df = pd.concat([filepaths, labels], axis=1)
image_df
lst = []
for l in image_df['Label'].unique():
    lst.append(image_df[image_df['Label'] == l])
# Concatenate the DataFrames
image_df = pd.concat(lst)
image_df
auto_dir = Path("/kaggle/input/indian-vehicle-dataset/Vehicle_5_classes_sample/Auto")
#  Get filepaths and labels
filepaths = list(auto_dir.glob(r'*'))
labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))
print(labels)
filepaths = pd.Series(filepaths, name='Filepath').astype(str)
labels = pd.Series(labels, name='Label')

# Concatenate filepaths and labels
auto_df = pd.concat([filepaths, labels], axis=1)
auto_df
bus_dir = Path("/kaggle/input/indian-vehicle-dataset/Vehicle_5_classes_sample/Bus")
# Get filepaths and labels
filepaths = list(bus_dir.glob(r'*'))
labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))
print(labels)
filepaths = pd.Series(filepaths, name='Filepath').astype(str)
labels = pd.Series(labels, name='Label')

# Concatenate filepaths and labels
bus_df = pd.concat([filepaths, labels], axis=1)
bus_df
 tempo_dir = Path("/kaggle/input/indian-vehicle-dataset/Vehicle_5_classes_sample/Tempo Traveller")
# Get filepaths and labels
filepaths = list(tempo_dir.glob(r'*'))
labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))
print(labels)
filepaths = pd.Series(filepaths, name='Filepath').astype(str)
labels = pd.Series(labels, name='Label')

# Concatenate filepaths and labels
tempo_df = pd.concat([filepaths, labels], axis=1)
tempo_df['Label'] = 'Tempo'
tempo_df
tractor_dir = Path("/kaggle/input/indian-vehicle-dataset/Vehicle_5_classes_sample/Tractor")
# Get filepaths and labels
filepaths = list(tractor_dir.glob(r'*'))
labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))
print(labels)
filepaths = pd.Series(filepaths, name='Filepath').astype(str)
labels = pd.Series(labels, name='Label')

# Concatenate filepaths and labels
tractor_df = pd.concat([filepaths, labels], axis=1)
tractor_df
truck_dir = Path("/kaggle/input/indian-vehicle-dataset/Vehicle_5_classes_sample/Truck")
# Get filepaths and labels
filepaths = list(truck_dir.glob(r'*'))
labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))
print(labels)
filepaths = pd.Series(filepaths, name='Filepath').astype(str)
labels = pd.Series(labels, name='Label')

# Concatenate filepaths and labels
truck_df = pd.concat([filepaths, labels], axis=1)
truck_df
# Shuffling the data
im_df = image_df.sample(frac=1).reset_index(drop = True)

# Show the result
im_df.head()
# image_df.head()
fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(15, 7),
                        subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(axes.flat):
    img = plt.imread(im_df.Filepath[i])
    ax.imshow(img)
    ax.set_title(im_df.Label[i])
plt.tight_layout()
plt.show()
train_df, test_df = train_test_split(image_df, train_size=0.8, shuffle=True, random_state=1)
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define your model
model = Sequential()

# Input layer
model.add(Conv2D(32, (3, 3), input_shape=(224, 224, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Convolutional layers
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#Flatten the output for fully connected layers
model.add(Flatten())

# Fully connected layers
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

#Output layer
model.add(Dense(5, activation='softmax'))

#Compile the model
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

#Save the model
model.save('custom_cnn_model.h5')
from keras.applications.vgg16 import preprocess_input
import cv2
# Define a custom preprocessing function
def custom_preprocess_input(image):
    # Resize the image to a specific size (e.g., 224x224)
    image = cv2.resize(image, (224, 224))
    # Apply mean subtraction
    image = image - [123.68, 116.78, 103.94]  # Mean values for BGR channels
    # Perform any other desired preprocessing steps
    
    return image

train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,            # Rescale pixel values to [0, 1]
    rotation_range=20,         # Random rotation
    width_shift_range=0.2,    # Random horizontal shift
    height_shift_range=0.2,   # Random vertical shift
    horizontal_flip=True,     # Random horizontal flip
    zoom_range=0.2,  # Random zoom
    validation_split = 0.1
)

test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,            # Rescale pixel values to [0, 1]
    rotation_range=20,         # Random rotation
    width_shift_range=0.2,    # Random horizontal shift
    height_shift_range=0.2,   # Random vertical shift
    horizontal_flip=True,     # Random horizontal flip
    zoom_range=0.2            # Random zoom
)
train_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    seed=42,
    subset='training'
)

val_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    seed=42,
    subset='validation'
)
test_images = test_generator.flow_from_dataframe(
    dataframe=test_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=False
)
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
#  Train the model
# model.fit(train_images, validation_data=val_images, epochs=20)
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16

# Load pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom classification layers
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
predictions = layers.Dense(5, activation='softmax')(x)  # num_classes is the number of vehicle classes

model = models.Model(inputs=base_model.input, outputs=predictions)

# Fine-tune some top layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model using your data generator (train_images)
model.fit(train_images, epochs=50, validation_data=val_images)
import os
import shutil

# Define the source directory containing the files you want to copy
source_dir = '/kaggle/input/indian-vehicle-dataset/Vehicle_5_classes_sample/'

# Define the destination directory where you have write access
destination_dir = '/kaggle/working/valid_images/'

# Ensure the destination directory exists, and create it if necessary
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# List of valid file extensions (adjust to your needs)
valid_extensions = ['.jpg', '.png']

# Iterate through files in the source directory
for root, dirs, files in os.walk(source_dir):
    for file in files:
        if os.path.splitext(file)[1] in valid_extensions:
            source_file = os.path.join(root, file)
            destination_file = os.path.join(destination_dir, file)
            shutil.copy(source_file, destination_file)  # Corrected indentation

print("Valid files have been copied to the destination directory:", destination_dir)
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16

# Load pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom classification layers
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dense(128, activation='relu')(x)
predictions = layers.Dense(5, activation='softmax')(x)  # num_classes is the number of vehicle classes

model = models.Model(inputs=base_model.input, outputs=predictions)

# Fine-tune some top layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model using your data generator (train_images)
model.fit(train_images, epochs=50, validation_data=val_images)
results = model.evaluate(test_images, verbose=0)

print("    Test Loss:",(results[0]))
print("Test Accuracy:",(results[1] * 100),"%")
import cv2
import numpy as np

# Load and preprocess the image
image_path = '/kaggle/input/cleaned-vehicle-dataset/Vehicle_5_classes_sample/Auto/Datacluster Auto (11).jpg'
image = cv2.imread(image_path)
image = cv2.resize(image, (224, 224))  # Resize the image to match your model's input size
image = image / 255.0  # Normalize the pixel values

# Make predictions
predictions = model.predict(np.expand_dims(image, axis=0))  # Add a batch dimension
print(predictions)
# The 'predictions' variable now contains the predicted probabilities for each class.
# You can use np.argmax(predictions) to find the index of the class with the highest probability.

# For example, if you want to find the class with the highest probability:
predicted_class = np.argmax(predictions)

# You might also want to decode the class index to get the class label based on your dataset.
class_labels = ["Auto", "Bus", "Tempo", "Tractor", "Truck"]
predicted_label = class_labels[predicted_class]

# Print the predicted class label
print("Predicted Class:", predicted_label)
%pip install ultralytics
import ultralytics
ultralytics.checks()
from flask import Flask, render_template, request, redirect, url_for, Response
import cv2
import numpy as np
import time
import os
from tensorflow.keras.models import load_model
from ultralytics import YOLO

app = Flask(__name__)

# Load pre-trained models
classifier_model = load_model('custom_cnn_model.h5')
yolo_model = YOLO("yolo-Weights/yolov8n.pt")

# Class labels
class_labels = ["Auto", "Bus", "Tempo", "Tractor", "Truck"]
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

confThreshold = 0.5
nmsThreshold = 0.4
objHeight = 1.0  # Actual height of the object in meters
objWidth = 0.5  # Actual width of the object in meters
focalLength = 480  # Focal length of the camera in pixels

def calculate_time(distance, speed):
    if speed == 0:
        return float('inf')
    else:
        return distance / speed

def create_video_writer(video_cap, output_filename):
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    writer = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
    return writer

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    input_type = request.form.get('input_type')
    if input_type == 'webcam':
        return redirect(url_for('webcam'))
    elif input_type == 'video':
        video_file = request.files['video_file']
        video_path = 'static/' + video_file.filename
        video_file.save(video_path)
        return redirect(url_for('video', video_path=video_path))
    elif input_type == 'image':
        image_file = request.files['image_file']
        image_path = 'static/' + image_file.filename
        image_file.save(image_path)
        return redirect(url_for('image', image_path=image_path))
    else:
        return "Invalid input type", 400

@app.route('/webcam')
def webcam():
    return Response(stream_video(0), mimetype='multipart/x-mixed-replace; boundary=frame')

def stream_video(source):
    cap = cv2.VideoCapture(source)
    prevDistance = 0.0
    prevTime = time.time()
    speed = 0.0

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break

        results = yolo_model(img, stream=True)

        for r in results:
            detections = r.boxes
            for detection in detections:
                score = detection.conf[0]
                if score > confThreshold:
                    x1, y1, x2, y2 = detection.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Draw bounding box
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                    # Calculate distance
                    objectWidth = x2 - x1
                    objectHeight = y2 - y1
                    distance = (objWidth * focalLength) / objectWidth
                    distance = distance * (objHeight / objectHeight)

                    # Calculate speed
                    currTime = time.time()
                    timeDiff = currTime - prevTime
                    speed = (distance - prevDistance) / timeDiff
                    speed = round(speed, 2)

                    # Calculate time to collision
                    time_to_collision = calculate_time(distance, speed)

                    if time_to_collision < 10:
                        cv2.putText(img, "Warning: Object is too close!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    prevDistance = distance
                    prevTime = currTime

                    # Display details
                    cv2.putText(img, f"Speed: {speed} m/s", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, f"Distance: {distance} m", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, f"Time to Collision: {time_to_collision} s", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video/<video_path>')
def video(video_path):
    return Response(stream_video(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/image/<image_path>')
def image(image_path):
    img = cv2.imread(image_path)
    results = yolo_model(img, stream=True)

    for r in results:
        detections = r.boxes
        for detection in detections:
            score = detection.conf[0]
            if score > confThreshold:
                x1, y1, x2, y2 = detection.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

    output_path = 'static/output.jpg'
    cv2.imwrite(output_path, img)
    return redirect(url_for('static', filename='output.jpg'))

if __name__ == '__main__':
    app.run(debug=True)

      
