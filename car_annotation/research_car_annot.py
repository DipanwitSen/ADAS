!pip install ultralytics
import numpy as np
import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt
from IPython.display import Image, display
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input
import cv2
from PIL import ImageFile
import shutil
import ultralytics
from flask import Flask, render_template_string, request, redirect, url_for, Response
import time
from tensorflow.keras.models import load_model
from ultralytics import YOLO

# Install ultralytics
!pip install ultralytics

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Define paths
image_dir = Path('/kaggle/input/cleaned-vehicle-dataset/Vehicle_5_classes_sample')

# Get filepaths and labels
filepaths = list(image_dir.glob(r'*/*'))
labels = [os.path.split(os.path.split(filepath)[0])[1] for filepath in filepaths]

filepaths = pd.Series(filepaths, name='Filepath').astype(str)
labels = pd.Series(labels, name='Label')

# Concatenate filepaths and labels
image_df = pd.concat([filepaths, labels], axis=1)

# Prepare DataFrame for each class
def prepare_class_df(class_dir):
    filepaths = list(class_dir.glob(r'*'))
    labels = [os.path.split(os.path.split(filepath)[0])[1] for filepath in filepaths]
    filepaths = pd.Series(filepaths, name='Filepath').astype(str)
    labels = pd.Series(labels, name='Label')
    return pd.concat([filepaths, labels], axis=1)

auto_df = prepare_class_df(Path("/kaggle/input/indian-vehicle-dataset/Vehicle_5_classes_sample/Auto"))
bus_df = prepare_class_df(Path("/kaggle/input/indian-vehicle-dataset/Vehicle_5_classes_sample/Bus"))
tempo_df = prepare_class_df(Path("/kaggle/input/indian-vehicle-dataset/Vehicle_5_classes_sample/Tempo Traveller"))
tempo_df['Label'] = 'Tempo'  # Renaming the label for consistency
tractor_df = prepare_class_df(Path("/kaggle/input/indian-vehicle-dataset/Vehicle_5_classes_sample/Tractor"))
truck_df = prepare_class_df(Path("/kaggle/input/indian-vehicle-dataset/Vehicle_5_classes_sample/Truck"))

# Shuffle the data
im_df = image_df.sample(frac=1).reset_index(drop=True)

# Show some images
fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(15, 7), subplot_kw={'xticks': [], 'yticks': []})
for i, ax in enumerate(axes.flat):
    img = plt.imread(im_df.Filepath[i])
    ax.imshow(img)
    ax.set_title(im_df.Label[i])
plt.tight_layout()
plt.show()

# Split data into training and testing sets
train_df, test_df = train_test_split(image_df, train_size=0.8, shuffle=True, random_state=1)

# Define and compile the model
model = Sequential([
    Conv2D(32, (3, 3), input_shape=(224, 224, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(5, activation='softmax')
])
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.save('custom_cnn_model.h5')

# Data generators
train_generator = ImageDataGenerator(
    rescale=1./255, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True, zoom_range=0.2, validation_split=0.1
)
test_generator = ImageDataGenerator(
    rescale=1./255, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True, zoom_range=0.2
)

train_images = train_generator.flow_from_dataframe(
    dataframe=train_df, x_col='Filepath', y_col='Label', target_size=(224, 224), color_mode='rgb', class_mode='categorical', batch_size=32, shuffle=True, seed=42, subset='training'
)
val_images = train_generator.flow_from_dataframe(
    dataframe=train_df, x_col='Filepath', y_col='Label', target_size=(224, 224), color_mode='rgb', class_mode='categorical', batch_size=32, shuffle=True, seed=42, subset='validation'
)
test_images = test_generator.flow_from_dataframe(
    dataframe=test_df, x_col='Filepath', y_col='Label', target_size=(224, 224), color_mode='rgb', class_mode='categorical', batch_size=32, shuffle=False
)

# Load pre-trained VGG16 model
base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(512, activation='relu')(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
predictions = tf.keras.layers.Dense(5, activation='softmax')(x)
model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

# Fine-tune top layers
for layer in base_model.layers:
    layer.trainable = False

# Compile and train the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, epochs=60, validation_data=val_images)
results = model.evaluate(test_images, verbose=0)

print("    Test Loss:", results[0])
print("Test Accuracy:", results[1] * 100, "%")

# Save valid files to a directory
source_dir = '/kaggle/input/indian-vehicle-dataset/Vehicle_5_classes_sample/'
destination_dir = '/kaggle/working/valid_images/'

if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

valid_extensions = ['.jpg', '.png']

for root, dirs, files in os.walk(source_dir):
    for file in files:
        if os.path.splitext(file)[1] in valid_extensions:
            source_file = os.path.join(root, file)
            destination_file = os.path.join(destination_dir, file)
            shutil.copy(source_file, destination_file)

print("Valid files have been copied to the destination directory:", destination_dir)

# Flask app for serving the model
app = Flask(__name__)
classifier_model = load_model('custom_cnn_model.h5')
yolo_model = YOLO("yolo-Weights/yolov8n.pt")

class_labels = ["Auto", "Bus", "Tempo", "Tractor", "Truck"]
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

confThreshold = 0.5
nmsThreshold = 0.4
objHeight = 1.0
objWidth = 0.5
focalLength = 480

def calculate_time(distance, speed):
    return distance / speed if speed != 0 else float('inf')

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
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                w, h = x2 - x1, y2 - y1
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                conf = box.conf[0]
                cls = int(box.cls[0])

                if conf >= confThreshold:
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                    cv2.circle(img, (cx, cy), 2, (0, 0, 255), 2)
                    cv2.putText(img, f'{classNames[cls]}: {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    objPixelHeight = abs(y2 - y1)
                    distance = (objHeight * focalLength) / objPixelHeight
                    currentTime = time.time()
                    deltaTime = currentTime - prevTime
                    if deltaTime > 0:
                        speed = (distance - prevDistance) / deltaTime
                    timeToCollision = calculate_time(distance, speed)
                    
                    prevDistance = distance
                    prevTime = currentTime

                    cv2.putText(img, f'Speed: {speed:.2f} m/s', (10, img.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(img, f'Time to Collision: {timeToCollision:.2f} s', (10, img.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template_string('''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Vehicle Classification and Detection</title>
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css">
        <style>
            body {
                margin: 0;
                padding: 0;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                height: 100vh;
                background-color: #f8f9fa;
            }
            .container {
                max-width: 600px;
                padding: 20px;
                background-color: white;
                border-radius: 8px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            }
            .btn-primary {
                background-color: #007bff;
                border: none;
            }
            .btn-primary:hover {
                background-color: #0056b3;
            }
            .video-stream {
                margin-top: 20px;
                max-width: 100%;
                height: auto;
            }
        </style>
    </head>
    <body>
        <div class="container text-center">
            <h1 class="mb-4">Vehicle Classification and Detection</h1>
            <form action="/classify" method="post" enctype="multipart/form-data">
                <div class="form-group">
                    <label for="image">Upload an image for classification:</label>
                    <input type="file" class="form-control-file" id="image" name="image" required>
                </div>
                <button type="submit" class="btn btn-primary">Classify Image</button>
            </form>
            <div class="video-stream mt-5">
                <h2>Live Video Stream</h2>
                <img src="{{ url_for('video_feed') }}" alt="Video Stream">
            </div>
        </div>
    </body>
    </html>
    ''')

@app.route('/classify', methods=['POST'])
def classify_image():
    image_file = request.files['image']
    if image_file:
        image_path = f'static/uploads/{image_file.filename}'
        image_file.save(image_path)
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        predictions = classifier_model.predict(img_array)
        predicted_class = class_labels[np.argmax(predictions[0])]
        return render_template_string('''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Classification Result</title>
            <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css">
        </head>
        <body>
            <div class="container text-center">
                <h1 class="mt-5">Classification Result</h1>
                <img src="{{ url_for('static', filename='uploads/' + filename) }}" alt="Uploaded Image" class="img-fluid mt-3">
                <h2 class="mt-4">Predicted Class: {{ predicted_class }}</h2>
                <a href="/" class="btn btn-primary mt-4">Go Back</a>
            </div>
        </body>
        </html>
        ''', filename=image_file.filename, predicted_class=predicted_class)
    return redirect(url_for('index'))

@app.route('/video_feed')
def video_feed():
    return Response(stream_video(0), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
