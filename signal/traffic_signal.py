import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from skimage import color, exposure, transform, io
NUM_CLASSES = 43
IMG_SIZE = 48
TRAINING_PATH = 'data/traffic_sign_dataset/Final_Training/Images/'
TEST_PATH = 'data/traffic_sign_dataset/Final_Test/Images/'
BATCH_SIZE = 32
EPOCHS = 30
def correct_all_paths(img_paths):
    new_paths = []
    for path in img_paths:
        path = path.replace('\\', '/')
        new_paths.append(path)
    return new_paths
img_paths = glob.glob(os.path.join(TRAINING_PATH, '*/*.ppm'))
img_paths = correct_all_paths(img_paths)
np.random.shuffle(img_paths)
for i in range(0, 9):
    plt.subplot(330 + 1 + i)
    plt.imshow(io.imread(img_paths[i], cmap=plt.get_cmap('gray')))
#Show the plot
plt.show()
def preprocess_images(img):
    # return image in HSV format
    hsv = color.rgb2hsv(img)
    # return image after histogram equilization
    hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
    img = color.hsv2rgb(hsv)
    
    # resizing image to fixed dimension
    min_side = min(img.shape[:-1])
    center =img.shape[0] // 2, img.shape[1] // 2
    img = img[center[0] - min_side // 2: center[0] + min_side // 2,
              center[1] - min_side // 2: center[0] + min_side // 2,
              :]
    img = transform.resize(img,(IMG_SIZE, IMG_SIZE), mode = 'constant')

    return img
def get_class(img_path):
    return int(img_path.split('/')[-2])
images = []
labels = []
for img_path in img_paths:
    img = preprocess_images(io.imread(img_path))
    label = get_class(img_path)
    images.append(img)
    labels.append(label)
X = np.array(images, dtype = 'float32')
Y = np.eye(NUM_CLASSES, dtype = 'uint8')[labels]
for i in range(0, 9):
    plt.subplot(330 + 1 + i)
    plt.imshow(images[i], cmap=plt.get_cmap('gray'))
#Show the plot
plt.show()
from keras.models import Sequential
from keras.layers.core import Dropout, Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD
def build_cnn_model():
    model = Sequential() 
    model.add(Conv2D(32, (3, 3), padding ='same', input_shape = (IMG_SIZE, IMG_SIZE, 3), activation = 'relu'))
    model.add(Conv2D(32, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(64, (3, 3), padding ='same', activation = 'relu'))
    model.add(Conv2D(64, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(128, (3, 3), padding ='same', activation = 'relu'))
    model.add(Conv2D(128, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.2))
    
    model.add(Flatten())
    model.add(Dense(512, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation = 'softmax'))
    return model
    model = build_cnn_model()
lr = 0.01
sgd = SGD(lr = lr, decay = 1e-6, momentum = 0.9, nesterov = True) 
model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping
def learning_rate_scheduler(epoch):
    return lr * (0.1 ** int(epoch / 10))
import h5py as h5py
model_history = model.fit(X, Y,
                          batch_size = BATCH_SIZE,
                          epochs = EPOCHS,
                          validation_split = 0.2,
                          verbose = 1, 
                          callbacks = [LearningRateScheduler(learning_rate_scheduler),
                                      ModelCheckpoint('model.h5', save_best_only=True),
                                      EarlyStopping(monitor='val_acc', min_delta=0.00001, patience=5, \
                                                     verbose=1, mode='auto')])
import pandas as pd
test_data = pd.read_csv('data/traffic_sign_dataset/GT-final_test.csv', sep =';' )

# Loading test data
X_test = []
y_test = []
for file_name, class_id in zip(list(test_data['Filename']), list(test_data['ClassId'])):
    img_path = os.path.join('data/traffic_sign_dataset/Final_Test/Images', file_name)
    X_test.append(preprocess_images(io.imread(img_path)))
    y_test.append(class_id)
    
X_test = np.array(X_test)
y_test = np.array(y_test)

# predict and evaluate
y_predict = model.predict_classes(X_test)
accuracy = np.sum(y_predict == y_test) / np.size(y_predict)
print("Test accuracy = {}".format(accuracy))
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size =0.2, random_state = 42)

datagen = ImageDataGenerator(featurewise_center = False,
                            featurewise_std_normalization = False,
                            width_shift_range = 0.1,
                            height_shift_range = 0.1,
                            zoom_range =0.2,
                            shear_range = 0.1,
                            rotation_range = 10)
datagen.fit(X_train)
model = build_cnn_model()

lr = 0.01
sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
          optimizer=sgd,
          metrics=['accuracy'])
model_history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size= BATCH_SIZE),
                            steps_per_epoch=X_train.shape[0] / BATCH_SIZE,
                            epochs= EPOCHS,
                            validation_data=(X_val, Y_val),
                            callbacks=[LearningRateScheduler(learning_rate_scheduler),
                                       ModelCheckpoint('model_aug.h5',save_best_only=True),
                                       EarlyStopping(monitor='val_acc', min_delta=0.00001, patience=5, \
                                                     verbose=1, mode='auto')])
# predict again and re-evaluate
y_predict = model.predict_classes(X_test)
accuracy = np.sum(y_predict == y_test) / np.size(y_predict)
print("Test accuracy = {}".format(accuracy))
