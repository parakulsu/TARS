import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import os
import cv2
import numpy as np

width, height = 550, 550
images = []
labels = []
channels = 3

data_dir = r'C:\Users\User\Documents\Bitbucket\TARS2\img'

for label in os.listdir(data_dir):
    label_path = os.path.join(data_dir, label)
    for filename in os.listdir(label_path):
        img_path = os.path.join(label_path, filename)
        img = cv2.imread(img_path)

        h, w, _ = img.shape
        left = (w - width) // 2
        top = (h - height) // 2
        right = (w + width) // 2
        bottom = (h + height) // 2

        img = img[top:bottom, left:right]

        img = cv2.resize(img, (width, height))
        images.append(img)
        labels.append(label)

images = np.array(images)
labels = np.array(labels)

label_dict = {label: idx for idx, label in enumerate(np.unique(labels))}
labels = np.vectorize(label_dict.get)(labels)
labels = to_categorical(labels, num_classes=3)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(width, height, channels)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(images, labels, epochs=20)

model.save('model.h5')
