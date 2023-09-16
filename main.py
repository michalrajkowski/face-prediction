import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import dlib
import time
import random
import pickle
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

# Prepare the output folder
OUTPUT_IMG_DIR = 'output images'
INPUT_IMG_DIR = 'input images'

for f in os.listdir(OUTPUT_IMG_DIR):
    os.remove(os.path.join(OUTPUT_IMG_DIR, f))

CATEGORIES = ["Female", "Male"]
IMG_SIZE = 50

# Load model
test_X = []
test_X = np.array(test_X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

model = Sequential()
model.add(Conv2D(128, (4,4), input_shape = test_X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128, (4,4)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128, (4,4)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])

model_path = './model/aug_model'
model.load_weights(model_path)


def display_prediction(img, label, chance, pred_cat):
  plt.imshow(img)
  prd_gender = "Predicted: " + CATEGORIES[int(pred_cat)]
  prd_percent = "with " + str(chance) +" prob."
  plt.title( "{}\n{}".format( prd_gender, prd_percent))
  save_path = os.path.join(OUTPUT_IMG_DIR, str(time.time()) + '.png')
  plt.savefig(save_path)

def load_test_images():
    path = INPUT_IMG_DIR
    for img in os.listdir(path):
        try:
            color_image = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
            RGB_img = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            class_num = CATEGORIES[0]
            special_data.append([new_array, class_num])
            original_images.append(RGB_img)
        except Exception as e:
            print("File corrupted: " + img)


SPECIAL_DATADIR = "./test_data"
special_data = []
original_images = []

load_test_images()

special_X = []
special_Y = []

for features, label in special_data:
  special_X.append(features)
  special_Y.append(label)

special_X = np.array(special_X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
special_Y = np.array(special_Y)

special_X = special_X/255.0

special_pred = model.predict(special_X, batch_size=32)

for i in range(len(special_pred)):
  (img, label) = special_data[i]
  colored_image = original_images[i]
  chance = special_pred[i]
  pred_gender = 0 if chance < 0.5 else 1

  display_prediction(colored_image, label, chance, pred_gender)

# How to save data?
# Save each image with the prediction and chance
# Save prediction to txt file as well

