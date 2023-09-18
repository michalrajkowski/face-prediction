import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
# import dlib
import time
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

def build_model():
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

  return model
######

def display_prediction(img, chance, pred_cat):
  plt.imshow(img)
  prd_gender = "Predicted: " + CATEGORIES[int(pred_cat)]
  prd_percent = "with " + str(chance) +" prob."
  plt.title( "{}\n{}".format( prd_gender, prd_percent))
  save_path = os.path.join(OUTPUT_IMG_DIR, str(time.time()) + '.png')
  plt.savefig(save_path)

def create_prediction_plot(face_image, prediction_chane, prediction_cathegory, image_name):
  plt.imshow(face_image)
  prd_gender_string = "Predicted: " + prediction_cathegory
  prd_percent_string = "with " + str(prediction_chane) +"% prob."
  plt.title( "{}\n{}\n{}".format(image_name ,prd_gender_string, prd_percent_string))

def load_test_images(special_data, original_images, image_names):
    path = INPUT_IMG_DIR
    for img in os.listdir(path):
        try:
            image_names.append(img)
            color_image = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
            RGB_img = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            class_num = CATEGORIES[0]
            special_data.append([new_array, class_num])
            original_images.append(RGB_img)
        except Exception as e:
            print("File corrupted: " + img)
# Create model
model = build_model()

# Load special data and make predictions
SPECIAL_DATADIR = "./test_data"
special_data = []
original_images = []
image_names = []

load_test_images(special_data, original_images, image_names)

special_X = []
special_Y = []

for features, label in special_data:
  special_X.append(features)
  special_Y.append(label)

special_X = np.array(special_X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
special_Y = np.array(special_Y)

special_X = special_X/255.0

special_pred = model.predict(special_X, batch_size=32)

# Prepare results
gender_chance = []
predicted_cathegories = []
# calculate predictions:

for i in range(len(special_pred)):
  chance = special_pred[i][0]
  predicted_gender = 0 if chance < 0.5 else 1
  perc = 1.0 - chance if chance < 0.5 else chance
  perc = round(perc*100.0, 2)
  gender_chance.append(perc)
  predicted_cathegories.append(CATEGORIES[predicted_gender])

# Save predictions as images 
for i in range(len(special_pred)):
  predicted_cat = predicted_cathegories[i]
  colored_image = original_images[i]
  certainity_percent = gender_chance[i]
  image_name = image_names[i]
  create_prediction_plot(colored_image, certainity_percent, predicted_cat, image_name)
  save_path = os.path.join(OUTPUT_IMG_DIR, image_name)
  plt.savefig(save_path)

# Save predictions as txt file
output_txt_file = os.path.join(OUTPUT_IMG_DIR, 'predictions.txt')

with open(output_txt_file, 'w') as txt_file:
    for i in range(len(special_pred)):
        certainty_percent = gender_chance[i]
        image_name = image_names[i]
        predicted_cat = predicted_cathegories[i]
        txt_file.write(f"{image_name} {predicted_cat} {certainty_percent}\n")

txt_file.close()