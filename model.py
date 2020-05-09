import os
import csv

import tensorflow as tf
from keras.layers import Dense, Flatten, Lambda, Activation, MaxPooling2D, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from transformations import Compose, Crop, Resize, Random_Gamma, Random_HFlip, Random_Rotate, Random_Shear
from data_utils import data_generator

activation_relu = 'relu'
# Our model is based on NVIDIA's "End to End Learning for Self-Driving Cars" paper
# Source:  https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
model = Sequential()

# Remove 50 top pixels and 20 down ones. None on the sides.
model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))

# Normalize data
model.add(Lambda(lambda x: (x / 255.0) - 0.5))

# starts with five convolutional and maxpooling layers
model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation(activation_relu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation(activation_relu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation(activation_relu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
model.add(Activation(activation_relu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
model.add(Activation(activation_relu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Flatten())

# Next, five fully connected layers
model.add(Dense(1164))
model.add(Activation(activation_relu))

model.add(Dense(100))
model.add(Activation(activation_relu))

model.add(Dense(50))
model.add(Activation(activation_relu))

model.add(Dense(10))
model.add(Activation(activation_relu))

model.add(Dense(1))

model.summary()

number_of_epochs = 10
learning_rate = 1e-4
batch_size = 64

model.compile(optimizer=Adam(learning_rate), loss="mse")

# create two generators for training and validation


train_transformations = Compose([
    # Crop(top_percent=0.35, bottom_percent=0.10, p=1.0),
    Random_Rotate(rotation_angle_limit=15, p=0.5),
    Random_Shear(shear_range=20, p=0.5),
    Random_HFlip(p=0.5),
    Random_Gamma(p=0.5),
    # Resize(new_size=(64, 64), p=1.0),
], p=1.0)

val_transformations = Compose([
    # Crop(top_percent=0.35, bottom_percent=0.10, p=1.0),
    # Random_Rotate(rotation_angle_limit=15, p=0.5),
    # Random_Shear(shear_range=20, p=0.5),
    # Random_HFlip(p=0.5),
    # Random_Gamma(p=0.5),
    Resize(new_size=(64, 64), p=1.0),
], p=1.0)

working_dir = './'
img_dir = os.path.join(working_dir, 'record_data', 'IMG')
driving_log_path = './record_data/driving_log.csv'
samples = []
with open(driving_log_path) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2, shuffle=True, random_state=2020)
# Some useful constants
STEERING_CORRECTION = 0.22

train_gen = data_generator(train_samples, img_dir=img_dir, transformations=train_transformations, batch_size=batch_size,
                           STEERING_CORRECTION=STEERING_CORRECTION)

validation_gen = data_generator(validation_samples, img_dir=img_dir, transformations=None,
                                batch_size=batch_size, STEERING_CORRECTION=STEERING_CORRECTION)

history = model.fit_generator(train_gen,
                              steps_per_epoch=len(train_samples) // batch_size,
                              epochs=number_of_epochs,
                              validation_data=validation_gen,
                              validation_steps=len(validation_samples) // batch_size,
                              verbose=1)

# history = model.fit_generator(train_gen,
#                               samples_per_epoch=number_of_samples_per_epoch,
#                               nb_epoch=number_of_epochs,
#                               validation_data=validation_gen,
#                               nb_val_samples=number_of_validation_samples,
#                               verbose=1)

# finally save our model and weights
model.save('model.h5')
