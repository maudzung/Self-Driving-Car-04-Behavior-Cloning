import os

import tensorflow as tf

tf.config.experimental.set_visible_devices([], 'GPU')

from keras.layers import Dense, Flatten, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from transformations import Compose, Random_Brightness, Resize, Crop, Convert_RGB2YUV, Random_HFlip, Random_Rotate, \
    Random_Shear
from data_utils import data_generator, get_data_infor


def create_model():
    model = Sequential()

    # Crop the original image
    model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))

    # Normalize data
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))

    # Convolutional layers, ReLU activation
    model.add(Convolution2D(24, kernel_size=(5, 5), strides=(2, 2), activation='relu'))  # 5x5 kernel, 2x2 stride
    model.add(Convolution2D(36, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(Convolution2D(48, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(Convolution2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))  # 3x3 kernel, 1x1 stride
    model.add(Convolution2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))

    model.add(Flatten())

    # The 4 fully connected layers, ReLU activation
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    return model


if __name__ == '__main__':
    model = create_model()
    model.summary()

    number_of_epochs = 10
    learning_rate = 1e-3
    batch_size = 64

    model.compile(optimizer=Adam(learning_rate), loss="mse")

    # create two generators for training and validation
    train_transformations = Compose([
        Random_HFlip(p=0.5),
        Random_Brightness(p=0.5),
        # Crop(top=50, bottom=-20, p=1.),
        # Resize(new_size=(200, 66), p=1.),
        # Convert_RGB2YUV(p=1.),
    ], p=1.0)

    samples = get_data_infor()

    train_samples, validation_samples = train_test_split(samples, test_size=0.2, shuffle=True, random_state=2020)

    train_gen = data_generator(train_samples, transformations=None, batch_size=batch_size)
    validation_gen = data_generator(validation_samples, transformations=None, batch_size=batch_size)

    checkpoint = ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5')

    history = model.fit_generator(train_gen,
                                  steps_per_epoch=len(train_samples) // batch_size,
                                  epochs=number_of_epochs,
                                  validation_data=validation_gen,
                                  validation_steps=len(validation_samples) // batch_size,
                                  callbacks=[checkpoint],
                                  verbose=1)
    model.save('model.h5')

    output_images_dir = './images'
    if not os.path.isdir(output_images_dir):
        os.makedirs(output_images_dir)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot(history.history['loss'])
    ax.plot(history.history['val_loss'])
    ax.set_title('Loss values')
    ax.set_ylabel('loss')
    ax.set_xlabel('epoch')
    ax.legend(['train', 'validation'], loc='upper left')
    plt.savefig(os.path.join(output_images_dir, 'train_val_process.jpg'))
