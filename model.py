import tensorflow as tf

tf.config.experimental.set_visible_devices([], 'GPU')

from keras.layers import Dense, Flatten, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

from transformations import Compose, Crop, Resize, Random_Gamma, Random_HFlip, Random_Rotate, Random_Shear
from data_utils import data_generator, get_data_infor


def create_model():
    # Model from this paper:  https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
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

    # The five fully connected layers, ReLU activation
    model.add(Dense(1164, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    return model


if __name__ == '__main__':
    model = create_model()
    model.summary()

    number_of_epochs = 5
    learning_rate = 1e-3
    batch_size = 128

    model.compile(optimizer=Adam(learning_rate), loss="mse")

    # create two generators for training and validation
    train_transformations = Compose([
        Random_Rotate(rotation_angle_limit=15, p=0.5),
        Random_Shear(shear_range=20, p=0.5),
        Random_HFlip(p=0.5),
        Random_Gamma(p=0.5),
    ], p=1.0)

    samples = get_data_infor()

    train_samples, validation_samples = train_test_split(samples, test_size=0.2, shuffle=True, random_state=2020)
    # Some useful constants

    train_gen = data_generator(train_samples, transformations=train_transformations, batch_size=batch_size)
    validation_gen = data_generator(validation_samples, transformations=None, batch_size=batch_size)

    history = model.fit_generator(train_gen,
                                  steps_per_epoch=len(train_samples) // batch_size,
                                  epochs=number_of_epochs,
                                  validation_data=validation_gen,
                                  workers=8,
                                  use_multiprocessing=True,
                                  validation_steps=len(validation_samples) // batch_size,
                                  verbose=1)

    # finally save our model and weights
    model.save('model.h5')
