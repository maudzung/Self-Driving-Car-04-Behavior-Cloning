import os
import csv

import cv2
import numpy as np
import sklearn


def data_generator(samples, img_dir=None, transformations=None, batch_size=32, STEERING_CORRECTION=0.2):
    num_samples = len(samples)

    while 1:  # Loop forever so the generator never terminates
        np.random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]
            images = []
            angles = []
            for b_sample in batch_samples:
                center_angle = float(b_sample[3])

                # random_number = np.random.randint(0, 3)
                random_number=0
                if random_number == 0:
                    # Center
                    center_img_path = os.path.join(img_dir, b_sample[0].split('/')[-1])
                    img = cv2.cvtColor(cv2.imread(center_img_path), cv2.COLOR_BGR2RGB)  # BGR --> RGB
                    angle = center_angle
                elif random_number == 1:
                    # Left
                    left_img_path = os.path.join(img_dir, b_sample[1].split('/')[-1])
                    img = cv2.cvtColor(cv2.imread(left_img_path), cv2.COLOR_BGR2RGB)  # BGR --> RGB
                    angle = center_angle + STEERING_CORRECTION
                else:
                    # Right
                    right_img_path = os.path.join(img_dir, b_sample[2].split('/')[-1])
                    img = cv2.cvtColor(cv2.imread(right_img_path), cv2.COLOR_BGR2RGB)  # BGR --> RGB
                    angle = center_angle - STEERING_CORRECTION
                if transformations is not None:
                    img, angle = transformations(img, angle)
                images.append(img)
                angles.append(angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            # y_train = np.array(angles).reshape(len(angles), 1)

            yield sklearn.utils.shuffle(X_train, y_train)


if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from transformations import Compose, Crop, Resize, Random_Gamma, Random_HFlip, Random_Rotate, Random_Shear

    train_transformations = Compose([
        Crop(top_percent=0.35, bottom_percent=0.10, p=1.0),
        Random_Rotate(rotation_angle_limit=15, p=0.5),
        Random_Shear(shear_range=20, p=0.5),
        Random_HFlip(p=0.5),
        Random_Gamma(p=0.5),
        Resize(new_size=(64, 64), p=1.),
    ], p=1)

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
    STEERING_CORRECTION = 0.23

    # data_generator(train_samples, transformations=None, batch_size=32, STEERING_CORRECTION=STEERING_CORRECTION)
