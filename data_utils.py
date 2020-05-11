import os
import csv

import cv2
import numpy as np
import sklearn


def get_data_infor():
    working_dir = './'
    img_dir = os.path.join(working_dir, 'data', 'IMG')
    driving_log_path = './data/driving_log.csv'
    samples = []
    STEERING_CORRECTION = 0.22
    with open(driving_log_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            center_angle = float(line[3])
            # Center
            center_img_path = os.path.join(img_dir, line[0].split('/')[-1])
            angle = center_angle
            samples.append([center_img_path, angle])
            # Left
            left_img_path = os.path.join(img_dir, line[1].split('/')[-1])
            angle = center_angle + STEERING_CORRECTION
            samples.append([left_img_path, angle])
            # Right
            right_img_path = os.path.join(img_dir, line[2].split('/')[-1])
            angle = center_angle - STEERING_CORRECTION
            samples.append([right_img_path, angle])

    return samples


def data_generator(samples, transformations=None, batch_size=32):
    num_samples = len(samples)

    while 1:  # Loop forever so the generator never terminates
        np.random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]
            images = []
            angles = []
            for sub_sample in batch_samples:
                img_path = sub_sample[0]
                angle = float(sub_sample[1])
                img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)  # BGR --> RGB

                if transformations is not None:
                    img, angle = transformations(img, angle)

                images.append(img)
                angles.append(angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)

            yield sklearn.utils.shuffle(X_train, y_train)
