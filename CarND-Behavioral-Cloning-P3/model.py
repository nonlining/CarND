#-------------------------------------------------------------------------------
# Name:        model.py
# Purpose:     This for Behavioral-Cloning project.
#
# Author:      Min-Jung Wang
#
# Created:     01/08/2017
# Copyright:   (c) Min-Jung Wang 2017
# Licence:     <your licence>
#-------------------------------------------------------------------------------
from os import path
import sys

import csv
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import cv2
from os import listdir
from os.path import isfile, join , isdir


def cnn_model():

    model = Sequential()
    model.add(Lambda(lambda x: (x/127.5) - 1.0, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25),(0,0))))

    model.add(Convolution2D(32,3,3,activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.1))
    model.add(Convolution2D(64,3,3,activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.1))
    model.add(Convolution2D(128,3,3, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(256,3,3, activation='relu'))
    model.add(MaxPooling2D())

    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(20))
    model.add(Dense(1))

    return model


def data_augmentation(images, angles):
    augmented_images = []
    augmented_angles = []

    for image, angle in zip(images, angles):

        augmented_images.append(image)
        augmented_angles.append(angle)

        flipped_image = cv2.flip(image,1)
        flipped_angle = -1.0 * angle
        augmented_images.append(flipped_image)
        augmented_angles.append(flipped_angle)
    return augmented_images, augmented_angles



def pre_processing(data, file_path):


    center_images = list(map(lambda x: file_path+x[0][2:], (i for i in data)))
    left_images = list(map(lambda x: file_path+x[1][2:], (i for i in data)))
    right_images = list(map(lambda x: file_path+x[2][2:], (i for i in data)))
    angles = list(map(float, (i[3] for i in data)))


    for i in range(len(data)):
        data[i][0] = center_images[i]
        data[i][1] = left_images[i]
        data[i][2] = right_images[i]
        data[i][3] = angles[i]


    return data


def generator(samples, batch_size=32, train_flag = True):
    num_samples = len(samples)
    correction = 0.2

    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_path = batch_sample[0]

                center_image = cv2.imread(center_path)

                center_angle = batch_sample[3]

                images.append(center_image)
                angles.append(center_angle)

                if train_flag:
                    left_path = batch_sample[1]
                    right_path = batch_sample[2]
                    left_image = cv2.imread(left_path)
                    right_image = cv2.imread(right_path)
                    images.append(left_image)
                    angles.append(center_angle + correction)
                    images.append(right_image)
                    angles.append(center_angle - correction)

            images, angles = data_augmentation(images, angles)

            X_train = np.array(images)
            y_train = np.array(angles)

            yield shuffle(X_train, y_train)


def main():

    file_path = "."
    num_epoch = 10

    csv_lines = []

    csvfiles = [f for f in listdir(file_path+ "\\Data\\") if isfile(join(file_path + "\\Data\\",f))]
    for c in csvfiles:
        print(c)
        with open(file_path + "\\Data\\"+c) as f:
            content = csv.reader(f)
            for line in content:
                csv_lines.append(line)

    data = pre_processing(csv_lines, file_path)
    train_data, valid_data = train_test_split(data, test_size=0.2)
    train_generator = generator(train_data, batch_size=32)
    validation_generator = generator(valid_data, batch_size=32, train_flag=False)

    model = cnn_model()

    #model.summary()

    model.compile(loss='mse', optimizer='adam')
    model.fit_generator(train_generator, samples_per_epoch= len(train_data)*6 , validation_data=validation_generator, nb_val_samples=len(valid_data), nb_epoch= num_epoch)

    model.save('model.h5')






if __name__ == '__main__':
    main()
