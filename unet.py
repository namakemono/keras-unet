import os
import glob
import numpy as np
import cv2
import pandas as pd
from keras.optimizers import Adam, SGD
from keras.models import Sequential, Model
from keras.layers import BatchNormalization, Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Input, merge, UpSampling2D
from keras.optimizers import Adam
from keras import backend as K

def preprocess_input(X):
    return np.asarray([((x - np.mean(x)) / np.std(x)) for x in X])

def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def UNet(input_shape, learn_rate=1e-3):
    inputs = Input(input_shape)
    filter_size = 32
    growth_step = 32
    x = BatchNormalization()(inputs)
    conv1 = Convolution2D(filter_size, 3, 3, activation='relu', border_mode='same')(x)
    conv1 = Convolution2D(filter_size, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    pool1 = BatchNormalization()(pool1)
    filter_size += growth_step
    conv2 = Convolution2D(filter_size, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(filter_size, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = BatchNormalization()(pool2)

    filter_size += growth_step
    conv3 = Convolution2D(filter_size, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(filter_size, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = BatchNormalization()(pool3)

    filter_size += growth_step
    conv4 = Convolution2D(filter_size, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(filter_size, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = BatchNormalization()(pool4)

    conv5 = Convolution2D(filter_size, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Convolution2D(filter_size, 3, 3, activation='relu', border_mode='same', name="conv5b")(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2), name="pool5")(conv5)
    pool5 = BatchNormalization()(pool5)

    conv6 = Convolution2D(filter_size, 3, 3, activation='relu', border_mode='same')(pool5)
    conv6 = Convolution2D(filter_size, 3, 3, activation='relu', border_mode='same', name="conv6b")(conv6)

    up6 = UpSampling2D(size=(2, 2), name="up6")(conv6)
    up6 = merge([up6, conv5], mode='concat', concat_axis=3)
    up6 = BatchNormalization()(up6)

    filter_size -= growth_step
    conv66 = Convolution2D(filter_size, 3, 3, activation='relu', border_mode='same')(up6)
    conv66 = Convolution2D(filter_size, 3, 3, activation='relu', border_mode='same')(conv66)

    up7 = merge([UpSampling2D(size=(2, 2))(conv66), conv4], mode='concat', concat_axis=3)
    up7 = BatchNormalization()(up7)

    filter_size -= growth_step
    conv7 = Convolution2D(filter_size, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Convolution2D(filter_size, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv3], mode='concat', concat_axis=3)
    up8 = BatchNormalization()(up8)
    filter_size -= growth_step
    conv8 = Convolution2D(filter_size, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Convolution2D(filter_size, 3, 3, activation='relu', border_mode='same')(conv8)


    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv2], mode='concat', concat_axis=3)
    up9 = BatchNormalization()(up9)
    conv9 = Convolution2D(filter_size, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Convolution2D(filter_size, 3, 3, activation='relu', border_mode='same')(conv9)

    up10 = UpSampling2D(size=(2, 2))(conv9)
    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(up10)

    model = Model(input=inputs, output=conv10)
    # model.compile(optimizer=SGD(lr=learn_rate, momentum=0.9, nesterov=True), loss=dice_coef_loss, metrics=[dice_coef])
    model.compile(optimizer="nadam", loss=dice_coef_loss, metrics=[dice_coef])

    return model

if __name__ == "__main__":
    model = UNet(input_shape=(320, 320, 1))
    print model.summary()
