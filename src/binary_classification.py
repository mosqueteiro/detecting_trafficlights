from math import ceil
import numpy as np
import pandas as pd
np.random.seed(1337)  # for reproducibility

import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.utils import np_utils
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from sklearn.model_selection import train_test_split

from trafficlight_data import load_binary_train


def rgb_AlexNet(input_shape):
    model = Sequential()
    model.add(Conv2D(96, 7, strides=2, input_shape=(*input_shape, 3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2))
    # 100x100 -> 28x28

    model.add(Conv2D(128, 5, strides=1, padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2))
    # 28x28 -> 19x19

    model.add(Conv2D(192, 3, strides=2))
    model.add(Activation('relu'))
    # 19x19 -> 9x9

    model.add(Conv2D(192, 3, strides=1, padding='same'))
    model.add(Activation('relu'))

    model.add(Conv2D(128, 3, strides=1, padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))

    model.compile('adam', loss='binary_crossentropy', metrics=['acc'])

    return model



if __name__ == "__main__":
    # allow gpu memory growth as needed
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    ###################################
    X, y = load_binary_train()
    label = ['not_tl','traffic_light']
    y_label = y.apply(lambda i: label[1] if i == 1 else label[0])
    mask_tl = y == 1
    df = X.join(y_label)
    undersample = np.random.choice(df[~mask_tl].index, size=5000, replace=False)
    df_sample = df.loc[mask_tl].append(df.loc[undersample])
    # X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample,
                                                        # test_size=0.10)
    '''Train parameters'''
    batch_size = 20
    target_size = (100,100)

    # add callbacks
    tensorBoard = TensorBoard(log_dir='../tb_log', histogram_freq=2,
                              batch_size=batch_size, write_graph=True,
                              write_grads=False, write_images=True)
    modelCheckpoint = ModelCheckpoint('../models/model_epoch{epoch:02d}.hdf5',
                                      verbose=1, period=1)

    '''Data Generators'''
    df_datagen = ImageDataGenerator(shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       validation_split=0.10)
    train_generator = df_datagen.flow_from_dataframe(
                        df_sample,
                        x_col='local_path',
                        y_col='category',
                        target_size=target_size,
                        color_mode='rgb',
                        class_mode='binary',
                        batch_size=batch_size,
                        subset='training'
                        )
    test_generator = df_datagen.flow_from_dataframe(
                        df_sample,
                        x_col='local_path',
                        y_col='category',
                        target_size=target_size,
                        color_mode='rgb',
                        class_mode='binary',
                        batch_size=batch_size,
                        subset='validation'
                        )

    '''Model creation and training'''
    model = rgb_AlexNet(target_size)
    model.fit_generator(
        train_generator,
        steps_per_epoch=ceil(len(df) / batch_size),
        epochs=10,
        verbose=1,
        validation_data=test_generator,
        validation_steps=ceil(len(df)*0.1 / batch_size),
        callbacks=[modelCheckpoint]
    )

    score = model.evaluate_generator(test_generator,
                                    len(test_generator),
                                    verbose=0)
    print('Test score: {}\tTest accuracy: {}'.format(*score))
