from math import ceil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(1337)  # for reproducibility

import tensorflow as tf
from keras import optimizers
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.utils import np_utils
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from sklearn.model_selection import train_test_split

from trafficlight_data import load_binary_train


def rgb_AlexNet(input_shape, **kwargs):
    # Hyperparameters
    optim = {
        'lr':kwargs.get('lr', 0.001),
        'beta_1':kwargs.get('beta_1', 0.9),
        'beta_2':kwargs.get('beta_2', 0.999),
        'epsilon':kwargs.get('epsilon', None),
        'decay':kwargs.get('decay', 0.0)
    }


    model = Sequential()
    model.add(Conv2D(48, 7, strides=2, input_shape=(*input_shape, 3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2))
    # model.add(Dropout(0.5))
    # 100x100 -> 28x28

    model.add(Conv2D(128, 5, strides=1, padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2))
    # 28x28 -> 19x19

    model.add(Conv2D(192, 3, strides=2))
    model.add(Activation('relu'))
    # 19x19 -> 9x9
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))
    adam = optimizers.Adam(**optim)
    model.compile(adam, loss='binary_crossentropy', metrics=['acc'])

    return model

def train_model(model, train_gen, test_gen, steps, val_steps, **kwargs):
    '''
    '''
    kw = {
        'epochs':5,
        'initial_epoch': 0,
        'callbacks': []
    }
    kw.update(kwargs)
    history = model.fit_generator(
        train_gen,
        steps_per_epoch=steps,
        epochs=kw['epochs'],
        verbose=1,
        validation_data=test_gen,
        validation_steps=val_steps,
        callbacks=kw['callbacks'],
        initial_epoch=kw['initial_epoch']
    )
    return history

def plot_acc(history, fig=None):
    metrics = {'acc':[], 'val_acc':[]}
    for hist in history:
        for k,v in hist.history.items():
            metrics[k] = metrics.get(k, []) + v
    if isinstance(fig, type(plt.figure())):
        ax = fig.axes[0]
    else:
        fig, ax = plt.subplots(figsize=(10,8))
    ax.plot(metrics['acc'], label='train')
    ax.plot(metrics['val_acc'], label='test')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy per Epoch')
    fig.show()


if __name__ == "__main__":
    # allow gpu memory growth as needed
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth=True
    # sess = tf.Session(config=config)
    ###################################
    train_args = {'dataset':'train2017',
		  'host':'pg_serv',
		  'user':'postgres',
		  'data_dir':'../data/coco/'
		 }
    X, y = load_binary_train(**train_args)
    label = ['not_tl','traffic_light']
    y_label = y.apply(lambda i: label[1] if i == 1 else label[0])
    mask_tl = y == 1
    df = X.join(y_label)
    undersample = np.random.choice(df[~mask_tl].index, size=5000, replace=False)
    df_sample = df.loc[mask_tl].append(df.loc[undersample])

    '''Train parameters'''
    batch_size = 100
    val_split = 0.10
    steps = ceil(len(df_sample)*(1-val_split) / batch_size)
    val_steps = ceil(len(df_sample)*val_split / batch_size)
    target_size = (100,100)
    epochs = 200
    initial_epoch = 0

    # add callbacks
    tensorBoard = TensorBoard(log_dir='../tb_log', histogram_freq=2,
                              batch_size=batch_size, write_graph=True,
                              write_grads=False, write_images=True)
    modelCheckpoint = ModelCheckpoint(
        '../models/binary_epoch{epoch:02d}.hdf5',
        verbose=1, period=1
    )

    '''Data Generators'''
    df_datagen = ImageDataGenerator(shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       validation_split=val_split)
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
    model = rgb_AlexNet(target_size, lr=0.3, decay=0.01)
    # model = load_model('../models/binary_epoch1_half_size.hdf5')
    # history = model.fit_generator(
    #     train_generator,
    #     steps_per_epoch=ceil(len(df) / batch_size),
    #     epochs=epochs,
    #     verbose=1,
    #     validation_data=test_generator,
    #     validation_steps=ceil(len(df)*0.1 / batch_size),
    #     # callbacks=[modelCheckpoint],
    #     initial_epoch=initial_epoch
    # )
    history = train_model(model, train_generator, test_generator, steps,
                val_steps=val_steps, epochs=epochs, initial_epoch=initial_epoch)

    score = model.evaluate_generator(test_generator,
                                    len(test_generator),
                                    verbose=0)
    print('Test score: {:.3f}\tTest accuracy: {:.3f}'.format(*score))
