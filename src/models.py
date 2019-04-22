import tensorflow as tf
from keras import optimizers
from keras.models import Sequential, Model, load_model
from keras.initializers import he_uniform, glorot_uniform
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization
from keras.layers.convolutional import Conv2D



def like_AlexNet(input_shape, **kwargs):
    # Hyperparameters
    optim = {
        'lr':kwargs.get('lr', 0.001),
        'beta_1':kwargs.get('beta_1', 0.9),
        'beta_2':kwargs.get('beta_2', 0.999),
        'epsilon':kwargs.get('epsilon', None),
        'decay':kwargs.get('decay', 0.0)
    }
    # Model structure parameters
    activation = kwargs.get('activation', 'relu')
    seed = kwargs.get('seed', 1)
    initializer = kwargs.get('initializer', he_uniform(seed))
    conv_act = kwargs.get('conv_act', activation)
    dense_act = kwargs.get('dense_act', activation)
    dropout_rate = kwargs.get('dropout_rate', 0.5)

    model = Sequential()
    model.add(Conv2D(48, 7, strides=2, input_shape=input_shape,
                     activation=conv_act,
                     kernel_initializer=initializer))
    model.add(Dropout(rate=dropout_rate))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2))
    # 100x100 -> 28x28

    model.add(Conv2D(128, 5, strides=1, padding='same',
                     activation=conv_act,
                     kernel_initializer=initializer))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2))
    # 28x28 -> 19x19

    model.add(Conv2D(192, 3, strides=2,
                     activation=conv_act,
                     kernel_initializer=initializer))
    # 19x19 -> 9x9
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(2048, activation=dense_act, kernel_initializer=initializer))
    model.add(Dropout(rate=dropout_rate))

    model.add(Dense(2048, activation=dense_act, kernel_initializer=initializer))
    model.add(Dropout(rate=dropout_rate))

    model.add(Dense(1, activation='sigmoid'))
    adam = optimizers.Adam(**optim)
    model.compile(adam, loss='binary_crossentropy', metrics=['acc'])

    return model


def AlexNet_model(input_shape, **kwargs):
    # Hyperparameters
    optim = {
        'lr':kwargs.get('lr', 0.001),
        'beta_1':kwargs.get('beta_1', 0.9),
        'beta_2':kwargs.get('beta_2', 0.999),
        'epsilon':kwargs.get('epsilon', None),
        'decay':kwargs.get('decay', 0.0)
    }
    # Model structure parameters
    activation = kwargs.get('activation', 'relu')
    seed = kwargs.get('seed', 1)
    initializer = kwargs.get('initializer', he_uniform(seed))
    conv_act = kwargs.get('conv_act', activation)
    dense_act = kwargs.get('dense_act', activation)
    dropout_rate = kwargs.get('dropout_rate', 0.5)

    model = Sequential()
    model.add(Conv2D(96, 7, strides=2, input_shape=input_shape,
                     activation=conv_act,
                     kernel_initializer=initializer))
    model.add(Dropout(rate=dropout_rate))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2))

    # 100x100 -> 28x28

    model.add(Conv2D(256, 5, strides=1, padding='same',
                     activation=conv_act,
                     kernel_initializer=initializer))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2))
    # 28x28 -> 19x19

    model.add(Conv2D(384, 3, strides=1, padding='same',
                     activation=conv_act,
                     kernel_initializer=initializer))

    model.add(Conv2D(384, 3, strides=1, padding='same',
                     activation=conv_act,
                     kernel_initializer=initializer))

    model.add(Conv2D(256, 3, strides=2,
                     activation=conv_act,
                     kernel_initializer=initializer))
    # 19x19 -> 9x9
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(4096, activation=dense_act, kernel_initializer=initializer))
    model.add(Dropout(rate=dropout_rate))

    model.add(Dense(4096, activation=dense_act, kernel_initializer=initializer))
    model.add(Dropout(rate=dropout_rate))

    model.add(Dense(1, activation='sigmoid'))
    adam = optimizers.Adam(**optim)
    model.compile(adam, loss='binary_crossentropy', metrics=['acc'])

    return model
