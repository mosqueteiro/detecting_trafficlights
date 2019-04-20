from math import ceil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time


import tensorflow as tf
from keras.utils import np_utils
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from sklearn.model_selection import train_test_split

from trafficlight_data import load_binary_train
from image_processing import ImageProcessor
from models import like_AlexNet



def traingen_model(model, train_gen, test_gen, steps, **kwargs):
    '''
    '''
    kw = {
        'epochs':5,
        'initial_epoch': 0,
        'callbacks': [],
        'verbose': 1
    }
    kw.update(kwargs)
    history = model.fit_generator(
        train_gen,
        steps_per_epoch=steps,
        validation_data=test_gen,
        **kw
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

    '''Training parameters'''
    batch_size = 100
    val_split = 0.10
    target_size = (100,100)
    epochs = 200
    initial_epoch = 0
    seed = 1337
    color = 'rgb'


    '''Loading and Splitting Images'''
    train_args = {
        'dataset': 'train2017',
        'host': 'pg_serv',
        'user': 'postgres',
        'data_dir': '../data/coco/'
    }
    X, y = load_binary_train(**train_args)
    label = ('not_tl','traffic_light')
    y_label = y.apply(lambda i: label[1] if i == 1 else label[0])
    mask_tl = y == 1
    df = X.join(y_label)
    undersample = np.random.choice(df[~mask_tl].index, size=5000, replace=False)
    df_balanced = df.loc[mask_tl].append(df.loc[undersample])
    ttsplit = {'test_size':val_split, 'random_state':seed}
    df_train, df_test = train_test_split(df_balanced, **ttsplit)
    y_test = df_test.category
    imgProc = ImageProcessor(df_test.local_path)
    imgProc.resize_imgs(target_size)
    X_test = imgProc.images


    '''Data Generators'''
    df_datagen = ImageDataGenerator(shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   validation_split=0.0)
    gen_ops = {
        'x_col': 'local_path',
        'y_col': 'category',
        'target_size': target_size,
        'color_mode': 'rgb',
        'class_mode': 'binary',
        'batch_size': batch_size,
        'seed': seed,
    }
    train_generator = df_datagen.flow_from_dataframe(
                        df_train,
                        # subset='training',
                        **gen_ops
                        )
    # test_generator = df_datagen.flow_from_dataframe(
    #                     df_test,
    #                     # subset='validation',
    #                     **gen_ops
    #                     )

    '''Model creation and training'''
    # Hyperparameters for model
    hyper = {
        'lr': 0.03,
    }
    _ = {'rgb':3, 'gray':1}
    input_shape = (*target_size, _.get(color, 3))
    model = like_AlexNet(input_shape, **hyper)

    # add callbacks
    tb_log = '../tb_logs/binary_lAN_{}_lr{}_{}'.format(color,
                                                       hyper['lr'],
                                                       time())
    tensorBoard = TensorBoard(log_dir=tb_log,
                              histogram_freq=1,
                              batch_size=batch_size,
                              write_graph=True,
                              write_grads=False,
                              write_images=True)
    modelCheckpoint = ModelCheckpoint(
        '../models/binary_epoch{epoch:02d}.hdf5',
        verbose=1, period=1
    )

    # training
    steps = ceil(len(df_balanced)*(1-val_split) / batch_size)
    val_steps = ceil(len(df_balanced)*val_split / batch_size)
    history = traingen_model(model, train_generator, (X_test, y_test), steps,
                epochs=epochs, initial_epoch=initial_epoch,
                callbacks=[tensorBoard]
                )

    score = model.evaluate_generator(test_generator,
                                    len(test_generator),
                                    verbose=0)
    print('Test score: {:.3f}\tTest accuracy: {:.3f}'.format(*score))
