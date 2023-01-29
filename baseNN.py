# first neural network with keras tutorial
from numpy import loadtxt, random, binary_repr
import numpy as np
from tensorflow.keras.models import Sequential, model_from_json, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.metrics import binary_accuracy
import tensorflow as tf
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint
from configManager import getConfig, createConfig
from bitarray import bitarray

config = getConfig()

################ ОПИСАНИЕ НЕЙРОННОЙ СЕТИ НАЧАЛО ######################
# Здесь описывается сама нейронная сеть, можно играться, добавлять слои нейронов,
# изменять функции и прочее.
def prepareModel():
    # define the keras model
    model = Sequential()
    model.add(Dense(1024, input_shape=(15,), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])#metrics=[binary_accuracy])
    plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
    return model
################ ОПИСАНИЕ НЕЙРОННОЙ СЕТИ КОНЕЦ ######################

def reshapeDataLine(input_string):
    window_size = config["window_size"]
    n = window_size - 1

    numbers = tf.io.decode_csv(input_string, record_defaults=[0.0]*window_size, field_delim=',')

    private = numbers[n]
    public = numbers[0:n]
    
    return (public, private)

################ ОБУЧЕНИЕ НЕЙРОННОЙ СЕТИ КОНЕЦ ######################
# Здесь нейронная сеть обучается и сохраняется, если показатели улучшились.
def trainModel(model):
    checkpoint = ModelCheckpoint(config["model_file_name"], monitor='loss', verbose=1, save_best_only=True, mode='min')

    # Define the CSV file path
    csv_file = config["train"]["file_name"]

    # Load the data from the CSV file
    batch_size = 1
    ds = tf.data.TextLineDataset(csv_file)
    ds = ds.map(reshapeDataLine).batch(batch_size)
#     for p in ds.as_numpy_iterator():
#         print(p)
    
    model.fit(ds, epochs=1, verbose=1, callbacks=[checkpoint])

    return model
################ ОБУЧЕНИЕ НЕЙРОННОЙ СЕТИ КОНЕЦ ######################

def saveModel(model):
    model.save(config["model_file_name"])
    print('Saved model to disk')


def loadModel():
    model = load_model(config["model_file_name"])
    model.summary()
    return model
