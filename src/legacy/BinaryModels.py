# Stephen Bonner 2016 - Durham University
# Keras deep feedforward models for binary classification

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import Utils as ut
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.models import model_from_yaml
from sklearn import cross_validation


def binaryModel1h(optimizer='rmsprop', init='glorot_uniform'):

    # Create a binary model
    model = Sequential()
    model.add(Dense(64, input_dim=54, kernel_initializer=init))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    # Single output layer using sigmoid activation
    model.add(Dense(1, kernel_initializer=init))
    model.add(Activation('sigmoid'))

    # Compile the model
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model

def binaryModel2h(optimizer='rmsprop', init='glorot_uniform'):

    # Create the model
    model = Sequential()

    # Input layer plus first hidden layer
    model.add(Dense(32, input_dim=54, kernel_initializer=init))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    # Add second hidden layer
    model.add(Dense(16, kernel_initializer=init))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    # Add output layer
    model.add(Dense(1, kernel_initializer=init))
    model.add(Activation('sigmoid'))

    # Compile the model
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model

def binaryModel3h(optimizer='rmsprop', init='glorot_uniform'):
    # Create the model
    model = Sequential()
    # Input layer plus first hidden layer
    model.add(Dense(64, input_dim=54, kernel_initializer=init))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    # Second layer
    model.add(Dense(32, kernel_initializer=init))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    # Add third hidden layer
    model.add(Dense(16, kernel_initializer=init))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    # Add third hidden layer
    model.add(Dense(16, kernel_initializer=init))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    # Add third hidden layer
    model.add(Dense(16, kernel_initializer=init))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    # Add output layer
    model.add(Dense(1, kernel_initializer=init))
    model.add(Activation('sigmoid'))

    # Compile the model
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model
