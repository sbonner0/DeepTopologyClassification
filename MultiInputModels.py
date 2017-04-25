# Stephen Bonner 2016 - Durham University
# Keras deep feedforward models for multiclass classification

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

def createModel1H(optimizer='rmsprop', init='glorot_uniform'):

    model = Sequential()
    model.add(Dense(32, input_dim=54, kernel_initializer=init))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(5, kernel_initializer=init))
    model.add(Activation('softmax'))

    # Compile the model
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def createModel2H(optimizer='rmsprop', init='glorot_uniform'):
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
    model.add(Dense(5, kernel_initializer=init))
    model.add(Activation('softmax'))

    # Compile the model
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def createMode3HlWithDropout(optimizer='rmsprop', init='glorot_uniform'):
    # Create the model
    model = Sequential()
    # Input layer plus first hidden layer
    model.add(Dense(1000, input_dim=54, kernel_initializer=init))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    # Second layer
    model.add(Dense(512, kernel_initializer=init))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    # Third layer
    model.add(Dense(256, kernel_initializer=init))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    # Four layer
    model.add(Dense(32, kernel_initializer=init))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    # Add output layer
    model.add(Dense(5, kernel_initializer=init))
    model.add(Activation('softmax'))

    # Compile the model
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model
